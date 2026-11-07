"""
Quick behavioural checks for two unofficial Titans implementations.

    python verify_titans.py --repo lucidrains --path /path/to/titans-pytorch
    python verify_titans.py --repo neuranox   --path /path/to/titans-memory

Checks
  1. causality      : perturb token j; outputs at positions < j must not change.
  2. cross-segment  : perturb a token early in the sequence; outputs far beyond
                      the attention receptive field should change ONLY if the
                      long-term memory actually carries information forward.
  3. scan-consistency (Neuranox, CUDA only): GPU log-space scan vs exact scan.

Everything runs on CPU in float32 with tiny models; it takes seconds.
A causal model gives a max |delta| of exactly 0 (or ~1e-7) before position j.
"""

import argparse
import sys

import torch

TOL = 1e-5


def max_delta(a, b):
    return (a - b).abs().max().item() if a.numel() else 0.0


def perturb(ids, j, vocab):
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j] + 1) % vocab
    return ids2


@torch.no_grad()
def causality(name, logits_fn, vocab, seq_len, positions):
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (1, seq_len))
    base = logits_fn(ids)
    worst = 0.0
    for j in positions:
        out = logits_fn(perturb(ids, j, vocab))
        leak = max_delta(out[0, :j], base[0, :j])
        worst = max(worst, leak)
        print(f"  [{name}] perturb pos {j:3d}: max|delta| on earlier positions = {leak:.3e}")
    verdict = "PASS (causal)" if worst < TOL else "FAIL (future information leaks backwards)"
    print(f"  [{name}] causality: {verdict}\n")


@torch.no_grad()
def cross_segment(name, logits_fn, vocab, seq_len, j, far_start):
    torch.manual_seed(1)
    ids = torch.randint(0, vocab, (1, seq_len))
    base = logits_fn(ids)
    out = logits_fn(perturb(ids, j, vocab))
    d = max_delta(out[0, far_start:], base[0, far_start:])
    verdict = "memory carries information forward" if d > TOL else "NO long-range effect (memory not carrying state)"
    print(f"  [{name}] perturb pos {j}, max|delta| at positions >= {far_start}: {d:.3e} -> {verdict}\n")


def run_lucidrains():
    from titans_pytorch import MemoryAsContextTransformer, NeuralMemory

    V, S = 64, 16  # vocab, segment length

    # standalone neural memory
    torch.manual_seed(0)
    mem = NeuralMemory(dim=16, chunk_size=4).eval()
    seq = torch.randn(1, 48, 16)
    with torch.no_grad():
        base, _ = mem(seq)
        worst = 0.0
        for j in (5, 8, 13, 30):
            seq2 = seq.clone()
            seq2[0, j] += 1.0
            out, _ = mem(seq2)
            worst = max(worst, max_delta(out[0, :j], base[0, :j]))
    print(f"  [NeuralMemory] causality max|delta| = {worst:.3e} -> {'PASS' if worst < TOL else 'FAIL'}\n")

    configs = {
        "MAC-default (memory residual + segmented attn)": dict(neural_mem_gate_attn_output=False, sliding_window_attn=False),
        "MAG-like (gate attn output, sliding window)": dict(neural_mem_gate_attn_output=True, sliding_window_attn=True),
    }
    for name, kw in configs.items():
        torch.manual_seed(0)
        model = MemoryAsContextTransformer(
            num_tokens=V, dim=32, depth=2, segment_len=S,
            num_persist_mem_tokens=2, num_longterm_mem_tokens=2,
            neural_memory_layers=(1, 2), **kw,
        ).eval()
        fn = lambda ids, m=model: m(ids)
        causality(name, fn, V, 96, positions=(3, 15, 16, 40, 70))
        # attention reach: block-diagonal = 1 segment; sliding = ~2 windows per layer
        cross_segment(name, fn, V, 160, j=2, far_start=120)


def run_neuranox():
    from titans import NeuralMemory, TitansMAC, TitansMAG, TitansMAL, TitansLMM

    V = 64
    common = dict(vocab_size=V, d_model=32, n_layers=2, mem_layers=2,
                  n_persistent=2, chunk_size=8, max_seq_len=512)

    torch.manual_seed(0)
    mem = NeuralMemory(d_model=16, n_layers=2, chunk_size=4).eval()
    seq = torch.randn(1, 48, 16)
    with torch.no_grad():
        base = mem(seq)
        worst = 0.0
        for j in (5, 8, 13, 30):
            seq2 = seq.clone()
            seq2[0, j] += 1.0
            worst = max(worst, max_delta(mem(seq2)[0, :j], base[0, :j]))
    print(f"  [NeuralMemory] causality max|delta| = {worst:.3e} -> {'PASS' if worst < TOL else 'FAIL'}\n")

    models = {
        "MAC": TitansMAC(**common),
        "MAG": TitansMAG(window=8, **common),
        "MAL": TitansMAL(window=8, **common),
        "LMM": TitansLMM(**common),
    }
    for name, model in models.items():
        model.eval()
        fn = lambda ids, m=model: m(ids)["logits"]
        causality(name, fn, V, 96, positions=(3, 7, 8, 20, 50))
        # MAC attention is segment-local; SWA reach is ~window * layers
        cross_segment(name, fn, V, 160, j=2, far_start=120)

    if torch.cuda.is_available():
        from titans.ops.scan import _scan_parallel_log, _scan_sequential
        for T in (32, 64, 128, 256):
            eta = torch.rand(2, T, 8, device="cuda") * 0.9 + 0.05
            u = torch.randn(2, T, 8, device="cuda")
            d = max_delta(_scan_parallel_log(eta, u), _scan_sequential(eta, u))
            print(f"  [scan] T={T:3d}: GPU log-space vs exact max|delta| = {d:.3e}")
    else:
        print("  [scan] CUDA not available; skipped GPU scan check")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", choices=["lucidrains", "neuranox"], required=True)
    p.add_argument("--path", required=True, help="path to the cloned repository root")
    args = p.parse_args()
    sys.path.insert(0, args.path)
    torch.set_grad_enabled(False)
    print(f"== {args.repo} ==\n")
    (run_lucidrains if args.repo == "lucidrains" else run_neuranox)()
