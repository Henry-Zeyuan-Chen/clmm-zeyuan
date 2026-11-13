"""Frozen-reader replay timing diagnostic; no training or checkpoint modification.

Both replay arms consume the identical ordered snapshot once per block.  The
snapshot is the preceding eight *observations*, never previous replay writes.
Only memory weights are available to the reader at the evaluated boundaries.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


PREFILL = BLOCK_SIZE = 8
BLOCKS = 8
EVAL_BLOCKS = (2, 4, 8)
ALGORITHMS = ("online", "continuous", "periodic")


def make_episodes(n, seed, nkeys=16, nvals=8):
    """Generate all streams on CPU before batching, so batch size changes no data."""
    if n < 1 or nkeys < 3 or nvals < 2:
        raise ValueError("Need positive episodes, at least 3 keys and 2 values")
    g = torch.Generator().manual_seed(seed)
    revised = torch.randint(nkeys, (n,), generator=g)
    raw = torch.randint(nkeys - 1, (n,), generator=g)
    stable = raw + (raw >= revised).long()
    lower, upper = torch.minimum(revised, stable), torch.maximum(revised, stable)
    keys = torch.randint(nkeys - 2, (n, PREFILL + BLOCKS * BLOCK_SIZE), generator=g)
    keys += (keys >= lower[:, None]).long()
    keys += (keys >= upper[:, None]).long()
    values = torch.randint(nvals, keys.shape, generator=g)
    old = torch.randint(nvals, (n,), generator=g)
    new = (old + torch.randint(1, nvals, (n,), generator=g)) % nvals
    keys[:, 0], values[:, 0] = revised, old
    keys[:, 1] = stable
    keys[:, 4], values[:, 4] = revised, new
    assert torch.all(old != new)
    assert torch.all(keys[:, PREFILL:] != revised[:, None])
    assert torch.all(keys[:, PREFILL:] != stable[:, None])
    return keys, values


def block_events(algorithm, block):
    """Return (kind, source-observation index); replay queue stores indices only."""
    if algorithm not in ALGORITHMS or not 1 <= block <= BLOCKS:
        raise ValueError("Unknown algorithm or block outside 1..8")
    start = PREFILL + (block - 1) * BLOCK_SIZE
    bank = tuple(range(start - BLOCK_SIZE, start))
    incoming = tuple(range(start, start + BLOCK_SIZE))
    if algorithm == "continuous":
        events = [item for pair in zip(
            (("observation", i) for i in incoming),
            (("replay", i) for i in bank)) for item in pair]
    else:
        events = [("observation", i) for i in incoming]
        if algorithm == "periodic":
            events += [("replay", i) for i in bank]
    return bank, events


def assert_evicted(keys, bank, block, pending_replays=()):
    """Check both raw-data stores and the pending replay queue before querying."""
    end = PREFILL + block * BLOCK_SIZE
    fifo = tuple(range(end - BLOCK_SIZE, end))
    targets = keys[:, [0, 1]]
    assert not pending_replays, "Query attempted before replay queue was exhausted"
    for indices in (bank, fifo):
        resident = keys[:, list(indices)]
        assert not torch.any(resident[:, :, None] == targets[:, None, :]), (
            "A query target is still accessible in a replay bank or observation FIFO")


def read_frozen(model, queries, fast):
    """Pass only query keys and fast weights to the original Reader.read method."""
    snapshot = {name: tensor.clone() for name, tensor in fast.items()}
    predictions = []
    for query in queries:
        logits = model.read(query, fast)
        assert torch.isfinite(logits).all(), "Nonfinite query logits"
        predictions.append(logits.argmax(dim=-1))
    assert all(torch.equal(snapshot[name], tensor) for name, tensor in fast.items()), (
        "Reader.read mutated the fast memory")
    return predictions


@torch.no_grad()
def evaluate_batch(model, keys, values, algorithm, *, replay_decay_scale=1):
    """Evaluate one paired stream batch, resetting fast weights for this arm."""
    if replay_decay_scale not in (0, 1):
        raise ValueError("replay_decay_scale must be 0 or 1")
    if keys.shape != values.shape or keys.shape[1] != PREFILL + BLOCKS * BLOCK_SIZE:
        raise ValueError("Expected matching [batch,72] key/value streams")
    fast = model.memory.init_weights(keys.shape[0])
    past = None
    writes = replays = observations = 0
    last_target_write = {}
    rows = []

    def write(index, kind):
        nonlocal fast, past, writes, replays, observations
        fact = torch.stack((model.key(keys[:, index:index + 1]),
                            model.value(values[:, index:index + 1])))
        hook = None
        if kind == "replay" and replay_decay_scale == 0:
            # to_decay_factor emits logits; sigmoid(-inf) = 0 disables only
            # this replay write's multiplicative forgetting. Slow weights and
            # observation-write gates remain untouched.
            hook = model.memory.to_decay_factor.register_forward_hook(
                lambda _module, _inputs, output: torch.full_like(output, -torch.inf))
        try:
            _, returned = model.memory.store_memories(
                fact, weights=fast, past_state=past, seq_index=writes,
                return_surprises=False)
        finally:
            if hook is not None:
                hook.remove()
        past = returned.states
        fast = past[0]
        writes += 1
        replays += kind == "replay"
        observations += kind == "observation"
        if index in (0, 4):
            last_target_write["revised"] = writes - 1
        elif index == 1:
            last_target_write["stable"] = writes - 1

    for index in range(PREFILL):
        write(index, "observation")
    for block in range(1, BLOCKS + 1):
        bank, events = block_events(algorithm, block)
        # This immutable index tuple is a view of an eight-pair snapshot.  It is
        # not extended by new observations or replayed items during the block.
        pending = [i for kind, i in events if kind == "replay"]
        replayed = []
        for kind, index in events:
            write(index, kind)
            if kind == "replay":
                assert pending.pop(0) == index
                replayed.append(index)
        assert replayed == ([] if algorithm == "online" else list(bank))
        assert observations == PREFILL + block * BLOCK_SIZE
        assert replays == (0 if algorithm == "online" else block * BLOCK_SIZE)
        assert writes == observations + replays
        if block not in EVAL_BLOCKS:
            continue
        assert_evicted(keys, bank, block, pending)
        before_read = writes
        predictions = read_frozen(model, (keys[:, 0], keys[:, 1]), fast)
        assert writes == before_read
        for condition, position, last_write, predicted in (
                ("revised", 0, 4, predictions[0]), ("stable", 1, 1, predictions[1])):
            query = keys[:, position].tolist()
            target = values[:, last_write].tolist()
            old = values[:, 0].tolist()
            for episode, prediction in enumerate(predicted.tolist()):
                rows.append(dict(
                    episode=episode, algorithm=algorithm, block=block,
                    condition=condition, target_key=query[episode],
                    target_value=target[episode],
                    old_value=old[episode] if condition == "revised" else "",
                    prediction=prediction, correct=int(prediction == target[episode]),
                    stale_old=int(prediction == old[episode]) if condition == "revised" else "",
                    future_observations=observations - 1 - last_write,
                    total_observations=observations, total_memory_writes=writes,
                    total_replay_writes=replays,
                    last_target_observation_index=last_write,
                    last_target_memory_write_index=last_target_write[condition],
                    future_memory_writes_since_last_target=writes - 1 - last_target_write[condition],
                    target_absent_from_fifo=True, target_absent_from_bank=True,
                    pending_replays=0, local_attention_tokens=0))
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pilot(path):
    spec = importlib.util.spec_from_file_location("clmm_replay_pilot_source", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import pilot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def process_peak_rss_bytes():
    """Whole-process peak RSS/working set, not an algorithm-specific allocation."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    start = time.perf_counter()
    repo, pilot_path, out = args.repo.resolve(), args.pilot.resolve(), args.out.resolve()
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(pilot_path.parent))
    if args.episodes < 1 or args.batch_size < 1 or args.threads not in (1, 2):
        raise ValueError("Positive episodes/batch size and 1 or 2 CPU threads required")
    device = torch.device(args.device)
    # Required by deterministic cuBLAS matrix multiplication on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    pilot = load_pilot(pilot_path)
    keys, values = make_episodes(args.episodes, args.seed, pilot.NKEYS, pilot.NVALS)
    stream_data = json.dumps([keys.tolist(), values.tolist()], separators=(",", ":"))
    source_paths = [Path(__file__).resolve(), pilot_path,
                    repo / "titans_pytorch" / "neural_memory.py",
                    repo / "titans_pytorch" / "memory_models.py"]
    sources = {str(path): sha256(path) for path in source_paths}
    out.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.csv", "episodes.csv", "summary.csv", "metadata.json"):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite {out / name}")
    with (out / "episodes.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "eval_seed", "keys", "values"))
        for episode, (k, v) in enumerate(zip(keys.tolist(), values.tolist())):
            writer.writerow((episode, args.seed, " ".join(map(str, k)), " ".join(map(str, v))))
    counts = defaultdict(lambda: dict(n=0, correct=0, stale_old=0))
    checkpoint_reports = []
    with (out / "predictions.csv").open("w", newline="", encoding="utf8") as handle:
        writer = None
        for checkpoint_index, path in enumerate(args.checkpoints):
            path = path.resolve()
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            config = checkpoint.get("config", {})
            model = pilot.Reader(**config).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            frozen_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            # This assay requires a bare memory reader, with no local attention
            # host or raw-data retrieval path.  Other architectures need a new assay.
            assert set(dict(model.named_children())) == {"key", "value", "memory", "head"}
            assert model.memory.store_chunk_size == 1
            assert model.memory.qkv_receives_diff_views
            checkpoint_hash = sha256(path)
            elapsed_by_arm = defaultdict(float)
            synchronize(device)
            checkpoint_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for offset in range(0, args.episodes, args.batch_size):
                k = keys[offset:offset + args.batch_size].to(device)
                v = values[offset:offset + args.batch_size].to(device)
                for algorithm in ALGORITHMS:
                    synchronize(device)
                    arm_start = time.perf_counter()
                    rows = evaluate_batch(model, k, v, algorithm,
                                          replay_decay_scale=args.replay_decay_scale)
                    synchronize(device)
                    elapsed_by_arm[algorithm] += time.perf_counter() - arm_start
                    for row in rows:
                        row["episode"] += offset
                        row.update(checkpoint_index=checkpoint_index,
                                   checkpoint_path=str(path), checkpoint_sha256=checkpoint_hash,
                                   init_seed=checkpoint.get("init_seed", ""),
                                   train_seed=checkpoint.get("train_seed", ""), eval_seed=args.seed)
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        count = counts[(checkpoint_index, algorithm, row["block"], row["condition"])]
                        count["n"] += 1
                        count["correct"] += row["correct"]
                        count["stale_old"] += row["stale_old"] if row["condition"] == "revised" else 0
                handle.flush()
            synchronize(device)
            assert all(torch.equal(frozen_state[name], tensor) for name, tensor in model.state_dict().items()), (
                "Evaluation mutated a checkpoint parameter or buffer")
            report = dict(path=str(path), sha256=checkpoint_hash, config=config,
                          init_seed=checkpoint.get("init_seed"), train_seed=checkpoint.get("train_seed"),
                          steps=checkpoint.get("steps"),
                          registered_parameters=sum(p.numel() for p in model.parameters()),
                          dtype=str(next(model.parameters()).dtype),
                          checkpoint_parameters_and_buffers_unchanged=True,
                          elapsed_seconds=time.perf_counter() - checkpoint_start,
                          evaluation_seconds_by_algorithm=dict(elapsed_by_arm))
            if device.type == "cuda":
                report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            checkpoint_reports.append(report)
            print(json.dumps(report), flush=True)
            del model, frozen_state
    with (out / "summary.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("checkpoint_index", "algorithm", "block", "condition", "n",
                         "correct", "accuracy", "stale_old", "stale_rate"))
        for (checkpoint_index, algorithm, block, condition), count in sorted(counts.items()):
            assert count["n"] == args.episodes
            revised = condition == "revised"
            writer.writerow((checkpoint_index, algorithm, block, condition, count["n"],
                             count["correct"], count["correct"] / count["n"],
                             count["stale_old"] if revised else "",
                             count["stale_old"] / count["n"] if revised else ""))
    assert all(sha256(path) == digest for path, digest in sources.items()), "Source changed during evaluation"
    metadata = dict(
        command=sys.argv, repo=str(repo), repo_commit=revision, source_sha256=sources,
        design_sha256=sha256(args.design) if args.design else None,
        stream_sha256=hashlib.sha256(stream_data.encode("utf8")).hexdigest(),
        eval_seed=args.seed, episodes=args.episodes, batch_size=args.batch_size,
        torch=str(torch.__version__), python=sys.version, platform=platform.platform(),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=True, cuda_tf32=False,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        nkeys=pilot.NKEYS, nvals=pilot.NVALS,
        expected_uniform_chance=1 / pilot.NVALS,
        algorithms=list(ALGORITHMS), prefill=PREFILL, block_size=BLOCK_SIZE, blocks=BLOCKS,
        eval_blocks=list(EVAL_BLOCKS), observation_fifo_capacity=8,
        immutable_replay_snapshot_capacity=8, replay_arm_raw_pair_capacity=16,
        queue="Indices into immutable snapshot; no extra copied pairs",
        evaluator_holds_full_stream=True,
        model_reads="Query key embedding and fast memory weights only; no local attention, bank or FIFO access",
        replay_selection="All previous 8 observations, chronological order, once per block",
        replay_decay_scale=args.replay_decay_scale,
        replay_decay_intervention=("standard learned decay on all writes" if args.replay_decay_scale == 1
                                  else "replay writes only: decay logits forced to -inf, sigmoid decay=0; observation decay unchanged"),
        no_checkpoint_training=True, checkpoints=checkpoint_reports,
        process_peak_rss_bytes=process_peak_rss_bytes(),
        elapsed_seconds_including_setup_io=time.perf_counter() - start,
        caveats=["Online has fewer memory updates; replay-vs-online is not compute matched.",
                 "Continuous vs periodic differ only in interleaving at evaluated block boundaries.",
                 "Block1 snapshot includes both old and new target facts; later snapshots contain neither target.",
                 "Delays count future observations, not replay updates: revised19/35/67, stable22/38/70.",
                 "Longer streams and repeated keys extend beyond the original short unique-key pilot training distribution.",
                 "Same ordered replay inputs do not imply identical gradients or decay effects.",
                 "Last target replay occurs at different memory-update ages in the two replay schedules.",
                 "Whole-process peak RSS includes imports, evaluator data and output records; not model-only storage."])
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", type=Path, help="Optional predeclared design to hash")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=51131)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replay-decay-scale", type=int, choices=(0, 1), default=1,
                        help="1: standard decay; 0: disable forgetting on replay writes only")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
