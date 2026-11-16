"""Fixed-size digit versus atomic-key binding experiment (no replay pressure)."""
from __future__ import annotations
import argparse, csv, gzip, json, os, shutil, subprocess, sys, time, traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import load_checkpoint, save_checkpoint, reset_memory_at_segments, zero_memory_output
from clmm.pretrain.key_binding import make_codec, training_worlds, training_arrays, evaluation_worlds, make_samples, ENCODINGS
from clmm.pretrain.sequence_freshness_data import collect_histories
from scripts.sequence_freshness_experiment import state_sha, parent_model, verify_packet
from scripts.freshness_experiment import capture
from scripts import fact_revision_experiment as common


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    codec = make_codec()
    worlds = training_worlds(args.steps, args.contexts, args.train_seed, codec)
    arrays = training_arrays(worlds, codec)
    np.savez_compressed(out / 'training.npz', **arrays)
    with gzip.open(out / 'training_plan.json.gz', 'wt', encoding='utf8') as f:
        json.dump(worlds, f)
    common.json_file(out / 'key_codec.json', codec.to_dict())
    supplied = json.loads(args.exclusions.read_text(encoding='utf8'))
    if (not isinstance(supplied, list) or any(type(s) is not str or len(s) != 64 for s in supplied)):
        raise ValueError('Expected frozen signature union')
    seen = set(supplied)
    manifest = [dict(path=str(args.exclusions), sha256=common.sha256(args.exclusions), histories=len(seen))]
    for packet in args.exclude_packet:
        for filename in ('training_plan.json.gz', 'episodes.json'):
            path = packet / filename
            with (gzip.open(path, 'rt', encoding='utf8') if path.suffix == '.gz' else path.open(encoding='utf8')) as f:
                ids = collect_histories(json.load(f))
            seen.update(ids)
            manifest.append(dict(path=str(path), sha256=common.sha256(path), histories=len(ids)))
    own = collect_histories(worlds)
    seen.update(own)
    common.json_file(out / 'excluded_signatures.json', sorted(seen))
    common.json_file(out / 'exclusion_manifest.json', dict(inputs=manifest, current_training_unique=len(own), union_count=len(seen)))
    data = evaluation_worlds(seen, codec, args.dev_seed, args.episodes)
    common.json_file(out / 'episodes.json', data)
    common.json_file(out / 'samples.json', make_samples(data, codec))
    (out / 'parents').mkdir()
    parents = []
    for seed in (68101, 68105):
        origin = args.parents / f'lm-{seed}/checkpoint.pt'
        target = out / 'parents' / f'{seed}.pt'
        shutil.copy2(origin, target)
        model, config, payload = load_checkpoint(target, map_location='cpu')
        assert sum(p.numel() for p in model.parameters()) == 839248
        assert payload['training_info']['init_seed'] == seed and payload['training_info'].get('task') is None
        assert (config.segment_len, config.memory_segment_len, config.memory_batch_size) == (32, 8, 32)
        from dataclasses import asdict
        parents.append(dict(seed=seed, path=target.relative_to(out).as_posix(), sha256=common.sha256(target),
                            state_sha256=state_sha(model), config=asdict(config), original_training=payload['training_info']))
    common.json_file(out / 'parents.json', parents)
    if args.design:
        shutil.copy2(args.design, out / 'design.json')
    sources = capture(out)
    names = ('training.npz', 'training_plan.json.gz', 'key_codec.json', 'episodes.json', 'samples.json',
             'parents.json', 'excluded_signatures.json', 'exclusion_manifest.json')
    if args.design:
        names += ('design.json',)
    meta = dict(status='succeeded', mode='prepare', source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                source_sha256=sources, files_sha256={n: common.sha256(out / n) for n in names},
                steps=args.steps, contexts=args.contexts, episodes=args.episodes, train_seed=args.train_seed, development_seed=args.dev_seed,
                training_answers=args.steps * args.contexts * 2, training_input_bytes=int(arrays['lengths'].sum()) * args.contexts * 2,
                case_evaluation_vectors=args.episodes * 40, command=[sys.executable, *sys.argv],
                environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__), elapsed_seconds=time.perf_counter() - start)
    common.json_file(out / 'metadata.json', meta)
    print(json.dumps({k: meta[k] for k in ('status', 'steps', 'episodes', 'training_answers', 'training_input_bytes', 'case_evaluation_vectors', 'elapsed_seconds')}), flush=True)


@torch.no_grad()
def evaluate(model, args, out, deadline):
    out.mkdir()
    tick = time.perf_counter()
    model.eval()
    before = state_sha(model)
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(args.device).clone() if args.device.startswith('cuda') else None
    rows = [r for r in json.loads((args.packet / 'samples.json').read_text()) if r['encoding'] == args.encoding]
    logits = np.empty((len(rows), 256), dtype=np.float32)
    resources, buckets = [], {}
    for i, row in enumerate(rows):
        buckets.setdefault((row['condition'], row['input_bytes']), []).append(i)
    for (condition, length), indices in buckets.items():
        scope = reset_memory_at_segments(model) if condition == 'reset_memory' else zero_memory_output(model) if condition == 'retrieval_zero' else nullcontext()
        with scope:
            for start in range(0, len(indices), args.eval_batch):
                common.before_deadline(deadline)
                ix = indices[start:start + args.eval_batch]
                t = time.perf_counter()
                x = torch.tensor([list(bytes.fromhex(rows[i]['prefix_hex'])) for i in ix], device=args.device)
                y = torch.tensor([rows[i]['target'] for i in ix], device=args.device)
                result = model(x)[:, -1]
                scores = {k: v.cpu().tolist() for k, v in common.score_logits(result, y).items()}
                logits[ix] = result.cpu().numpy()
                for j, i in enumerate(ix):
                    rows[i].update(score_index=i, **{k: v[j] for k, v in scores.items()}, correct=int(scores['prediction'][j] == rows[i]['target']))
                common.sync(torch.device(args.device))
                resources.append(dict(condition=condition, input_length=length, row_indices=json.dumps(ix), elapsed_seconds=time.perf_counter() - t))
    assert before == state_sha(model) and torch.equal(cpu, torch.random.get_rng_state())
    if cuda is not None:
        assert torch.equal(cuda, torch.cuda.get_rng_state(args.device))
    parity = {}
    for i, row in enumerate(rows):
        if not row['group'].startswith('query_only') and row['condition'] != 'retrieval_zero':
            parity.setdefault((row['group'], row['episode'], row['rank']), {})[row['condition']] = i
    assert all(np.array_equal(logits[v['normal']], logits[v['reset_memory']]) for v in parity.values())
    np.save(out / 'answer_logits.npy', logits)
    common.write_csv(out / 'predictions.csv', rows)
    common.write_csv(out / 'resources.csv', resources)
    summaries = []
    for group, condition in dict.fromkeys((r['group'], r['condition']) for r in rows):
        selected = [r for r in rows if (r['group'], r['condition']) == (group, condition)]
        pairs = {}
        for row in selected:
            pairs.setdefault(row['episode'], []).append(row)
        assert all(len(pair) == 2 for pair in pairs.values())
        summaries.append(dict(group=group, condition=condition, episodes=len(pairs),
            accuracy=float(np.mean([r['correct'] for r in selected])), both_correct=float(np.mean([all(r['correct'] for r in p) for p in pairs.values()])),
            prediction_changes=float(np.mean([p[0]['prediction'] != p[1]['prediction'] for p in pairs.values()])),
            nll=float(np.mean([r['nll'] for r in selected])), legal_probability_mass=float(np.mean([r['legal_probability_mass'] for r in selected]))))
    common.write_csv(out / 'summary.csv', summaries)
    return dict(status='succeeded', queries=len(rows), elapsed_seconds=time.perf_counter() - tick, pure_checkpoint=True,
                RNG_unchanged=True, normal_reset_parity=True, logits_sha256=common.sha256(out / 'answer_logits.npy'))


def execute(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    deadline = start + args.max_seconds
    packet = verify_packet(args.packet)
    model, config, parent = parent_model(args)
    meta = dict(status='running', mode=args.mode, parent_seed=args.parent, segment=128, encoding=args.encoding,
                source_commit=packet['source_commit'], data_sha256=packet['files_sha256'], parent=parent, parameter_count=839248,
                initial_state_sha256=state_sha(model), command=[sys.executable, *sys.argv], started_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=args.device,
                    device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU', dtype='float32', tf32=False, deterministic=True, threads=args.threads))
    common.json_file(out / 'metadata.json', meta)
    logs = []
    try:
        if args.mode in ('train', 'preflight'):
            with np.load(args.packet / 'training.npz') as archive:
                inputs, targets, lengths = archive[args.encoding], archive['targets'], archive['lengths']
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
            train_start = time.perf_counter()
            for i in range(packet['steps']):
                common.before_deadline(deadline)
                n = int(lengths[i])
                x = torch.tensor(inputs[i, :, :n].astype(np.int64), device=args.device)
                y = torch.tensor(targets[i].astype(np.int64), device=args.device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                result = model(x)[:, -1]
                loss = F.cross_entropy(result, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                memory_grads = [p.grad.detach().float().square().sum() for name, p in model.named_parameters()
                                if name.startswith('layers.') and '.4.' in name and p.grad is not None]
                memory_norm = torch.stack(memory_grads).sum().sqrt() if memory_grads else loss.new_tensor(0.)
                assert torch.isfinite(memory_norm)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                common.sync(torch.device(args.device))
                prediction = result.detach().argmax(-1).reshape(-1, 2)
                correct = (prediction == y.reshape(-1, 2)).float()
                row = dict(step=i + 1, stage='k1' if n == 63 else 'k2', loss=float(loss.detach()), accuracy=float(correct.mean()),
                    first_accuracy=float(correct[:, 0].mean()), second_accuracy=float(correct[:, 1].mean()),
                    prediction_changes=float((prediction[:, 0] != prediction[:, 1]).float().mean()),
                    gradient_norm=float(norm), memory_gradient_norm=float(memory_norm), answers=len(y), input_bytes=x.numel(), elapsed_seconds=time.perf_counter() - train_start)
                logs.append(row)
                with (out / 'training.csv').open('a', newline='', encoding='utf8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    if i == 0:
                        writer.writeheader()
                    writer.writerow(row)
                if i == 0 or (i + 1) % 100 == 0:
                    print(json.dumps(row), flush=True)
                if i + 1 == 1000:
                    save_checkpoint(out / 'midpoint.pt', model, config, source_info=dict(source_commit=packet['source_commit']),
                                    training_info=dict(task='key-binding-midpoint', steps=1000, selection='Recovery only, fixed terminal2000 primary'))
                    torch.save(optimizer.state_dict(), out / 'midpoint-optimizer.pt')
            torch.save(optimizer.state_dict(), out / 'optimizer.pt')
            meta.update(completed_steps=len(logs), training_elapsed_seconds=time.perf_counter() - train_start,
                supervised_answers=sum(r['answers'] for r in logs), training_input_bytes=sum(r['input_bytes'] for r in logs),
                memory_nonzero_gradient_steps=sum(r['memory_gradient_norm'] > 0 for r in logs))
        else:
            meta.update(completed_steps=0, training_elapsed_seconds=0, supervised_answers=0, training_input_bytes=0)
        meta['checkpoint_sha256'] = save_checkpoint(out / 'checkpoint.pt', model, config,
            source_info=dict(source_commit=packet['source_commit'], source_sha256=packet['source_sha256']),
            training_info=dict(task='key-binding', parent_init_seed=args.parent, parent_checkpoint_sha256=parent['sha256'],
                               steps=meta['completed_steps'], encoding=args.encoding, segment=128, fixed_terminal=True))
        meta['final_state_sha256'] = state_sha(model)
        sample = next(r for r in json.loads((args.packet / 'samples.json').read_text()) if r['encoding'] == args.encoding)
        probe = torch.tensor([list(bytes.fromhex(sample['prefix_hex']))], device=args.device)
        model.eval()
        with torch.no_grad():
            expected = model(probe)[:, -1].cpu()
        loaded, _, _ = load_checkpoint(out / 'checkpoint.pt', map_location=args.device)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(probe)[:, -1].cpu()
        assert torch.equal(expected, actual)
        meta['reload_max_error'] = float((expected - actual).abs().max())
        del loaded
        meta['evaluation'] = evaluate(model, args, out / 'evaluation', deadline)
        verify_packet(args.packet)
        meta['status'] = 'succeeded'
    except BaseException as error:
        meta.update(status='failed', failure=repr(error), traceback=traceback.format_exc(), completed_steps=len(logs))
        raise
    finally:
        meta.update(elapsed_seconds=time.perf_counter() - start, ended_utc=datetime.now(timezone.utc).isoformat())
        if args.device.startswith('cuda'):
            meta['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
        common.json_file(out / 'metadata.json', meta)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'initial', 'preflight', 'train'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--packet', type=Path)
    p.add_argument('--parents', type=Path)
    p.add_argument('--exclusions', type=Path)
    p.add_argument('--exclude-packet', action='append', type=Path, default=[])
    p.add_argument('--design', type=Path)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--contexts', type=int, default=8)
    p.add_argument('--train-seed', type=int, default=91402)
    p.add_argument('--dev-seed', type=int, default=91423)
    p.add_argument('--episodes', type=int, default=256)
    p.add_argument('--parent', type=int, choices=[68101, 68105], default=68101)
    p.add_argument('--segment', type=int, choices=[128], default=128)
    p.add_argument('--encoding', choices=ENCODINGS, default='digits')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--eval-batch', type=int, default=16)
    p.add_argument('--max-seconds', type=int, default=1800)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(91402)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (prepare if args.mode == 'prepare' else execute)(args)
