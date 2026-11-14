"""Bounded paired-key acquisition diagnostic for shared CLMM checkpoints.

Paired and duplicate-query arms share the same episode contexts and update,
answer and byte budgets. Both queries are independent full-prefix forwards;
neither sees the other's label. Saved stage checkpoints are descriptive, while
the final step is the primary endpoint. This is not a replay-method comparison.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import fact_revision_experiment as common
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import build_model, load_checkpoint, save_checkpoint, reset_memory_at_segments
from clmm.pretrain.fact_streams import make_episode, encode_example


def episode_for_seed(seed, facts, revision_fraction, split='train', family='seen'):
    """Condition two-fact contexts on distinct initial values, using rejection.

    Every candidate is drawn from the existing generator; no labels/records are
    edited to create inconsistent episodes. Query wording is fixed across keys.
    """
    for attempt in range(100):
        episode = make_episode(seed + attempt * 10_000_000_019, facts, 0, split, family,
                               revision_fraction, include_recent=False)
        if len({r.value for r in episode.events[:facts]}) != facts:
            continue
        template = episode.queries[0].template_index
        return replace(episode, queries=tuple(replace(q, template_index=template) for q in episode.queries))
    raise RuntimeError('Distinct-value episode rejection limit exceeded')


def make_batch(step, total_steps, batch_size, train_seed, arm):
    if (type(batch_size) is not int or batch_size < 2 or batch_size % 2
            or type(total_steps) is not int or total_steps < 8
            or type(step) is not int or not 0 <= step < total_steps
            or arm not in ('paired', 'duplicate')):
        raise ValueError('Require even batch >=2, steps >=8, valid step and paired/duplicate arm')
    # Per-step RNG isolates the episode stream from arm-specific query usage.
    rng = np.random.default_rng(np.random.SeedSequence([train_seed, step]))
    fraction = step / total_steps
    if fraction < .125:
        stage, facts, revision = 'k1', 1, 0.
    elif fraction < .75:
        stage, facts, revision = 'k2', 2, 0.
    else:
        stage = 'mixed_revision'
        facts, revision = ((1, 0.), (2, 0.), (1, 1.), (2, .5))[int(rng.integers(4))]
    seeds = rng.integers(1, 1_000_000_000, size=batch_size//2)
    selected = rng.integers(0, facts, size=batch_size//2)
    episodes = [episode_for_seed(int(seed), facts, revision) for seed in seeds]
    indices, examples = [], []
    for ep, chosen in zip(episodes, selected):
        pair = (0, 1) if arm == 'paired' and facts == 2 else (int(chosen), int(chosen))
        indices.extend(pair)
        examples.extend(encode_example(ep, ep.queries[index], 'online') for index in pair)
    x = np.stack([np.frombuffer(item[0], dtype=np.uint8) for item in examples])
    y = np.array([item[1] for item in examples], dtype=np.uint8)
    plan = dict(step=step+1, stage=stage, facts=facts, revision_fraction=revision,
                candidate_seeds=seeds.tolist(), episodes=[ep.to_dict() for ep in episodes],
                query_indices=indices, duplicate_choices=selected.tolist(),
                input_length=x.shape[1])
    return x, y, plan


@torch.no_grad()
def strict_probe(model, args, out, device, deadline):
    """Fresh key-only query switches, all distinct-value episodes, both ranks."""
    rows, prefixes, episodes = [], [], []
    for family in ('seen', 'heldout'):
        # Family shares semantic seeds. No test keys are accessed.
        rng = np.random.default_rng(args.dev_seed + 500_021)
        for i, seed in enumerate(rng.integers(3_100_000_000, 3_200_000_000, args.strict_episodes)):
            ep = episode_for_seed(int(seed), 2, 0., 'dev', family)
            episodes.append(dict(family=family, episode=i, semantic=ep.to_dict()))
            for condition in ('online', 'reset_memory'):
                for rank, query in enumerate(ep.queries):
                    prefix, target, _ = encode_example(ep, query, 'online')
                    rows.append(dict(family=family, episode=i, condition=condition, rank=rank,
                                     key=query.key, target=target, prefix=prefix.decode('ascii')))
                    prefixes.append(prefix)
    # Group conditions explicitly so an ablation cannot affect the wrong rows.
    original = {k:v.detach().clone() for k,v in model.state_dict().items()}
    all_logits = np.empty((len(rows), 256), dtype=np.float32)
    for condition in ('online', 'reset_memory'):
        indices = [i for i,r in enumerate(rows) if r['condition'] == condition]
        context = reset_memory_at_segments(model) if condition == 'reset_memory' else nullcontext()
        with context:
            for start in range(0, len(indices), args.eval_batch_size):
                common.before_deadline(deadline)
                ix = indices[start:start+args.eval_batch_size]
                x = torch.tensor([list(prefixes[i]) for i in ix], device=device)
                all_logits[ix] = model(x)[:, -1].cpu().numpy()
    if not all(torch.equal(v, model.state_dict()[k]) for k,v in original.items()):
        raise AssertionError('Strict probe changed checkpoint weights')
    if not np.isfinite(all_logits).all():
        raise RuntimeError('Nonfinite strict logits')
    predictions = all_logits.argmax(-1)
    for i,row in enumerate(rows):
        row.update(prediction=int(predictions[i]), correct=int(predictions[i] == row['target']))
    common.write_csv(out/'strict_predictions.csv', rows)
    np.save(out/'strict_logits.npy', all_logits)
    common.json_file(out/'strict_episodes.json', episodes)
    summary = []
    for family in ('seen','heldout'):
        for condition in ('online','reset_memory'):
            subset = [r for r in rows if r['family']==family and r['condition']==condition]
            pairs = [subset[i:i+2] for i in range(0,len(subset),2)]
            summary.append(dict(family=family, condition=condition, queries=len(subset), episodes=len(pairs),
                accuracy=float(np.mean([r['correct'] for r in subset])),
                first_accuracy=float(np.mean([pair[0]['correct'] for pair in pairs])),
                second_accuracy=float(np.mean([pair[1]['correct'] for pair in pairs])),
                both_correct=float(np.mean([all(r['correct'] for r in pair) for pair in pairs])),
                prediction_changes=float(np.mean([pair[0]['prediction'] != pair[1]['prediction'] for pair in pairs]))))
    common.json_file(out/'strict_summary.json', summary)
    return summary


def checkpoint_and_evaluate(model, config, optimizer, args, out, step, meta, device, deadline):
    folder = out/f'step-{step:04d}'
    folder.mkdir()
    info = dict(task='fact-paired-acquisition', parent_checkpoint_sha256=meta['checkpoint_sha256'],
                parent_init_seed=meta['parent_init_seed'], steps=step, arm=args.arm,
                train_seed=args.train_seed, lr=args.lr, optimizer='AdamW', weight_decay=.01,
                gradient_clip=1., batch_size=args.batch_size, fixed_final_checkpoint=(step==args.steps))
    digest = save_checkpoint(folder/'checkpoint.pt', model, config,
        source_info=dict(repo_commit=meta['repo_commit'], source_sha256=meta['source_sha256']), training_info=info)
    torch.save(optimizer.state_dict(), folder/'optimizer.pt')
    evaluation = folder/'evaluation'
    evaluation.mkdir()
    model.eval()
    datasets = common.build_evaluation(args, evaluation)
    evaluated = common.evaluate(model, datasets, args, evaluation, device, deadline)
    strict = strict_probe(model, args, evaluation, device, deadline)
    common.json_file(folder/'evaluation_metadata.json', dict(checkpoint_sha256=digest, step=step,
        **evaluated, strict_summary=strict, selection='Fixed stage endpoint; no best-checkpoint selection.'))
    model.train()
    return dict(step=step, checkpoint_sha256=digest, evaluation=evaluated, strict_summary=strict)


def run(args):
    if (args.steps < 8 or args.steps % 8 or args.batch_size < 2 or args.batch_size % 2
            or min(args.gate_episodes,args.strict_episodes,args.eval_batch_size,args.max_seconds,args.threads)<1
            or not np.isfinite(args.lr) or args.lr<=0):
        raise ValueError('Positive limits, even batch, steps multiple of8, positive finite lr required')
    out, device, meta = common.initialize(args)
    # Common initialization captures its own reusable runner plus all model files.
    name = Path(__file__).relative_to(REPO).as_posix()
    (out/'sources'/name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__,out/'sources'/name)
    meta['source_sha256'][name] = common.sha256(out/'sources'/name)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    model, optimizer, rows, plans, chunks, targets, offsets = None,None,[],[],[],[],[0]
    try:
        model, config, payload = load_checkpoint(args.checkpoint,map_location='cpu')
        init_seed = payload['training_info'].get('init_seed')
        if (init_seed not in (68101,68105) or payload['training_info'].get('task') is not None
                or payload['training_info'].get('parent_checkpoint_sha256') is not None
                or (config.segment_len,config.memory_segment_len,config.memory_batch_size)!=(32,8,32)):
            raise ValueError('Require one of the two original shared parent checkpoints')
        meta['parent_init_seed'] = init_seed
        if args.segment_len != config.segment_len:
            config = replace(config,segment_len=args.segment_len)
            with torch.random.fork_rng(devices=[]):
                changed = build_model(config)
            changed.load_state_dict(model.state_dict(),strict=True)
            model = changed
        model.to(device)
        meta['parameter_count'] = sum(p.numel() for p in model.parameters())
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},out/'initial_state.pt')
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
        stages = {args.steps//8, args.steps*3//4, args.steps}
        checkpoints = []
        for step in range(args.steps):
            common.before_deadline(deadline)
            x_np,y_np,plan = make_batch(step,args.steps,args.batch_size,args.train_seed,args.arm)
            x,y = torch.tensor(x_np.astype(np.int64),device=device),torch.tensor(y_np.astype(np.int64),device=device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[:, -1]
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite training logits')
            loss = F.cross_entropy(logits,y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            common.sync(device)
            row = dict(step=step+1,stage=plan['stage'],facts=plan['facts'],revision_fraction=plan['revision_fraction'],
                input_length=x.shape[1],input_bytes=x.numel(),supervised_answers=len(y),
                loss=float(loss.detach()),accuracy=float((logits.argmax(-1)==y).float().mean()),
                gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-started)
            rows.append(row); plans.append(plan); chunks.append(x_np.reshape(-1)); targets.append(y_np)
            offsets.append(offsets[-1]+x_np.size)
            with (out/'training.csv').open('a',newline='',encoding='utf8') as f:
                import csv
                writer=csv.DictWriter(f,fieldnames=row.keys())
                if step==0: writer.writeheader()
                writer.writerow(row)
            if step==0 or (step+1)%100==0: print(json.dumps(row),flush=True)
            if step+1 in stages:
                checkpoints.append(checkpoint_and_evaluate(model,config,optimizer,args,out,step+1,meta,device,deadline))
                meta['checkpoints'] = checkpoints
                common.json_file(out/'metadata.json',meta)
        meta.update(status='succeeded',completed_steps=len(rows),
            input_bytes=sum(r['input_bytes'] for r in rows),supervised_answers=len(rows)*args.batch_size)
        for name,digest in meta['source_sha256'].items():
            if common.sha256(out/'sources'/name)!=digest or common.sha256(REPO/name)!=digest:
                raise AssertionError(f'Source changed during run: {name}')
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc(),completed_steps=len(rows))
        if model is not None: torch.save(model.state_dict(),out/'partial_state.pt')
        raise
    finally:
        if chunks:
            np.savez_compressed(out/'training_inputs.npz',input_bytes=np.concatenate(chunks),
                step_offsets=np.array(offsets,dtype=np.int64),targets=np.stack(targets))
            common.json_file(out/'training_plan.json',plans)
        source_changed = None
        for name,digest in meta['source_sha256'].items():
            if common.sha256(out/'sources'/name)!=digest or common.sha256(REPO/name)!=digest:
                meta['status']='failed'
                meta['source_changed']=name
                source_changed=name
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        if device.type=='cuda': meta['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        common.json_file(out/'metadata.json',meta)
        if source_changed:
            raise AssertionError(f'Source changed during run: {source_changed}')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.set_defaults(mode='paired_acquisition',suite='gate',grid_episodes=32,primary_episodes=128,block_size=4,buffer_size=8)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--design',type=Path)
    p.add_argument('--arm',choices=('paired','duplicate'),required=True)
    p.add_argument('--segment-len',type=int,choices=(32,128),default=32)
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--train-seed',type=int,default=89102)
    p.add_argument('--dev-seed',type=int,default=89103)
    p.add_argument('--gate-episodes',type=int,default=64)
    p.add_argument('--strict-episodes',type=int,default=128)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-seconds',type=int,default=600)
    return p


if __name__=='__main__':
    run(parser().parse_args())
