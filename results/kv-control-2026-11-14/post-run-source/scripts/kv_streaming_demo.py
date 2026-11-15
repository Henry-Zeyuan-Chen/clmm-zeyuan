"""Verify a trained KV initializer with bounded live replay and session resume.

This is an integration check on one typed history, not an efficacy estimate.
The historical full evaluation remains frozen at ff62df0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from clmm.pretrain.kv_scheduler import StreamingReplay
from clmm.pretrain.kv_streams import episode, replay_plan
from scripts.kv_representation_control import equal_tree, load_checkpoint


def run(checkpoint: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, provenance = load_checkpoint(checkpoint, torch.device('cpu'))
    model.eval().requires_grad_(False)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    ep = episode(90323, facts=8, delay=16, revisions=2, pressure=True)
    queries = torch.tensor([[q['key'] for q in ep['queries']]])
    cpu_rng = torch.random.get_rng_state().clone()
    outcomes = []

    def write_event(state, event):
        return model.write(torch.tensor([event['key']]), torch.tensor([event['value']]), state)

    with torch.no_grad():
        for arm in ('online','repeat','continuous','periodic'):
            state = model.initial_state(1)
            policy = StreamingReplay(arm, ep['seed'])
            emitted = []
            for index, (key, value) in enumerate(zip(ep['keys'], ep['values'])):
                events = [policy.prime(key,value)] if index < ep['boundary'] else policy.step(key,value)
                emitted.extend(events)
                for event in events:
                    state = write_event(state,event)
                if index == ep['boundary']:
                    # Resume after the first delay arrival, inside a replay block.
                    before = model.read(queries,state).clone()
                    torch.save(model.export_state(state),out/f'{arm}-midblock.pt')
                    payload = policy.export_state()
                    (out/f'{arm}-scheduler.json').write_text(json.dumps(payload,indent=2)+'\n')
                    state = model.import_state(torch.load(out/f'{arm}-midblock.pt',weights_only=True))
                    policy = StreamingReplay.from_state(json.loads((out/f'{arm}-scheduler.json').read_text()))
                    assert torch.equal(before,model.read(queries,state))
            extra = policy.finish()
            emitted.extend(extra)
            for event in extra:
                state = write_event(state,event)
            plan = replay_plan(ep,arm)
            assert emitted == plan['events']
            offline = model.initial_state(1)
            for event in plan['events']:
                offline = write_event(offline,event)
            snapshot = model.export_state(state)
            live_scores = model.read(queries,state)
            assert equal_tree(snapshot,model.export_state(state))
            assert torch.equal(live_scores,model.read(queries,offline))
            target = [q['target'] for q in ep['queries']]
            predicted = live_scores.argmax(-1)[0].tolist()
            assert predicted == target
            outcomes.append(dict(arm=arm,writes=len(emitted),prediction=predicted,target=target,
                                 scores=live_scores[0].tolist(),live_offline_max_abs_error=0.0,
                                 exact_midblock_state_roundtrip=True))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    assert torch.equal(cpu_rng,torch.random.get_rng_state())
    result = dict(status='passed',scope='One fixed typed history, all four arms; integration only',
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  checkpoint_source_commit=provenance['source_commit'],
                  demo_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scheduler_source_sha256=hashlib.sha256((REPO/'clmm/pretrain/kv_scheduler.py').read_bytes()).hexdigest(),
                  environment=dict(python=sys.version,torch=str(torch.__version__),device='cpu'),
                  episode=ep, slow_parameters_unchanged=True, read_pure=True, outcomes=outcomes)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],checkpoint=str(checkpoint),out=str(out)),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    run(args.checkpoint,args.out)
