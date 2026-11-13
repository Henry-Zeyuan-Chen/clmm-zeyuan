"""Matched-write-budget diagnostic of retention under active-fact pressure.

All stochastic streams, Algorithm R draws and replay selections are generated
on CPU before model execution. NPZ traces are evaluator data, not model memory.
The logical replay budget is one working reservoir plus one block snapshot.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.research_memory import Reader
from scripts.replay_probe import process_peak_rss_bytes, read_frozen, sha256, synchronize


ARMS = ("online", "repeat", "continuous", "periodic")
BATCH_ARRAYS = (
    "keys", "values", "active_keys", "final_values", "reservoir_draw",
    "sample_slots", "sample_sources", "sample_value_sources",
    "working_sources_at_query", "working_value_sources_at_query",
)


def generate_data(episodes, facts, nkeys, nvals, delays, seed, buffer_size=32, block_size=8):
    """Generate independently seeded episodes and a shared, causal reservoir trace."""
    delays = sorted(set([0, *delays]))
    if episodes < 1 or not 8 <= facts <= nkeys or nvals < 2:
        raise ValueError("Need positive episodes, 8 <= facts <= nkeys, and at least 2 values")
    if block_size < 1 or buffer_size < block_size or facts + 2 < block_size:
        raise ValueError("Snapshot must fit the reservoir and the observed prefill")
    if any(delay < 0 or delay % block_size for delay in delays):
        raise ValueError("Evaluation delays must be nonnegative complete-block boundaries")
    maximum, prefill = max(delays), facts + 2
    length, blocks = prefill + maximum, maximum // block_size
    data = dict(facts=facts, nkeys=nkeys, nvals=nvals, seed=seed, prefill=prefill,
                buffer_size=buffer_size, block_size=block_size, delays=np.array(delays, dtype=np.int32))
    data.update(
        keys=np.empty((episodes, length), dtype=np.int32),
        values=np.empty((episodes, length), dtype=np.int16),
        active_keys=np.empty((episodes, facts), dtype=np.int32),
        final_values=np.empty((episodes, facts), dtype=np.int16),
        reservoir_draw=np.empty((episodes, length), dtype=np.int32),
        sample_slots=np.empty((episodes, blocks, block_size), dtype=np.int32),
        sample_sources=np.empty((episodes, blocks, block_size), dtype=np.int32),
        sample_value_sources=np.empty((episodes, blocks, block_size), dtype=np.int32),
        working_sources_at_query=np.empty((episodes, len(delays), buffer_size), dtype=np.int32),
        working_value_sources_at_query=np.empty((episodes, len(delays), buffer_size), dtype=np.int32))
    query_index = {delay: index for index, delay in enumerate(delays)}
    for episode in range(episodes):
        # Separate random streams prevent model/batching/arm choices from
        # influencing observations, admissions, or replay sampling.
        stream_rng = np.random.default_rng(np.random.SeedSequence([seed, facts, episode, 0]))
        reservoir_rng = np.random.default_rng(np.random.SeedSequence([seed, facts, episode, 1]))
        sample_rng = np.random.default_rng(np.random.SeedSequence([seed, facts, episode, 2]))
        active = stream_rng.choice(nkeys, facts, replace=False)
        initial = stream_rng.integers(nvals, size=facts)
        final = initial.copy()
        final[-2:] = (initial[-2:] + stream_rng.integers(1, nvals, size=2)) % nvals
        distractors = stream_rng.integers(facts - 4, size=maximum)
        keys = np.concatenate((active, active[-2:], active[distractors]))
        values = np.concatenate((initial, final[-2:], final[distractors]))
        data["keys"][episode], data["values"][episode] = keys, values
        data["active_keys"][episode], data["final_values"][episode] = active, final
        sources = np.full(buffer_size, -1, dtype=np.int32)
        value_sources = sources.copy()
        for index in range(length):
            # Freeze a sample before this block's first new observation.
            if index >= prefill and (index - prefill) % block_size == 0:
                block = (index - prefill) // block_size
                occupied = np.flatnonzero(sources >= 0)
                slots = sample_rng.choice(occupied, block_size, replace=False)
                data["sample_slots"][episode, block] = slots
                data["sample_sources"][episode, block] = sources[slots]
                data["sample_value_sources"][episode, block] = value_sources[slots]
                assert np.all(sources[slots] < index) and np.all(value_sources[slots] < index)
            # Revise every resident copy before Algorithm R admission. The
            # observation reservoir can contain multiple tickets for a key.
            occupied = np.flatnonzero(sources >= 0)
            matching = occupied[keys[sources[occupied]] == keys[index]]
            value_sources[matching] = index
            draw = index if index < buffer_size else int(reservoir_rng.integers(index + 1))
            data["reservoir_draw"][episode, index] = draw
            if draw < buffer_size:
                sources[draw], value_sources[draw] = index, index
            delay = index + 1 - prefill
            if delay in query_index:
                qi = query_index[delay]
                data["working_sources_at_query"][episode, qi] = sources
                data["working_value_sources_at_query"][episode, qi] = value_sources
    # The protocol has no post-prefill label changes or target observations.
    assert not np.any(data["keys"][:, prefill:, None] == data["active_keys"][:, None, -4:])
    return data


def slice_data(data, start, stop):
    return {name: value[start:stop] if name in BATCH_ARRAYS else value for name, value in data.items()}


def block_events(arm, block, prefill, block_size):
    """Indices address observations or the frozen block sample, never a live FIFO."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    observations = [("observation", prefill + block * block_size + rank) for rank in range(block_size)]
    if arm == "online":
        return observations
    if arm == "repeat":
        return [event for observation in observations for event in (observation, ("repeat", observation[1]))]
    replay = [("replay", rank) for rank in range(block_size)]
    if arm == "continuous":
        return [event for pair in zip(observations, replay) for event in pair]
    return observations + replay


def query_spec(data, observed):
    """Four designated targets plus four latest distinct non-target keys."""
    keys, values, facts = data["keys"], data["values"], data["facts"]
    batch = len(keys)
    indices = np.empty((batch, 8), dtype=np.int32)
    indices[:, :4] = [facts - 4, facts - 3, facts, facts + 1]
    for episode in range(batch):
        excluded = set(data["active_keys"][episode, -4:].tolist())
        positions = []
        for index in range(observed - 1, -1, -1):
            key = int(keys[episode, index])
            if key not in excluded:
                positions.append(index)
                excluded.add(key)
            if len(positions) == 4:
                break
        assert len(positions) == 4
        indices[episode, 4:] = positions
    return np.take_along_axis(keys, indices, 1), np.take_along_axis(values, indices, 1), indices


@torch.no_grad()
def evaluate_batch(model, data, arm, device="cpu", aux_decay_scale=0):
    """Execute one arm; query data and shadow state never enter Reader.read."""
    if arm not in ARMS or aux_decay_scale not in (0, 1):
        raise ValueError("Invalid arm or auxiliary decay scale")
    device = torch.device(device)
    keys, values = data["keys"], data["values"]
    batch, length = keys.shape
    prefill, block_size = data["prefill"], data["block_size"]
    key_tensor = torch.as_tensor(keys, device=device, dtype=torch.long)
    value_tensor = torch.as_tensor(values, device=device, dtype=torch.long)
    sample_keys = np.take_along_axis(keys[:, None, :], data["sample_sources"], axis=2)
    sample_values = np.take_along_axis(values[:, None, :], data["sample_value_sources"], axis=2)
    for block in range(data["sample_sources"].shape[1]):
        boundary = prefill + block * block_size
        for name in ("sample_sources", "sample_value_sources"):
            assert np.all((data[name][:, block] >= 0) & (data[name][:, block] < boundary)), "Noncausal replay source"
        for slots in data["sample_slots"][:, block]:
            assert len(set(slots)) == block_size and np.all((slots >= 0) & (slots < data["buffer_size"]))
    assert np.array_equal(sample_keys, np.take_along_axis(keys[:, None, :], data["sample_value_sources"], axis=2)), "Replay value belongs to another key"
    sample_key_tensor = torch.as_tensor(sample_keys, device=device, dtype=torch.long)
    sample_value_tensor = torch.as_tensor(sample_values, device=device, dtype=torch.long)
    # Revision handling guarantees each replay tuple uses its final key label.
    for episode in range(batch):
        final = dict(zip(data["active_keys"][episode].tolist(), data["final_values"][episode].tolist()))
        assert all(final[int(key)] == int(value) for key, value in zip(
            sample_keys[episode].flat, sample_values[episode].flat))
    fast, past = model.memory.init_weights(batch), None
    last_any = np.full((batch, data["nkeys"]), -1, dtype=np.int32)
    last_aux, last_replay, last_shadow = last_any.copy(), last_any.copy(), last_any.copy()
    last_source = last_any.copy()
    aux_count, replay_count, shadow_count = (np.zeros_like(last_any) for _ in range(3))
    episodes = np.arange(batch)
    writes = auxiliary = observed = 0
    rows = []

    def write(k, v, numeric_keys, source_indices, kind):
        nonlocal fast, past, writes, auxiliary, observed
        fact = torch.stack((model.key(k[:, None]), model.value(v[:, None])))
        handle = None
        if kind != "observation" and aux_decay_scale == 0:
            handle = model.memory.to_decay_factor.register_forward_hook(
                lambda _module, _inputs, output: torch.full_like(output, -torch.inf))
        try:
            _, returned = model.memory.store_memories(
                fact, weights=fast, past_state=past, seq_index=writes, return_surprises=False)
        finally:
            if handle is not None:
                handle.remove()
        past, fast = returned.states, returned.states[0]
        last_any[episodes, numeric_keys] = writes
        last_source[episodes, numeric_keys] = source_indices
        if kind != "observation":
            last_aux[episodes, numeric_keys] = writes
            aux_count[episodes, numeric_keys] += 1
            auxiliary += 1
        if kind == "replay":
            last_replay[episodes, numeric_keys] = writes
            replay_count[episodes, numeric_keys] += 1
        observed += kind == "observation"
        writes += 1

    def probe(delay):
        assert observed == prefill + delay
        assert writes == prefill + delay * (1 if arm == "online" else 2)
        assert auxiliary == (0 if arm == "online" else delay)
        # A block's logical replay snapshot is discarded before this call.
        # Only the common working reservoir defines the residency subgroup.
        query, answers, observed_indices = query_spec(data, observed)
        qt = torch.as_tensor(query, device=device, dtype=torch.long)
        predictions = torch.stack(read_frozen(model, qt.unbind(dim=1), fast), dim=1).cpu().numpy()
        qi = list(data["delays"]).index(delay)
        working = data["working_sources_at_query"][:, qi]
        for episode in range(batch):
            resident_indices = working[episode][working[episode] >= 0]
            resident_keys = keys[episode, resident_indices]
            for rank in range(8):
                key, answer = int(query[episode, rank]), int(answers[episode, rank])
                condition = "stable" if rank < 2 else "revised" if rank < 4 else "recent"
                old_value = int(values[episode, data["facts"] - 2 + (rank - 2)]) if condition == "revised" else ""
                any_write, aux_write, replay_write = (
                    int(array[episode, key]) for array in (last_any, last_aux, last_replay))
                shadow_block = int(last_shadow[episode, key])
                rows.append(dict(
                    episode=episode, facts=data["facts"], arm=arm, delay=delay,
                    condition=condition, query_rank=rank if rank < 2 else rank - 2 if rank < 4 else rank - 4,
                    key=key, target_value=answer, old_value=old_value,
                    prediction=int(predictions[episode, rank]), correct=int(predictions[episode, rank] == answer),
                    stale_old=int(predictions[episode, rank] == old_value) if condition == "revised" else "",
                    initial_event_multiplicity=1 if condition == "stable" else 2 if condition == "revised" else "",
                    last_observation_index=int(observed_indices[episode, rank]),
                    future_observations=observed - 1 - int(observed_indices[episode, rank]),
                    total_observations=observed, total_memory_writes=writes, total_aux_writes=auxiliary,
                    last_any_write_index=any_write, future_memory_writes=writes - 1 - any_write,
                    last_write_value_source_index=int(last_source[episode, key]),
                    ever_aux_written=aux_write >= 0, last_aux_write_index=aux_write,
                    auxiliary_writes_for_key=int(aux_count[episode, key]),
                    future_writes_since_aux=writes - 1 - aux_write if aux_write >= 0 else "",
                    ever_replayed=replay_write >= 0, last_replay_write_index=replay_write,
                    replay_writes_for_key=int(replay_count[episode, key]),
                    future_writes_since_replay=writes - 1 - replay_write if replay_write >= 0 else "",
                    shadow_ever_selected_for_replay=shadow_block >= 0, last_shadow_selection_block=shadow_block,
                    shadow_selection_count=int(shadow_count[episode, key]),
                    working_resident=bool(np.any(resident_keys == key)),
                    working_resident_copies=int(np.sum(resident_keys == key)),
                    snapshot_resident=False, snapshot_active_pairs=0, pending_auxiliary_writes=0,
                    local_attention_tokens=0))

    for index in range(prefill):
        write(key_tensor[:, index], value_tensor[:, index], keys[:, index], index, "observation")
    probe(0)
    for block in range((length - prefill) // block_size):
        for rank in range(block_size):
            last_shadow[episodes, sample_keys[:, block, rank]] = block
            shadow_count[episodes, sample_keys[:, block, rank]] += 1
        replayed = []
        for kind, index in block_events(arm, block, prefill, block_size):
            if kind == "replay":
                write(sample_key_tensor[:, block, index], sample_value_tensor[:, block, index],
                      sample_keys[:, block, index], data["sample_value_sources"][:, block, index], kind)
                replayed.append(index)
            else:
                write(key_tensor[:, index], value_tensor[:, index], keys[:, index], index, kind)
        assert replayed == (list(range(block_size)) if arm in ("continuous", "periodic") else [])
        # Snapshot is now invalid for all arms, including their shadow schedule.
        delay = (block + 1) * block_size
        if delay in data["delays"]:
            probe(delay)
    return rows


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("metadata.json", "predictions.csv", "summary.csv", "sources")) or list(out.glob("streams_*.npz")):
        raise FileExistsError("Refusing to overwrite existing pressure-probe artifacts")
    start = time.perf_counter()
    metadata = dict(status="running", command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoints=[], runtime_by_arm=[], datasets=[])
    metadata_path = out / "metadata.json"
    try:
        if args.episodes < 1 or args.batch_size < 1:
            raise ValueError("Episode and batch counts must be positive")
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested CPU/CUDA device is unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        revision = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        source_paths = [Path(__file__).resolve(), REPO / "scripts" / "research_memory.py",
                        REPO / "scripts" / "replay_probe.py", REPO / "titans_pytorch" / "neural_memory.py",
                        REPO / "titans_pytorch" / "memory_models.py"]
        sources = {}
        for source in source_paths:
            destination = out / "sources" / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            sources[str(source)] = sha256(source)
        checkpoint_data = []
        for path in args.checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if "config" not in checkpoint:
                raise ValueError("Pressure checkpoints must record their Reader config")
            checkpoint_data.append((path.resolve(), checkpoint))
        vocabularies = {(entry["config"].get("nkeys", 16), entry["config"].get("nvals", 8)) for _, entry in checkpoint_data}
        if len(vocabularies) != 1:
            raise ValueError("All checkpoints must share the key/value vocabulary")
        nkeys, nvals = next(iter(vocabularies))
        if nvals != 8:
            raise ValueError("This protocol fixes eight value classes")
        facts_list, delays = sorted(set(args.facts)), sorted(set([0, *args.delays]))
        metadata.update(repo=str(REPO), repo_commit=revision, source_sha256=sources,
                        design_sha256=sha256(args.design) if args.design else None,
                        eval_seed=args.seed, episodes=args.episodes, batch_size=args.batch_size,
                        facts=facts_list, delays=delays, nkeys=nkeys, nvals=nvals,
                        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                        torch=str(torch.__version__), numpy=np.__version__, python=sys.version,
                        threads=1, interop_threads=1, deterministic_algorithms=True, cuda_tf32=False,
                        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                        aux_decay_scale=args.aux_decay_scale, arms=list(ARMS),
                        working_capacity=args.buffer_size, snapshot_capacity=args.block_size,
                        logical_pair_capacity=args.buffer_size + args.block_size,
                        capacity_measurement="Logical policy capacity only; no claim that process RSS is limited to 40 pairs",
                        simulator_trace_access="Evaluator preloads full observations and future sampled indices; source-index checks enforce strict causal replay access, and Reader.read receives only query keys and fast weights",
                        expected_uniform_chance=1 / nvals,
                        target_positions="Stable K-4,K-3; old revised K-2,K-1; replacement writes K,K+1",
                        fixed_post_prefill_labels=True, auxiliary_writes_during_prefill=0,
                        snapshot_discarded_before_query=True,
                        replay_tuple_matching="Continuous and periodic use identical frozen sampled slots, keys, values and order",
                        reservoir="Algorithm R over observation events; update all resident copies of observed key before admission",
                        raw_trace="NPZ contains observations, active/final labels, Algorithm R draws, sampled slot and source indices, query-time working slots",
                        source_index_meaning="sample_sources is admitted event/key; sample_value_sources includes resident-label refreshes",
                        auxiliary_definition="repeat and both replay arms add exactly one write per delay observation; online is unmatched",
                        ever_replayed_definition="Reservoir replay only; ever_aux_written additionally includes immediate repetition",
                        shadow_selection="Identical precomputed reservoir samples for all arms; unused samples remain evaluator-only",
                        query_access="Reader.read sees query key and fast weights only; no reservoir or local-attention access",
                        inference_unit="Episode; two stable, two revised, four recent probes are correlated",
                        caveats=["Working residency is descriptive, not randomly assigned treatment.",
                                 "Revised targets have two observation tickets, stable targets one; raw type differences are not causal revision effects.",
                                 "Target observation ages match across fact counts; recent-query ages need not.",
                                 "All pretrained checkpoint parameters remain fixed; fast weights update online.",
                                 "Saved evaluator traces exceed the logical model reservoir budget and are never available to query retrieval."])
        datasets = {}
        for facts in facts_list:
            data = generate_data(args.episodes, facts, nkeys, nvals, delays, args.seed,
                                 args.buffer_size, args.block_size)
            trace_path = out / f"streams_K{facts}.npz"
            np.savez_compressed(trace_path, **data)
            metadata["datasets"].append(dict(facts=facts, path=str(trace_path), sha256=sha256(trace_path)))
            datasets[facts] = data
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
        counts = defaultdict(lambda: [0, 0, 0])
        with (out / "predictions.csv").open("w", newline="", encoding="utf8") as handle:
            writer = None
            for checkpoint_index, (path, checkpoint) in enumerate(checkpoint_data):
                model = Reader(**checkpoint["config"]).to(device)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                assert set(dict(model.named_children())) == {"key", "value", "memory", "head"}
                assert model.memory.store_chunk_size == 1 and model.memory.to_momentum is None
                frozen = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                checkpoint_hash = sha256(path)
                report = dict(index=checkpoint_index, path=str(path), sha256=checkpoint_hash,
                              config=checkpoint["config"], init_seed=checkpoint.get("init_seed"),
                              train_seed=checkpoint.get("train_seed"), steps=checkpoint.get("steps"),
                              parameters=sum(parameter.numel() for parameter in model.parameters()))
                metadata["checkpoints"].append(report)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for facts, data in datasets.items():
                    elapsed = defaultdict(float)
                    for offset in range(0, args.episodes, args.batch_size):
                        batch = slice_data(data, offset, offset + args.batch_size)
                        for arm in ARMS:
                            synchronize(device)
                            arm_start = time.perf_counter()
                            rows = evaluate_batch(model, batch, arm, device, args.aux_decay_scale)
                            synchronize(device)
                            elapsed[arm] += time.perf_counter() - arm_start
                            for row in rows:
                                row["episode"] += offset
                                row.update(checkpoint_index=checkpoint_index, checkpoint_sha256=checkpoint_hash,
                                           init_seed=checkpoint.get("init_seed", ""), train_seed=checkpoint.get("train_seed", ""),
                                           eval_seed=args.seed, aux_decay_scale=args.aux_decay_scale)
                                if writer is None:
                                    writer = csv.DictWriter(handle, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                                count = counts[(checkpoint_index, facts, arm, row["delay"], row["condition"])]
                                count[0] += 1
                                count[1] += row["correct"]
                                count[2] += row["stale_old"] if row["condition"] == "revised" else 0
                        handle.flush()
                    for arm in ARMS:
                        entry = dict(checkpoint_index=checkpoint_index, facts=facts, arm=arm,
                                     elapsed_seconds=elapsed[arm], episodes=args.episodes,
                                     observation_writes=args.episodes * (data["prefill"] + max(delays)),
                                     auxiliary_writes=0 if arm == "online" else args.episodes * max(delays))
                        entry["total_memory_writes"] = entry["observation_writes"] + entry["auxiliary_writes"]
                        metadata["runtime_by_arm"].append(entry)
                    print(json.dumps(dict(checkpoint=checkpoint_index, facts=facts, seconds=dict(elapsed))), flush=True)
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
                assert all(torch.equal(frozen[name], tensor) for name, tensor in model.state_dict().items()), "Checkpoint mutated"
                report["checkpoint_parameters_and_buffers_unchanged"] = True
                if device.type == "cuda":
                    report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                del model, frozen
        with (out / "summary.csv").open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("checkpoint_index", "facts", "arm", "delay", "condition", "n", "correct", "accuracy", "stale_old"))
            for group, (n, correct, stale) in sorted(counts.items()):
                assert n == args.episodes * (4 if group[-1] == "recent" else 2)
                writer.writerow((*group, n, correct, correct / n, stale if group[-1] == "revised" else ""))
        assert all(sha256(path) == digest for path, digest in sources.items()), "Source changed during run"
        metadata.update(status="complete", process_peak_rss_bytes=process_peak_rss_bytes())
    except Exception as error:
        metadata.update(status="failed", exception_type=type(error).__name__, exception=str(error),
                        traceback=traceback.format_exc())
        raise
    finally:
        metadata["elapsed_seconds_including_setup_io"] = time.perf_counter() - start
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--facts", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 64, 256, 1024])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=58111)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aux-decay-scale", type=int, choices=(0, 1), default=0)
    parser.add_argument("--buffer-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--design", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
