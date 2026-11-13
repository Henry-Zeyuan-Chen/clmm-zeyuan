"""Frozen-reader replay timing diagnostic; no training or checkpoint modification.

Both replay arms consume the identical ordered snapshot keys once per block.
Stored values also match by default; the optional latest-observed correction
may supply different values because observations arrive at different times.
The snapshot is the preceding eight *observations*, never previous replay writes.
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
from contextlib import ExitStack
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


def select_recent(keys, values, observations, count=4):
    """Latest distinct non-target keys and labels, using the observed prefix only."""
    if not 0 < observations <= keys.shape[1] or count < 1:
        raise ValueError("Invalid observation boundary or recent-query count")
    selected_keys, selected_values, selected_indices = [], [], []
    for episode_keys, episode_values in zip(
            keys[:, :observations].tolist(), values[:, :observations].tolist()):
        excluded = {episode_keys[0], episode_keys[1]}
        positions = []
        for index in range(observations - 1, -1, -1):
            key = episode_keys[index]
            if key in excluded:
                continue
            excluded.add(key)
            positions.append(index)
            if len(positions) == count:
                break
        if len(positions) != count:
            raise ValueError("Observed prefix has fewer than four distinct non-target keys")
        selected_keys.append([episode_keys[index] for index in positions])
        selected_values.append([episode_values[index] for index in positions])
        selected_indices.append(positions)
    return tuple(torch.tensor(data, device=keys.device, dtype=torch.long)
                 for data in (selected_keys, selected_values, selected_indices))


def latest_available_sources(keys, bank, observations, replay_index):
    """Resolve a bank key only within the bank and current arrivals (<=16 pairs)."""
    bank = tuple(bank)
    if len(bank) != BLOCK_SIZE or bank != tuple(range(bank[0], bank[0] + BLOCK_SIZE)):
        raise ValueError("Expected one chronological eight-observation bank")
    start = bank[-1] + 1
    if replay_index not in bank or not start <= observations <= start + BLOCK_SIZE:
        raise ValueError("Replay key or arrived-observation boundary is outside the block")
    available = bank + tuple(range(start, observations))
    assert len(available) <= 2 * BLOCK_SIZE
    indices = keys.new_tensor(available)
    matches = keys[:, list(available)] == keys[:, replay_index:replay_index + 1]
    # The original bank entry guarantees at least one match for every episode.
    return torch.where(matches, indices[None, :], -1).amax(dim=1)


@torch.no_grad()
def evaluate_batch(model, keys, values, algorithm, *, replay_decay_scale=1, recent_queries=0,
                   replay_value_source="stored", replay_diagnostics=None):
    """Evaluate one paired stream batch, resetting fast weights for this arm."""
    if replay_decay_scale not in (0, 1):
        raise ValueError("replay_decay_scale must be 0 or 1")
    if recent_queries not in (0, 4):
        raise ValueError("recent_queries must be 0 or 4")
    if replay_value_source not in ("stored", "latest-observed"):
        raise ValueError("Unknown replay value source")
    if keys.shape != values.shape or keys.shape[1] != PREFILL + BLOCKS * BLOCK_SIZE:
        raise ValueError("Expected matching [batch,72] key/value streams")
    fast = model.memory.init_weights(keys.shape[0])
    past = None
    writes = replays = observations = 0
    last_target_write = {}
    if recent_queries:
        last_key_write = torch.full((keys.shape[0], model.key.num_embeddings), -1,
                                    dtype=torch.long, device=keys.device)
        last_key_source = torch.full_like(last_key_write, -1)
    rows = []

    def write(index, kind, value_indices=None):
        nonlocal fast, past, writes, replays, observations
        written_values = (values[:, index:index + 1] if value_indices is None
                          else values.gather(1, value_indices[:, None]))
        fact = torch.stack((model.key(keys[:, index:index + 1]),
                            model.value(written_values)))
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
        if recent_queries:
            last_key_write.scatter_(1, keys[:, index:index + 1], writes - 1)
            if value_indices is None:
                last_key_source.scatter_(1, keys[:, index:index + 1], index)
            else:
                last_key_source.scatter_(1, keys[:, index:index + 1], value_indices[:, None])
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
            value_indices = None
            if kind == "replay" and (replay_value_source == "latest-observed" or replay_diagnostics is not None):
                latest_sources = latest_available_sources(keys, bank, observations, index)
                if replay_value_source == "latest-observed":
                    value_indices = latest_sources
            write(index, kind, value_indices)
            if kind == "replay":
                assert pending.pop(0) == index
                replayed.append(index)
                if replay_diagnostics is not None:
                    actual_sources = (torch.full_like(latest_sources, index)
                                      if value_indices is None else value_indices)
                    latest_values = values.gather(1, latest_sources[:, None])[:, 0]
                    actual_values = values.gather(1, actual_sources[:, None])[:, 0]
                    diagnostics = torch.stack((keys[:, index], values[:, index], latest_values,
                                               actual_values, latest_sources, actual_sources), dim=1).tolist()
                    for episode, (key, stored, latest, used, latest_source, used_source) in enumerate(diagnostics):
                        replay_diagnostics.append(dict(
                            episode=episode, algorithm=algorithm, block=block,
                            replay_rank=len(replayed) - 1, memory_write_index=writes - 1,
                            observed_count=observations, key=key, stored_observation_index=index,
                            stored_value=stored, latest_available_observation_index=latest_source,
                            latest_available_value=latest, used_observation_index=used_source,
                            used_value=used, stored_value_stale=int(stored != latest),
                            value_substituted=int(stored != used), source_substituted=int(index != used_source),
                            replay_value_source=replay_value_source,
                            bank_first_observation_index=bank[0], bank_last_observation_index=bank[-1],
                            available_pair_count=BLOCK_SIZE + observations - bank[-1] - 1))
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
        if recent_queries:
            query, target, observed_index = select_recent(keys, values, observations, recent_queries)
            latest_write = last_key_write.gather(1, query)
            latest_source = last_key_source.gather(1, query)
            latest_written_value = values.gather(1, latest_source)
            in_bank = (query[:, :, None] == keys[:, None, list(bank)]).any(dim=-1)
            in_fifo = (query[:, :, None] == keys[:, None, observations - BLOCK_SIZE:observations]).any(dim=-1)
            predicted = torch.stack(read_frozen(model, query.unbind(dim=1), fast), dim=1)
            assert writes == before_read
            query, target, observed_index, latest_write, latest_source, latest_written_value, in_bank, in_fifo, predicted = (
                tensor.tolist() for tensor in (query, target, observed_index, latest_write,
                                              latest_source, latest_written_value, in_bank, in_fifo, predicted))
            for episode in range(keys.shape[0]):
                for rank in range(recent_queries):
                    rows.append(dict(
                        episode=episode, algorithm=algorithm, block=block, condition="recent",
                        query_rank=rank, target_key=query[episode][rank],
                        target_value=target[episode][rank], old_value="",
                        prediction=predicted[episode][rank],
                        correct=int(predicted[episode][rank] == target[episode][rank]), stale_old="",
                        future_observations=observations - 1 - observed_index[episode][rank],
                        total_observations=observations, total_memory_writes=writes,
                        total_replay_writes=replays,
                        last_target_observation_index=observed_index[episode][rank],
                        last_target_memory_write_index=latest_write[episode][rank],
                        future_memory_writes_since_last_target=writes - 1 - latest_write[episode][rank],
                        last_memory_write_source_observation_index=latest_source[episode][rank],
                        last_memory_write_value=latest_written_value[episode][rank],
                        target_absent_from_fifo=not in_fifo[episode][rank],
                        target_absent_from_bank=not in_bank[episode][rank],
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
    for name in ("predictions.csv", "recent_predictions.csv", "replay_values.csv", "episodes.csv", "summary.csv", "metadata.json"):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite {out / name}")
    with (out / "episodes.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "eval_seed", "keys", "values"))
        for episode, (k, v) in enumerate(zip(keys.tolist(), values.tolist())):
            writer.writerow((episode, args.seed, " ".join(map(str, k)), " ".join(map(str, v))))
    counts = defaultdict(lambda: dict(n=0, correct=0, stale_old=0))
    checkpoint_reports = []
    with (out / "predictions.csv").open("w", newline="", encoding="utf8") as handle, ExitStack() as stack:
        writer = None
        recent_writer = None
        replay_writer = None
        replay_handle = stack.enter_context((out / "replay_values.csv").open("w", newline="", encoding="utf8"))
        recent_handle = (stack.enter_context((out / "recent_predictions.csv").open(
            "w", newline="", encoding="utf8")) if args.recent_queries else None)
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
                    replay_diagnostics = []
                    rows = evaluate_batch(model, k, v, algorithm,
                                          replay_decay_scale=args.replay_decay_scale,
                                          recent_queries=args.recent_queries,
                                          replay_value_source=args.replay_value_source,
                                          replay_diagnostics=replay_diagnostics)
                    synchronize(device)
                    elapsed_by_arm[algorithm] += time.perf_counter() - arm_start
                    for row in rows:
                        row["episode"] += offset
                        row.update(checkpoint_index=checkpoint_index,
                                   checkpoint_path=str(path), checkpoint_sha256=checkpoint_hash,
                                   init_seed=checkpoint.get("init_seed", ""),
                                   train_seed=checkpoint.get("train_seed", ""), eval_seed=args.seed)
                        if row["condition"] == "recent":
                            if recent_writer is None:
                                recent_writer = csv.DictWriter(recent_handle, fieldnames=list(row))
                                recent_writer.writeheader()
                            recent_writer.writerow(row)
                        else:
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                        count = counts[(checkpoint_index, algorithm, row["block"], row["condition"])]
                        count["n"] += 1
                        count["correct"] += row["correct"]
                        count["stale_old"] += row["stale_old"] if row["condition"] == "revised" else 0
                    assert len(replay_diagnostics) == (0 if algorithm == "online" else k.shape[0] * BLOCKS * BLOCK_SIZE)
                    for row in replay_diagnostics:
                        row["episode"] += offset
                        row.update(checkpoint_index=checkpoint_index, checkpoint_sha256=checkpoint_hash,
                                   init_seed=checkpoint.get("init_seed", ""),
                                   train_seed=checkpoint.get("train_seed", ""), eval_seed=args.seed)
                        if replay_writer is None:
                            replay_writer = csv.DictWriter(replay_handle, fieldnames=list(row))
                            replay_writer.writeheader()
                        replay_writer.writerow(row)
                handle.flush()
                replay_handle.flush()
                if recent_handle is not None:
                    recent_handle.flush()
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
            assert count["n"] == args.episodes * (args.recent_queries if condition == "recent" else 1)
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
        replay_value_source=args.replay_value_source,
        replay_value_resolution=("Original immutable bank value; latest-available candidates logged only"
                                 if args.replay_value_source == "stored" else
                                 "Latest observed occurrence among immutable bank plus current-block observations already arrived; at most 16 pairs"),
        replay_values_guaranteed_matched_between_schedules=args.replay_value_source == "stored",
        replay_value_diagnostics="replay_values.csv",
        latest_observed_intervention_purpose="Diagnostic correction of outdated replay labels; not a proposed final sleep algorithm",
        replay_decay_scale=args.replay_decay_scale,
        replay_decay_intervention=("standard learned decay on all writes" if args.replay_decay_scale == 1
                                  else "replay writes only: decay logits forced to -inf, sigmoid decay=0; observation decay unchanged"),
        recent_queries_per_episode=args.recent_queries,
        recent_query_selection="Reverse observed prefix; four latest distinct keys excluding revised and stable targets; label is latest observed value",
        recent_query_output="recent_predictions.csv" if args.recent_queries else None,
        recent_query_inference_unit="Episode, not individual probe; four correlated probes per episode",
        no_checkpoint_training=True, checkpoints=checkpoint_reports,
        process_peak_rss_bytes=process_peak_rss_bytes(),
        elapsed_seconds_including_setup_io=time.perf_counter() - start,
        caveats=["Online has fewer memory updates; replay-vs-online is not compute matched.",
                 ("Stored mode: continuous vs periodic differ only in interleaving at evaluated block boundaries."
                  if args.replay_value_source == "stored" else
                  "Latest-observed mode: schedules retain identical replay keys/order/counts, but available labels may differ; replay examples are not matched."),
                 "Block1 snapshot includes both old and new target facts; later snapshots contain neither target.",
                 "Delays count future observations, not replay updates: revised19/35/67, stable22/38/70.",
                 "Probe horizons and update schedules may differ from training; consult each checkpoint's training metadata.",
                 "Identical replay keys do not imply identical gradients; latest-observed labels may differ across schedules.",
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
    parser.add_argument("--recent-queries", type=int, choices=(0, 4), default=0,
                        help="4: also query four most recent distinct non-target keys; 0: disabled")
    parser.add_argument("--replay-value-source", choices=("stored", "latest-observed"), default="stored",
                        help="Replay stored labels, or resolve each key from bank plus current arrivals only")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
