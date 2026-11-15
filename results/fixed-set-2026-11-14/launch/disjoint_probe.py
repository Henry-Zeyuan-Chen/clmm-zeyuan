"""Bounded, training-disjoint novel-value evaluation of four frozen endpoints.

This external amendment imports the frozen experiment repository without
modifying it. Selection uses no model outputs and fails closed if the declared
8,192-candidate stream cannot supply four novel contexts per base pair.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


FROZEN_COMMIT = "5bccffd9084b543ca086721be2bd46e6a2ea27a6"
RUNS = ("fixed32-68101", "fixed128-68101", "fresh32-68101", "fresh128-68101")
SELECTION_SEED = 90205
CANDIDATES = 8192
BASE_PAIRS = 32
VARIANTS = 4
BATCH_SIZE = 16
FINAL_STEP = 2000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def before_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Disjoint probe wall-clock limit reached")


def bootstrap(repo):
    """Import only from the explicitly supplied frozen checkout."""
    repo = Path(repo).resolve()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(repo))
    common = importlib.import_module("scripts.fact_revision_experiment")
    pretrain = importlib.import_module("clmm.pretrain")
    facts = importlib.import_module("clmm.pretrain.fact_streams")
    association = importlib.import_module("clmm.pretrain.association_probe")
    for module in (common, pretrain, facts, association):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError(f"Imported source outside --repo: {module.__file__}")
    return SimpleNamespace(repo=repo, common=common, np=importlib.import_module("numpy"),
                           torch=importlib.import_module("torch"), facts=facts,
                           association=association, pretrain=pretrain)


def training_contexts(input_bytes, np):
    """Extract exact ordered 64-byte fact records, excluding query and label."""
    array = np.asarray(input_bytes)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 95:
        raise ValueError("Training input_bytes must be uint8 [steps,batch,95]")
    if array.shape[1] != BATCH_SIZE or array.shape[0] != FINAL_STEP:
        raise ValueError("Require complete step-2000 B16 training inputs")
    if not np.array_equal(array[:, 0::2, :64], array[:, 1::2, :64]):
        raise ValueError("Training paired queries do not share their fact context")
    return {row.tobytes() for row in array[..., :64].reshape(-1, 64)}


def context_bytes(episode):
    context = b"".join(record.encoded for record in episode.events)
    if len(context) != 64:
        raise AssertionError("Expected exactly two 32-byte facts")
    return context


def select_episodes(bank, excluded, runtime, deadline=None):
    """Keep first four eligible unique candidates for each ordered bank pair."""
    if len(bank.get("episodes", [])) != BASE_PAIRS:
        raise ValueError("The amended selection requires exactly 32 base pairs")
    originals = [runtime.facts.Episode.from_dict(data) for data in bank["episodes"]]
    if bank.get("key_pool") != list(runtime.facts.KEY_POOLS["train"][:16]):
        raise ValueError("Require the declared first 16 training keys")
    pools = {name: set(pool) for name, pool in runtime.facts.KEY_POOLS.items()}
    accepted = [[] for _ in originals]
    seen = [set() for _ in originals]
    counters = dict(visited=0, rejected_training=0, rejected_duplicate=0,
                    skipped_completed_base_pair=0)
    candidates = runtime.association.fresh_same(bank, SELECTION_SEED, count=CANDIDATES)
    for source_index, episode in enumerate(candidates):
        before_deadline(deadline)
        counters["visited"] += 1
        base_pair = source_index % BASE_PAIRS
        if len(accepted[base_pair]) == VARIANTS:
            counters["skipped_completed_base_pair"] += 1
            continue
        original = originals[base_pair]
        context = context_bytes(episode)
        keys = [record.key for record in episode.events]
        old_values = {record.value for record in original.events}
        values = {record.value for record in episode.events}
        if (keys != [record.key for record in original.events]
                or not set(keys) <= pools["train"]
                or set(keys) & (pools["dev"] | pools["test"])
                or len(values) != 2 or values & old_values
                or episode.split != "train" or episode.template_family != "seen"
                or any(record.template_index != 0 for record in episode.events)
                or any(query.template_index != 0 for query in episode.queries)):
            raise AssertionError("fresh_same violated the declared selection semantics")
        if context in excluded:
            counters["rejected_training"] += 1
            continue
        if context in seen[base_pair]:
            counters["rejected_duplicate"] += 1
            continue
        seen[base_pair].add(context)
        accepted[base_pair].append(dict(base_pair=base_pair,
            variant=len(accepted[base_pair]), source_candidate_index=source_index,
            context_sha256=hashlib.sha256(context).hexdigest(), episode=episode.to_dict()))
        if all(len(group) == VARIANTS for group in accepted):
            break
    if any(len(group) != VARIANTS for group in accepted):
        counts = [len(group) for group in accepted]
        raise RuntimeError(f"Candidate cap exhausted; accepted counts by base pair: {counts}")
    selected = [record for group in accepted for record in group]
    contexts = [context_bytes(runtime.facts.Episode.from_dict(record["episode"]))
                for record in selected]
    if len(set(contexts)) != BASE_PAIRS * VARIANTS or set(contexts) & excluded:
        raise AssertionError("Accepted contexts duplicate or overlap training")
    counters.update(accepted=len(selected), candidates_generated=CANDIDATES,
                    excluded_unique_contexts=len(excluded), accepted_unique_contexts=len(set(contexts)))
    return selected, counters


def load_artifacts(root, runtime, deadline=None):
    np = runtime.np
    branch = read_json(root / "branch.json")
    if branch.get("choice") != "fresh":
        raise ValueError("This amendment is authorized only for the declared fresh branch")
    records, arrays, banks = {}, {}, {}
    for name in RUNS:
        before_deadline(deadline)
        folder = root / name
        meta = read_json(folder / "metadata.json")
        args = meta.get("args", {})
        geometry = 32 if "32-" in name else 128
        regime = "fresh" if name.startswith("fresh") else "fixed"
        if (meta.get("status") != "succeeded" or meta.get("completed_steps") != FINAL_STEP
                or meta.get("parent_init_seed") != 68101 or meta.get("repo_commit") != FROZEN_COMMIT
                or args.get("regime") != regime or args.get("segment_len") != geometry
                or args.get("batch_size") != BATCH_SIZE or args.get("contexts") != BASE_PAIRS
                or args.get("key_count") != 16 or args.get("bank_seed") != 90201
                or args.get("train_seed") != 90202):
            raise ValueError(f"Run differs from the declared final endpoint: {name}")
        checkpoint = folder / "step-2000" / "checkpoint.pt"
        digest = sha256(checkpoint)
        final = [entry for entry in meta.get("checkpoints", []) if entry.get("step") == FINAL_STEP]
        stage = read_json(folder / "step-2000" / "metadata.json")
        if (len(final) != 1 or final[0].get("checkpoint_sha256") != digest
                or stage.get("checkpoint_sha256") != digest or stage.get("step") != FINAL_STEP):
            raise ValueError(f"Final checkpoint hash mismatch: {name}")
        training = folder / "training_inputs.npz"
        with np.load(training, allow_pickle=False) as saved:
            if set(saved.files) != {"input_bytes", "targets"}:
                raise ValueError(f"Unexpected NPZ arrays: {name}")
            x, y = saved["input_bytes"], saved["targets"]
        if y.dtype != np.uint8 or y.shape != (FINAL_STEP, BATCH_SIZE):
            raise ValueError(f"Incomplete or invalid target array: {name}")
        if not np.isin(y, [ord(value) for value in runtime.facts.VALUES]).all():
            raise ValueError(f"Nonlegal training labels: {name}")
        contexts = training_contexts(x, np)
        if regime == "fresh":
            arrays[name] = (x, y, contexts)
        banks[name] = read_json(folder / "bank.json")
        records[name] = dict(folder=str(folder), checkpoint_path=str(checkpoint),
            checkpoint_sha256=digest, parent_checkpoint_path=meta.get("checkpoint_path"),
            parent_checkpoint_sha256=meta.get("checkpoint_sha256"),
            training_npz_path=str(training), training_npz_sha256=sha256(training),
            bank_path=str(folder / "bank.json"), bank_sha256=sha256(folder / "bank.json"),
            metadata_path=str(folder / "metadata.json"), metadata_sha256=sha256(folder / "metadata.json"),
            segment_len=geometry, regime=regime, source_sha256=meta["source_sha256"])
    bank = banks[RUNS[0]]
    if any(other != bank for other in banks.values()):
        raise AssertionError("The four endpoints do not share an identical fixed bank")
    left, right = arrays["fresh32-68101"], arrays["fresh128-68101"]
    if not np.array_equal(left[0], right[0]) or not np.array_equal(left[1], right[1]):
        raise AssertionError("Fresh32 and fresh128 training arrays differ")
    if left[2] != right[2]:
        raise AssertionError("Fresh training context sets differ")
    parent_hashes = {record["parent_checkpoint_sha256"] for record in records.values()}
    if len(parent_hashes) != 1 or None in parent_hashes:
        raise AssertionError("Endpoints do not share the same declared parent checkpoint")
    return bank, left[2] | right[2], records


def capture_sources(out, repo, artifacts):
    """Check recorded frozen source hashes, then archive exact imported sources."""
    hashes = {}
    for record in artifacts.values():
        for name, digest in record["source_sha256"].items():
            if name in hashes and hashes[name] != digest:
                raise AssertionError(f"Run source hashes disagree: {name}")
            source = (repo / name).resolve()
            if not source.is_relative_to(repo) or sha256(source) != digest:
                raise AssertionError(f"Frozen repository source hash mismatch: {name}")
            hashes[name] = digest
    for name in hashes:
        target = out / "sources" / "repo" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
        if sha256(target) != hashes[name]:
            raise AssertionError(f"Source copy hash mismatch: {name}")
    own = Path(__file__).resolve()
    shutil.copy2(own, out / "sources" / "disjoint_probe.py")
    return dict(repo=hashes, disjoint_probe=sha256(own))


def make_samples(selected, runtime):
    samples = []
    for index, record in enumerate(selected):
        episode = runtime.facts.Episode.from_dict(record["episode"])
        for rank, query in enumerate(episode.queries):
            prefix, target, _ = runtime.facts.encode_example(episode, query, "online")
            if len(prefix) != 95 or prefix[:64] != context_bytes(episode):
                raise AssertionError("Evaluation prefix includes an answer or wrong fact context")
            if prefix[64:] != runtime.facts.render_query(query):
                raise AssertionError("Evaluation query is not the label-independent prefix")
            samples.append(dict(episode=index, base_pair=record["base_pair"],
                variant=record["variant"], source_candidate_index=record["source_candidate_index"],
                rank=rank, key=query.key, prefix=prefix.decode("ascii"), target=target))
    if len(samples) != BASE_PAIRS * VARIANTS * 2:
        raise AssertionError("Evaluation must contain the full 256 queries")
    return samples


def summarize(rows):
    summaries = []
    for run in RUNS:
        for condition in ("online", "reset_memory"):
            subset = [row for row in rows if row["run"] == run and row["condition"] == condition]
            if len(subset) != 256 or len({(row["episode"], row["rank"]) for row in subset}) != 256:
                raise AssertionError("Missing or duplicate evaluated queries")
            pairs = [subset[index:index + 2] for index in range(0, len(subset), 2)]
            if any([row["rank"] for row in pair] != [0, 1] or pair[0]["episode"] != pair[1]["episode"]
                   for pair in pairs):
                raise AssertionError("Query pairs are not aligned")
            mean = lambda values: sum(values) / len(values)
            summaries.append(dict(run=run, condition=condition, episodes=128, queries=256,
                accuracy=mean([row["correct"] for row in subset]),
                both_correct=mean([int(all(row["correct"] for row in pair)) for pair in pairs]),
                forced_accuracy=mean([row["forced_correct"] for row in subset]),
                forced_both_correct=mean([int(all(row["forced_correct"] for row in pair)) for pair in pairs]),
                first_accuracy=mean([pair[0]["correct"] for pair in pairs]),
                second_accuracy=mean([pair[1]["correct"] for pair in pairs]),
                nll=mean([row["nll"] for row in subset]),
                forced_nll=mean([row["forced_nll"] for row in subset]),
                legal_probability_mass=mean([row["legal_probability_mass"] for row in subset])))
    return summaries


def evaluate_run(name, artifact, samples, runtime, device, deadline, rows, logits_all):
    torch, np = runtime.torch, runtime.np
    before_deadline(deadline)
    model, config, payload = runtime.pretrain.load_checkpoint(artifact["checkpoint_path"], map_location="cpu")
    before_deadline(deadline)
    info = payload.get("training_info", {})
    if ((config.segment_len, config.memory_segment_len, config.memory_batch_size) !=
            (artifact["segment_len"], 8, 32)
            or info.get("parent_init_seed") != 68101 or info.get("steps") != FINAL_STEP
            or info.get("regime") != artifact["regime"] or info.get("task") != "fixed-set-generalization"
            or info.get("fixed_final_checkpoint") is not True
            or info.get("parent_checkpoint_sha256") != artifact["parent_checkpoint_sha256"]
            or payload.get("source_info", {}).get("source_sha256") != artifact["source_sha256"]
            or payload.get("source_info", {}).get("repo_commit") != FROZEN_COMMIT):
        raise AssertionError(f"Loaded checkpoint protocol/provenance mismatch: {name}")
    model.to(device).eval()
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with torch.no_grad():
        for condition in ("online", "reset_memory"):
            context = (runtime.pretrain.reset_memory_at_segments(model)
                       if condition == "reset_memory" else nullcontext())
            with context:
                for start in range(0, len(samples), BATCH_SIZE):
                    before_deadline(deadline)
                    batch = samples[start:start + BATCH_SIZE]
                    x = torch.tensor([list(row["prefix"].encode("ascii")) for row in batch], device=device)
                    y = torch.tensor([row["target"] for row in batch], device=device)
                    logits = model(x)[:, -1]
                    scores = {key: value.cpu().tolist()
                              for key, value in runtime.common.score_logits(logits, y).items()}
                    logits_all.append(logits.detach().cpu().numpy())
                    for index, sample in enumerate(batch):
                        row = dict(run=name, condition=condition, **sample)
                        row.update({key: value[index] for key, value in scores.items()})
                        row.update(correct=int(row["prediction"] == row["target"]),
                                   forced_correct=int(row["forced_prediction"] == row["target"]))
                        rows.append(row)
            runtime.common.sync(device)
            before_deadline(deadline)
    if not all(torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in before.items()):
        raise AssertionError(f"Evaluation changed outer checkpoint tensors: {name}")
    if not torch.equal(cpu_rng, torch.random.get_rng_state()):
        raise AssertionError(f"Evaluation consumed CPU training RNG: {name}")
    if gpu_rng is not None and not torch.equal(gpu_rng, torch.cuda.get_rng_state(device)):
        raise AssertionError(f"Evaluation consumed CUDA training RNG: {name}")
    if sha256(artifact["checkpoint_path"]) != artifact["checkpoint_sha256"]:
        raise AssertionError(f"Checkpoint file changed: {name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(outer_checkpoint_tensors_unchanged=True, training_rng_unchanged=True,
                checkpoint_file_unchanged=True)


def verify_sources(out, repo, hashes):
    changed = [name for name, digest in hashes["repo"].items()
               if sha256(repo / name) != digest or sha256(out / "sources" / "repo" / name) != digest]
    if (sha256(Path(__file__).resolve()) != hashes["disjoint_probe"]
            or sha256(out / "sources" / "disjoint_probe.py") != hashes["disjoint_probe"]):
        changed.append("disjoint_probe.py")
    if changed:
        raise AssertionError(f"Sources changed during evaluation: {changed}")


def run(args):
    if not 1 <= args.max_seconds <= 300 or args.threads < 1:
        raise ValueError("Require positive threads and a wall-clock cap of at most 300 seconds")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    repo, root, out = args.repo.resolve(), args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(status="running", mode="training-disjoint-novel-values-amendment",
        started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        selection=dict(seed=SELECTION_SEED, candidate_cap=CANDIDATES, variants_per_base_pair=VARIANTS,
            base_pairs=BASE_PAIRS, order="base_pair then acceptance variant",
            rule="First four distinct full64-byte fact contexts per pair absent from either fresh training NPZ; both original context values excluded"),
        evaluation=dict(runs=list(RUNS), parent_init_seed=68101, final_step=FINAL_STEP,
            conditions=["online", "reset_memory"], batch_size=BATCH_SIZE, episodes=128,
            queries_per_run_condition=256, output_vocabulary=256))
    json_file(out / "metadata.json", meta)
    runtime, source_hashes, artifacts = None, None, None
    rows, logits_all = [], []
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if commit != FROZEN_COMMIT:
            raise ValueError(f"--repo must be frozen at {FROZEN_COMMIT}; got {commit}")
        runtime = bootstrap(repo)
        torch, np = runtime.torch, runtime.np
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        meta.update(repo_commit=commit, environment=dict(python=sys.version, torch=str(torch.__version__),
            numpy=np.__version__, platform=platform.platform(), device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            deterministic_algorithms=True, tf32=False, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]))
        bank, excluded, artifacts = load_artifacts(root, runtime, deadline)
        source_hashes = capture_sources(out, repo, artifacts)
        meta.update(source_sha256=source_hashes, run_artifacts=artifacts,
                    fresh_training_arrays_identical=True, banks_identical=True,
                    branch_sha256=sha256(root / "branch.json"))
        selected, counters = select_episodes(bank, excluded, runtime, deadline)
        meta["selection"].update(counters)
        json_file(out / "episodes.json", dict(selection=meta["selection"], episodes=selected))
        meta["episodes_sha256"] = sha256(out / "episodes.json")
        samples = make_samples(selected, runtime)
        json_file(out / "metadata.json", meta)
        checks = {}
        for name in RUNS:
            checks[name] = evaluate_run(name, artifacts[name], samples, runtime, device, deadline,
                                        rows, logits_all)
            print(json.dumps(dict(completed_run=name, queries=512)), flush=True)
        summary = summarize(rows)
        if len(rows) != 2048 or len({(row["run"], row["condition"], row["episode"], row["rank"])
                                    for row in rows}) != 2048:
            raise AssertionError("Missing or duplicate complete evaluation rows")
        json_file(out / "summary.json", summary)
        runtime.common.write_csv(out / "summary.csv", summary)
        meta.update(status="succeeded", evaluation_checks=checks, evaluation_rows=len(rows), summary=summary)
    except BaseException as error:
        meta.update(status="failed", failure=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        changed = None
        if runtime is not None and rows:
            runtime.common.write_csv(out / "predictions.csv", rows)
            logits = runtime.np.concatenate(logits_all)
            if logits.shape != (len(rows), 256):
                meta.update(status="failed", alignment_failure=str(logits.shape))
                changed = AssertionError("Saved logits and prediction rows are misaligned")
            runtime.np.save(out / "answer_logits.npy", logits)
            meta.update(answer_logits_sha256=sha256(out / "answer_logits.npy"),
                        predictions_sha256=sha256(out / "predictions.csv"))
        try:
            if source_hashes is not None:
                verify_sources(out, repo, source_hashes)
            if artifacts is not None:
                for name, artifact in artifacts.items():
                    for field in ("training_npz", "checkpoint", "bank", "metadata"):
                        if sha256(artifact[field + "_path"]) != artifact[field + "_sha256"]:
                            raise AssertionError(f"Input {field} changed: {name}")
            before_deadline(deadline)
        except BaseException as error:
            changed = error
            meta.update(status="failed", source_or_input_recheck_failure=repr(error))
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        json_file(out / "metadata.json", meta)
        if changed is not None:
            raise changed


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--max-seconds", type=int, default=300)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
