"""Bounded independent GPU replay of saved KV-control evaluation artifacts.

This verifies saved evaluation scores and portable fast-state continuation. It
does not reproduce training or the separate initial-checkpoint evaluations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback


FROZEN_COMMIT = "ff62df0f85ff7cbf62f3209755ea066959aa7085"
BATCH_SIZE = 16
SEED = 90301
MAX_SECONDS = 180
ABS_TOLERANCE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_file(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf8")


def inside(path, parent):
    try:
        Path(path).relative_to(parent)
        return True
    except ValueError:
        return False


def semantic_key(row):
    return row["cell"], row["arm"], int(row["episode"]), int(row["rank"])


def select_groups(episodes, suite):
    """Derive the required cells from saved data, preserving all query ranks."""
    requested = [("acquire2", "online")]
    if suite == "full" and "pressure_k8_d16" in episodes:
        requested.extend(("pressure_k8_d16", arm) for arm in ("online", "continuous", "periodic"))
    groups = []
    for cell, arm in requested:
        if cell not in episodes or len(episodes[cell]) < BATCH_SIZE:
            raise AssertionError(f"{cell} must have at least {BATCH_SIZE} saved episodes")
        batch = episodes[cell][:BATCH_SIZE]
        query_counts = {len(ep["queries"]) for ep in batch}
        if len(query_counts) != 1 or next(iter(query_counts)) < 1:
            raise AssertionError(f"Nonrectangular queries in {cell}")
        groups.append(dict(cell=cell, arm=arm, episodes=BATCH_SIZE,
                           queries_per_episode=next(iter(query_counts)),
                           queries=sum(len(ep["queries"]) for ep in batch)))
    return groups


def compare_scores(actual, expected, saved_predictions, *, tolerance=ABS_TOLERANCE):
    """Compare every score coordinate with absolute tolerance and exact argmax."""
    import numpy as np

    if actual.shape != expected.shape or actual.shape[-1] != 16:
        raise AssertionError("Rechecked and saved scores must have the same [..., 16] shape")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Nonfinite rechecked or saved scores")
    if saved_predictions.shape != actual.shape[:-1]:
        raise AssertionError("Saved prediction geometry does not match scores")
    absolute_delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    actual_predictions = actual.argmax(axis=-1)
    reference_predictions = expected.argmax(axis=-1)
    score_mismatches = int(np.count_nonzero(absolute_delta > tolerance))
    argmax_mismatches = int(np.count_nonzero(actual_predictions != reference_predictions))
    saved_argmax_mismatches = int(np.count_nonzero(reference_predictions != saved_predictions))
    return dict(passed=bool(score_mismatches == argmax_mismatches == saved_argmax_mismatches == 0),
                max_abs_delta=float(absolute_delta.max()), compared_coordinates=int(actual.size),
                score_coordinate_mismatches=score_mismatches, argmax_mismatches=argmax_mismatches,
                saved_prediction_mismatches=saved_argmax_mismatches)


def main(args):
    started = time.perf_counter()
    deadline = started + MAX_SECONDS
    repo, root, out = args.repo.resolve(), args.root.resolve(), args.out.resolve()
    if inside(out, repo) or out == root or inside(root, out):
        raise ValueError("--out must be a new verification directory outside the frozen repository")
    # Reject overlap before creating even the verification source copy.
    if root.is_dir() and any(child.is_dir() and inside(out, child.resolve()) for child in root.iterdir()):
        raise ValueError("--out must not be inside an existing experiment directory")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="mismatch", started_utc=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], repo=str(repo), root=str(root),
                  frozen_commit=FROZEN_COMMIT, device="cuda", batch_size=BATCH_SIZE,
                  seed=SEED, max_seconds=MAX_SECONDS, absolute_tolerance=ABS_TOLERANCE,
                  interpretation="Saved-evaluation replay only; no training reproduction or initial-evaluation recheck.",
                  runs=[], groups=[], queries=0, compared_coordinates=0, max_abs_delta=0.0)
    inputs = {}
    source_hashes = {}
    timer_armed = False
    old_handler = None

    def check_deadline():
        if time.perf_counter() >= deadline:
            raise TimeoutError("GPU verification exceeded its internal 180-second budget")

    def alarm_handler(_signum, _frame):
        raise TimeoutError("GPU verification exceeded its internal 180-second budget")

    def remember(path, expected=None):
        check_deadline()
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise AssertionError(f"Artifact SHA256 mismatch: {path}")
        if str(path) in inputs and inputs[str(path)] != digest:
            raise AssertionError(f"Artifact changed during verification: {path}")
        inputs[str(path)] = digest
        return digest

    def validate_sources(mapping, archive=None):
        if not isinstance(mapping, dict) or not mapping:
            raise AssertionError("Missing source hashes")
        for name, expected in mapping.items():
            candidate = (repo / name).resolve()
            if not inside(candidate, repo) or not candidate.is_file():
                raise AssertionError(f"Invalid frozen-source path: {name}")
            if name in source_hashes and source_hashes[name] != expected:
                raise AssertionError(f"Conflicting source hashes for {name}")
            remember(candidate, expected)
            source_hashes[name] = expected
            if archive is not None:
                archived = (archive / name).resolve()
                if not inside(archived, archive.resolve()) or not archived.is_file():
                    raise AssertionError(f"Missing archived source: {name}")
                remember(archived, expected)

    try:
        if hasattr(signal, "setitimer"):
            old_handler = signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.perf_counter()))
            timer_armed = True
        report["signal_deadline_enabled"] = timer_armed
        original_script = Path(__file__).resolve()
        copied_script = out / "recheck_gpu.py"
        shutil.copyfile(original_script, copied_script)
        report["verification_source"] = dict(original=str(original_script), copy=str(copied_script),
                                              sha256=sha256(copied_script))
        remember(original_script, report["verification_source"]["sha256"])
        remember(copied_script, report["verification_source"]["sha256"])
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
        if commit != FROZEN_COMMIT:
            raise AssertionError(f"Expected frozen HEAD {FROZEN_COMMIT}, found {commit}")
        tracked_changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True, timeout=10)
        if tracked_changes.strip():
            raise AssertionError("Frozen repository has tracked modifications")
        pipeline_path = root / "pipeline.json"
        remember(pipeline_path)
        pipeline = json.loads(pipeline_path.read_text(encoding="utf8"))
        if pipeline.get("status") != "succeeded":
            raise AssertionError("The pipeline must have succeeded before GPU rechecking")
        successful = [item for item in pipeline["runs"] if item.get("status") == "succeeded"]
        if not successful or len({item["name"] for item in successful}) != len(successful):
            raise AssertionError("The pipeline needs a nonempty, unique successful run list")
        report["pipeline_status"] = pipeline["status"]
        report["skipped_runs"] = [dict(name=item.get("name"), status=item.get("status"))
                                  for item in pipeline["runs"] if item.get("status") != "succeeded"]
        # Set these before importing torch or any frozen repository modules.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(repo))
        import numpy as np
        import torch

        runner = importlib.import_module("scripts.kv_representation_control")
        streams = importlib.import_module("clmm.pretrain.kv_streams")
        control = importlib.import_module("clmm.pretrain.kv_control")
        neural_memory = importlib.import_module("titans_pytorch.neural_memory")
        module_paths = {
            "scripts.kv_representation_control": "scripts/kv_representation_control.py",
            "clmm.pretrain.kv_streams": "clmm/pretrain/kv_streams.py",
            "clmm.pretrain.kv_control": "clmm/pretrain/kv_control.py",
            "titans_pytorch.neural_memory": "titans_pytorch/neural_memory.py",
        }
        for module in (runner, streams, control, neural_memory):
            if Path(module.__file__).resolve() != repo / module_paths[module.__name__]:
                raise AssertionError(f"Imported source is outside the frozen repo: {module.__name__}")
        if (runner.REPO.resolve() != repo or runner.common.REPO.resolve() != repo
                or Path(runner.common.__file__).resolve() != repo / "scripts/fact_revision_experiment.py"):
            raise AssertionError("Frozen runner resolved an unexpected repository")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this GPU verification")
        device = torch.device("cuda")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        packages = {distribution.metadata["Name"]: distribution.version
                    for distribution in importlib.metadata.distributions()
                    if distribution.metadata.get("Name")}
        report["environment"] = dict(
            python=sys.version, executable=sys.executable, platform=platform.platform(),
            torch=str(torch.__version__), numpy=str(np.__version__), cuda_runtime=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), gpu_name=properties.name,
            gpu_total_memory_bytes=int(properties.total_memory),
            gpu_compute_capability=[int(properties.major), int(properties.minor)],
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
            matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
            package_versions=dict(sorted(packages.items())))

        def rng_snapshot():
            return (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone(),
                    np.random.get_state(), random.getstate())

        def assert_rng_unchanged(before):
            after = rng_snapshot()
            numpy_equal = (before[2][0] == after[2][0] and np.array_equal(before[2][1], after[2][1])
                           and before[2][2:] == after[2][2:])
            if (not torch.equal(before[0], after[0]) or not torch.equal(before[1], after[1])
                    or not numpy_equal or before[3] != after[3]):
                raise AssertionError("Evaluation consumed CPU, CUDA, numpy, or Python RNG state")

        all_passed = True
        for item in successful:
            check_deadline()
            name = item["name"]
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise AssertionError("Run names must be direct child directories")
            run_path = (root / name).resolve()
            if not inside(run_path, root) or inside(out, run_path):
                raise AssertionError("Verification output must not overlap a saved run")
            meta_path = run_path / "metadata.json"
            remember(meta_path)
            metadata = json.loads(meta_path.read_text(encoding="utf8"))
            if metadata.get("status") != "succeeded" or metadata.get("source_commit") != FROZEN_COMMIT:
                raise AssertionError(f"Run {name} is not a successful frozen-source evaluation")
            validate_sources(metadata["source_sha256"], run_path / "sources")
            checkpoint = run_path / "checkpoint.pt"
            if checkpoint.is_file():
                expected_digest = metadata.get("checkpoint_sha256")
                checkpoint_origin = "run_checkpoint"
            else:
                parent = metadata.get("args", {}).get("checkpoint")
                if not parent or not metadata.get("parent_sha256"):
                    raise AssertionError(f"Run {name} has no checkpoint or verified parent")
                checkpoint = Path(parent).expanduser()
                if not checkpoint.is_absolute():
                    checkpoint = repo / checkpoint
                expected_digest = metadata["parent_sha256"]
                checkpoint_origin = "metadata_parent"
            checkpoint = checkpoint.resolve()
            if not expected_digest:
                raise AssertionError(f"Run {name} is missing its checkpoint SHA256")
            checkpoint_digest = remember(checkpoint, expected_digest)
            model, payload = runner.load_checkpoint(checkpoint, device)
            if payload.get("format") != "clmm.kv.control.v1" or payload.get("source_commit") != FROZEN_COMMIT:
                raise AssertionError("Checkpoint is not the expected frozen KV-control format")
            validate_sources(payload["source_sha256"])
            if metadata.get("method") != model.method or metadata.get("config") != model.config.to_dict():
                raise AssertionError("Evaluation metadata does not match its checkpoint")
            if model.config.nvalues != 16:
                raise AssertionError("The frozen control must produce all 16 value coordinates")
            evaluation = run_path / "evaluation"
            episodes_path = evaluation / "episodes.json"
            predictions_path = evaluation / "predictions.csv"
            scores_path = evaluation / "scores.npy"
            remember(episodes_path)
            remember(predictions_path, metadata["evaluation"]["predictions_sha256"])
            remember(scores_path, metadata["evaluation"]["scores_sha256"])
            episodes = json.loads(episodes_path.read_text(encoding="utf8"))
            with predictions_path.open(newline="", encoding="utf8") as handle:
                rows = list(csv.DictReader(handle))
            scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
            if scores.shape != (len(rows), 16) or scores.dtype != np.float32:
                raise AssertionError("Saved score array geometry or dtype is unexpected")
            row_map = {}
            for index, row in enumerate(rows):
                key = semantic_key(row)
                if key in row_map:
                    raise AssertionError(f"Duplicate saved semantic prediction: {key}")
                row_map[key] = index
            groups = select_groups(episodes, metadata["args"]["suite"])
            run_report = dict(name=name, method=model.method, checkpoint_path=str(checkpoint),
                              checkpoint_sha256=checkpoint_digest, checkpoint_origin=checkpoint_origin,
                              checkpoint_format=payload["format"], config=model.config.to_dict(),
                              source_commit=payload["source_commit"], groups=groups,
                              planned_queries=sum(group["queries"] for group in groups),
                              original_environment=metadata.get("environment", {}), queries=0)
            report["runs"].append(run_report)
            model.eval()
            slow = {key: value.detach().clone() for key, value in model.state_dict().items()}
            rng = rng_snapshot()
            with torch.no_grad():
                for group in groups:
                    check_deadline()
                    group_started = time.perf_counter()
                    cell, arm = group["cell"], group["arm"]
                    batch = episodes[cell][:BATCH_SIZE]
                    plans = [streams.replay_plan(ep, arm) for ep in batch]
                    geometry = {(plan["writes"], plan["boundary"]) for plan in plans}
                    if len(geometry) != 1 or any(plan["boundary"] != ep["boundary"]
                                                 for plan, ep in zip(plans, batch)):
                        raise AssertionError("Saved episodes do not yield rectangular replay plans")
                    writes, boundary = next(iter(geometry))
                    if not 1 <= boundary <= writes:
                        raise AssertionError("Invalid saved episode boundary")
                    key_events = torch.tensor([[event["key"] for event in plan["events"]]
                                               for plan in plans], dtype=torch.long, device=device)
                    value_events = torch.tensor([[event["value"] for event in plan["events"]]
                                                 for plan in plans], dtype=torch.long, device=device)
                    # Query keys come exclusively from episodes, without targets.
                    queries = torch.tensor([[query["key"] for query in ep["queries"]]
                                            for ep in batch], dtype=torch.long, device=device)
                    state = model.initial_state(BATCH_SIZE)
                    serialized_bytes = None
                    for index in range(writes):
                        check_deadline()
                        state = model.write(key_events[:, index], value_events[:, index], state)
                        if index + 1 == boundary:
                            raw = io.BytesIO()
                            torch.save(model.export_state(state), raw)
                            serialized_bytes = raw.tell()
                            raw.seek(0)
                            restored = torch.load(raw, map_location="cpu", weights_only=True)
                            state = model.import_state(restored)
                            if state.batch_size != BATCH_SIZE or state.write_count != boundary:
                                raise AssertionError("Boundary reload changed the batch or write count")
                    if serialized_bytes is None or state.write_count != writes:
                        raise AssertionError("Boundary roundtrip or complete replay was skipped")
                    snapshot = model.export_state(state)
                    actual = model.read(queries, state).cpu().numpy()
                    if not runner.equal_tree(snapshot, model.export_state(state)):
                        raise AssertionError("Key-only read mutated fast state")
                    expected = np.empty_like(actual)
                    predictions = np.empty(actual.shape[:-1], dtype=np.int64)
                    for episode_index, ep in enumerate(batch):
                        for rank, query in enumerate(ep["queries"]):
                            row_index = row_map[(cell, arm, episode_index, rank)]
                            row = rows[row_index]
                            if int(row["key"]) != query["key"] or int(row["target"]) != query["target"]:
                                raise AssertionError("Saved semantic row does not match its episode query")
                            expected[episode_index, rank] = scores[row_index]
                            predictions[episode_index, rank] = int(row["prediction"])
                    statistics = compare_scores(actual, expected, predictions)
                    result = dict(run=name, cell=cell, arm=arm, episodes=BATCH_SIZE,
                                  queries=int(actual.shape[0] * actual.shape[1]),
                                  queries_per_episode=int(actual.shape[1]), writes_per_episode=int(writes),
                                  boundary_write_count=int(boundary), boundary_serialized_bytes=int(serialized_bytes),
                                  elapsed_seconds=time.perf_counter() - group_started, **statistics)
                    if not all(torch.equal(value, model.state_dict()[key]) for key, value in slow.items()):
                        raise AssertionError("Replay changed slow parameters or codebook buffers")
                    assert_rng_unchanged(rng)
                    result.update(outer_parameters_unchanged=True, rng_unchanged=True, read_state_unchanged=True)
                    report["groups"].append(result)
                    report["queries"] += result["queries"]
                    report["compared_coordinates"] += result["compared_coordinates"]
                    report["max_abs_delta"] = max(report["max_abs_delta"], result["max_abs_delta"])
                    run_report["queries"] += result["queries"]
                    all_passed = all_passed and result["passed"]
                    print(json.dumps(result, allow_nan=False), flush=True)
            run_report.update(outer_parameters_unchanged=True, rng_unchanged=True)
            del model, payload, slow, scores
        check_deadline()
        for path, digest in inputs.items():
            if sha256(path) != digest:
                raise AssertionError(f"Verification input changed: {path}")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip() != FROZEN_COMMIT:
            raise AssertionError("Frozen repository HEAD changed during verification")
        report["planned_queries"] = sum(run["planned_queries"] for run in report["runs"])
        if report["queries"] != report["planned_queries"]:
            raise AssertionError("Not all planned saved queries were rechecked")
        report.update(status="passed" if all_passed else "mismatch", artifacts_unchanged=True,
                      frozen_source_sha256=source_hashes,
                      run_count=len(report["runs"]), group_count=len(report["groups"]))
        check_deadline()
    except BaseException as error:
        report.update(status="mismatch", failure=repr(error), traceback=traceback.format_exc())
    finally:
        if timer_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        report.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started, input_sha256=inputs)
        json_file(out / "verification.json", report)
    print(json.dumps(dict(status=report["status"], runs=len(report["runs"]),
                          groups=len(report["groups"]), queries=report["queries"],
                          max_abs_delta=report["max_abs_delta"], verification=str(out / "verification.json"))), flush=True)
    return 0 if report["status"] == "passed" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
