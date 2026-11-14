"""Separate saved-artifact auditor for the exploratory 32/128 attention control.

Uses only low-level independent audit helpers; never imports experiment drivers,
data generators, or model code. The main v1 analyzer and its decisions are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import analyze_fact_revision as A

HERE = Path(__file__).resolve().parent
MAIN_ANALYZER_SHA256 = "3718b7b68a3f766c350c241599c9a88363b00d85d0b6171598fef08274bbf025"
PARENT_HASH = "a07762534f0b6311ec68c08b0d9f9c0ce74ec71577138eb13279a4a5dffbb24d"
SCOPE = "Single fixed-parent exploratory attention-access diagnostic; saved evidence audited, no training or inference rerun; does not rescue main v1."


def canonical_training(data):
    digest = hashlib.sha256()
    for name in sorted(data):
        array = np.ascontiguousarray(data[name])
        descriptor = json.dumps([name, array.dtype.str, list(array.shape)], separators=(",", ":"))
        digest.update(descriptor.encode("ascii") + b"\n")
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_control_checkpoint(path, parent, geometry, meta, expected_commit):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    A.equal(checkpoint["format_version"], parent["format_version"], "control checkpoint format")
    A.equal(checkpoint["architecture"], parent["architecture"], "control architecture")
    expected_config = dict(parent["config"], segment_len=geometry)
    A.equal(checkpoint["config"], expected_config, "only attention geometry differs from parent")
    A.finite_state(checkpoint["model_state"])
    A.equal(set(checkpoint["model_state"]), set(parent["model_state"]), "unchanged tensor names")
    for name, value in checkpoint["model_state"].items():
        A.equal(tuple(value.shape), tuple(parent["model_state"][name].shape), "unchanged tensor shape")
        A.equal(value.dtype, parent["model_state"][name].dtype, "unchanged tensor dtype")
    source = checkpoint["source_info"]
    A.equal(source["repo_commit"], expected_commit, "checkpoint source commit")
    A.equal(source["source_sha256"], meta["source_sha256"], "checkpoint source snapshot")
    A.equal(source["supplemental_diagnostic"], "fact_attention_control", "checkpoint diagnostic label")
    info = checkpoint["training_info"]
    expected = dict(task="fact-attention-control", primary_parent_init_seed=68101,
        steps=1000, parent_checkpoint_sha256=PARENT_HASH, train_seed=78122,
        fixed_final_checkpoint=True, optimizer="AdamW", lr=.0005, weight_decay=.01, gradient_clip=1.,
        facts=2, delay=0, revision_fraction=0., condition="online", input_length=95,
        answer_vocabulary=256, attention_segment_len=geometry, record_bytes=32)
    for key, value in expected.items():
        A.equal(info[key], value, "control checkpoint training provenance " + key)
    return checkpoint


def validate_training(run, meta, checkpoint):
    import torch
    A.checked_hash(run/"training_inputs.npz", meta["training_input_sha256"], "control training archive")
    A.checked_hash(run/"training_plan.json", meta["training_plan_sha256"], "control training plan")
    with np.load(run/"training_inputs.npz", allow_pickle=False) as archive:
        A.equal(set(archive.files), {"input_bytes", "targets", "episode_seeds", "query_indices"}, "control training arrays")
        data = {key: archive[key] for key in archive.files}
    A.equal(data["input_bytes"].shape, (1000, 16, 95), "full fixed input budget")
    A.require(data["input_bytes"].dtype == np.uint8, "Control inputs are not bytes")
    for key in ("targets", "episode_seeds", "query_indices"):
        A.equal(data[key].shape, (1000, 16), "control array shape " + key)
    A.require(data["targets"].dtype == data["query_indices"].dtype == np.uint8, "Control label/index byte dtypes")
    A.require(data["episode_seeds"].dtype == np.int64, "Control seed dtype")
    A.require(((data["targets"] >= 97) & (data["targets"] <= 112)).all(), "Control label outside full-vocabulary legal targets")
    A.require((data["query_indices"] < 2).all(), "Control query rank outside K2")
    canonical = canonical_training(data)
    A.equal(canonical, meta["training_data_sha256"], "canonical training data hash")
    A.equal(canonical, checkpoint["training_info"]["training_data_sha256"], "checkpoint training data hash")
    plan, trace = A.read_json(run/"training_plan.json"), A.csv_rows(run/"training.csv")
    A.equal(len(plan), 1000, "control training plan budget")
    A.equal(len(trace), 1000, "control training trace budget")
    rng = np.random.default_rng(78122)
    geometry = meta["attention_segment_len"]
    for step in range(1000):
        seeds = rng.integers(1, 1_000_000_000, size=16)
        queries = rng.integers(0, 2, size=16)
        A.require(np.array_equal(seeds, data["episode_seeds"][step]), "Control episode seed schedule changed")
        A.require(np.array_equal(queries, data["query_indices"][step]), "Control query schedule changed")
        expected = dict(step=step+1, facts=2, delay=0, revision_fraction=0., split="train",
                        condition="online", input_length=95, episode_seeds=seeds.tolist(), query_indices=queries.tolist())
        A.equal(plan[step], expected, "control fixed training plan")
        for index in range(16):
            A.parse_training_prefix(data["input_bytes"][step, index].tobytes(), data["targets"][step, index],
                                    2, 0, 0., int(queries[index]))
        row = trace[step]
        for key, value in dict(step=step+1, facts=2, delay=0, batch=16, input_length=95,
                               attention_segment_len=geometry, input_bytes=16*95, supervised_answers=16).items():
            A.equal(int(row[key]), value, "control trace " + key)
        A.near(row["revision_fraction"], 0., "control revision fraction", 0.)
        for key in ("loss", "accuracy", "gradient_norm", "elapsed_seconds"):
            A.require(math.isfinite(float(row[key])), "Nonfinite control training metric")
        A.require(0 <= float(row["accuracy"]) <= 1 and float(row["gradient_norm"]) >= 0, "Invalid control training metric")
    A.equal(meta["completed_steps"], 1000, "completed control steps")
    A.equal(meta["supervised_answers"], 16000, "completed control answer budget")
    A.equal(meta["input_bytes"], 1000*16*95, "completed control input budget")
    progress = A.read_json(run/"training_progress.json")
    A.equal(progress, dict(completed_steps=1000, planned_steps=1000, fixed_final_checkpoint_saved=True), "fixed final progress")
    optimizer = torch.load(run/"optimizer.pt", map_location="cpu", weights_only=True)
    A.require(bool(optimizer["state"]), "Missing optimizer update state")
    for state in optimizer["state"].values():
        A.equal(float(state["step"]), 1000., "optimizer step budget")
        A.finite_state({name: value for name, value in state.items() if isinstance(value, torch.Tensor)})
    for group in optimizer["param_groups"]:
        A.equal(group["lr"], .0005, "optimizer learning rate")
        A.equal(group["weight_decay"], .01, "optimizer weight decay")
    return dict(data=data, plan=plan, canonical_hash=canonical)


def verify_noop(logits, rows, geometry):
    positions = {}
    for i, row in enumerate(rows):
        key = A.row_key(row)
        A.require(key not in positions, "Duplicate no-op pairing sample")
        positions[key] = i
    differences = []
    for key, i in positions.items():
        if key[-1] != "online":
            continue
        j = positions[key[:-1] + ("reset_memory",)]
        differences.append(float(np.max(np.abs(logits[i].astype(float)-logits[j].astype(float)))))
        if geometry == 128:
            A.require(np.array_equal(logits[i], logits[j]), "128-byte short normal/reset logits differ: expected exact no-op")
    return dict(paired_questions=len(differences), max_abs_logit_difference=max(differences),
                exact_noop_verified=geometry == 128, expected_noop=geometry == 128)


def validate_evaluation(run, meta):
    for file, field in (("evaluation_data.jsonl.gz", "evaluation_data_sha256"),
                         ("answer_logits.npy", "answer_logits_sha256"), ("predictions.csv", "predictions_sha256")):
        A.checked_hash(run/file, meta[field], "control evaluation " + file)
    import gzip
    raw = gzip.decompress((run/"evaluation_data.jsonl.gz").read_bytes())
    canonical = A.digest_bytes(raw)
    A.equal(canonical, meta["evaluation_data_canonical_sha256"], "control canonical evaluation hash")
    A.equal((run/"evaluation_data_canonical.sha256").read_text().strip(), canonical, "control canonical hash sidecar")
    specs = A.expected_specs(meta["args"])
    ledger, episodes, actual, current = {}, {}, {}, None
    for line in raw.decode("utf8").splitlines():
        item = json.loads(line)
        if item["type"] == "episodes":
            spec = item["spec"]
            current = spec["cell"], spec["family"]
            A.require(current in specs and current not in actual, "Unknown/duplicate attention cell")
            A.equal(spec, specs[current], "attention gate specification")
            actual[current] = spec
            A.equal(len(item["episodes"]), 64, "attention episodes per cell")
            group = 1 if spec["cell"] == "revise_k1" else 0
            rng = np.random.default_rng(78123 + spec["facts"]*101 + group*10007)
            seeds = rng.integers(2_000_000_000, 3_000_000_000, size=64)
            A.require(len(set(map(int, seeds))) == 64, "Repeated control episode seed")
            for index, ep in enumerate(item["episodes"]):
                A.equal(ep["seed"], int(seeds[index]), "control development seed plan")
                for name, value in dict(facts=spec["facts"], delay=0, revision_fraction=spec["revision_fraction"],
                                        template_family=spec["family"]).items():
                    A.equal(ep[name], value, "control episode specification")
                A.validate_episode(ep)
                episodes[current+(index,)] = ep
        elif item["type"] == "sample":
            row = item["row"]
            A.equal(current, (row["cell"], row["family"]), "control sample dataset")
            ep = episodes[current+(row["episode"],)]
            A.verify_sample(item, ep, 4, 8)
            key = A.row_key(row)
            A.require(key not in ledger, "Duplicate control sample")
            ledger[key] = row
        else:
            raise A.AuditError("Unknown control dataset entry")
    A.equal(set(actual), set(specs), "complete control cell coverage")
    expected = {(cell, family, index, qi, arm) for (cell, family, index), ep in episodes.items()
                for qi in range(len(ep["queries"])) for arm in ("online", "reset_memory")}
    A.equal(set(ledger), expected, "all short queries/conditions present")
    A.equal(len(ledger), 1536, "full 64-episode control evaluation query budget")
    for family in ("seen", "heldout"):
        for index in range(64):
            before, changed = episodes["acquire_k2", family, index], episodes["counterfactual_k2", family, index]
            A.equal(before["seed"], changed["seed"], "control counterfactual seed")
            for left, right in zip(before["events"], changed["events"]):
                A.equal({k: v for k, v in left.items() if k != "value"},
                        {k: v for k, v in right.items() if k != "value"}, "control counterfactual unchanged fields")
                A.equal(right["value"], A.VALUES[(A.VALUES.index(left["value"])+1) % 16], "control counterfactual rotation")
            for left, right in zip(before["queries"], changed["queries"]):
                A.equal(A.query_bytes(left), A.query_bytes(right), "control counterfactual fixed query")
                A.equal(right["target"], A.VALUES[(A.VALUES.index(left["target"])+1) % 16], "control counterfactual target")
    for cell in A.SHORT:
        for index in range(64):
            seen, held = episodes[cell, "seen", index], episodes[cell, "heldout", index]
            for field in ("events", "queries"):
                A.equal([{k: v for k, v in r.items() if k != "template_family"} for r in seen[field]],
                        [{k: v for k, v in r.items() if k != "template_family"} for r in held[field]], "control template semantic pairing")
    saved = A.csv_rows(run/"predictions.csv")
    logits = np.load(run/"answer_logits.npy", allow_pickle=False)
    A.equal(logits.shape, (1536, 256), "control logits shape")
    A.equal(len(saved), 1536, "control CSV count")
    A.equal(meta["evaluation_queries"], 1536, "control recorded query budget")
    A.require(meta["checkpoint_unchanged"], "Control reports outer-state mutation during evaluation")
    A.equal({A.row_key(row) for row in saved}, set(ledger), "CSV/raw control sample identities")
    A.equal(len({A.row_key(row) for row in saved}), len(saved), "unique control CSV samples")
    recomputed = A.independent_scores(logits, [int(row["target"]) for row in saved])
    rows = []
    for i, record in enumerate(saved):
        source = ledger[A.row_key(record)]
        for key, value in source.items():
            A.equal(record[key], str(value), "control CSV/raw agreement " + key)
        row = dict(source)
        for key, values in recomputed.items():
            value = values[i].item()
            if key in ("prediction", "forced_prediction"):
                A.equal(int(record[key]), value, "control prediction from logits")
            else:
                A.near(record[key], value, "control metric from logits " + key)
            row[key] = value
        row.update(correct=int(row["prediction"] == row["target"]),
                   forced_correct=int(row["forced_prediction"] == row["target"]),
                   obsolete=int(row["old_target"] >= 0 and row["prediction"] == row["old_target"]),
                   last_observed_correct=int(row["last_observed_value"] == row["target"]),
                   last_presented_correct=int(row["last_presented_value"] == row["target"]))
        for key in ("correct", "forced_correct", "obsolete"):
            A.equal(int(record[key]), row[key], "control outcome " + key)
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["cell"], row["family"], row["category"], row["condition"]].append(row)
    summaries = A.csv_rows(run/"summary.csv")
    A.equal(len(summaries), len(groups), "control summary coverage")
    seen_summaries = set()
    for summary in summaries:
        key = summary["cell"], summary["family"], summary["category"], summary["condition"]
        A.require(key not in seen_summaries, "Duplicate control summary")
        seen_summaries.add(key)
        group = groups[key]
        A.equal(int(summary["queries"]), len(group), "control summary denominator")
        A.equal(int(summary["episodes"]), len({r["episode_seed"] for r in group}), "control summary episode count")
        for name, metric in dict(accuracy="correct", forced_accuracy="forced_correct", obsolete_rate="obsolete",
                                 nll="nll", legal_probability_mass="legal_probability_mass").items():
            A.near(summary[name], np.mean([r[metric] for r in group]), "control summary " + name)
    reset = A.read_json(run/"reset_diagnostic.json")
    A.equal(reset, meta["reset_diagnostic"], "control reset labels match metadata")
    A.equal(len(reset), 8, "control reset label coverage")
    reset_cells = set()
    geometry = meta["attention_segment_len"]
    for item in reset:
        key = item["cell"], item["family"]
        A.require(key in specs and key not in reset_cells, "Unknown/duplicate reset diagnostic")
        reset_cells.add(key)
        lengths = sorted({row["input_length"] for row in rows if (row["cell"], row["family"]) == key})
        A.equal(item["input_lengths"], lengths, "control reset input lengths")
        A.equal(item["attention_segment_len"], geometry, "control reset geometry")
        A.equal(item["reset_expected_noop"], all(length <= geometry for length in lengths), "control reset expectation")
    noop = verify_noop(logits, rows, geometry)
    return dict(rows=rows, ledger=ledger, canonical_hash=canonical, no_op=noop)


def validate_arm(run, geometry, parent_path, parent, expected_commit, design_hash):
    import torch
    meta = A.check_common(run, parent_path, expected_commit, design_hash, strict=True)
    required_sources = {"scripts/fact_attention_control.py", "scripts/fact_revision_experiment.py",
                        "clmm/pretrain/byte_lm.py", "clmm/pretrain/fact_streams.py",
                        "titans_pytorch/mac_transformer.py", "titans_pytorch/neural_memory.py",
                        "titans_pytorch/memory_models.py", "pyproject.toml", "requirements-lock.txt"}
    A.require(required_sources <= set(meta["source_sha256"]), "Incomplete control source snapshot")
    A.equal(meta["mode"], "attention_control", "control run mode")
    args = meta["args"]
    for key, expected in dict(segment_len=geometry, steps=1000, batch_size=16, train_seed=78122,
        dev_seed=78123, gate_episodes=64, suite="gate", mode="attention_control", lr=.0005,
        block_size=4, buffer_size=8, grid_episodes=32, primary_episodes=128).items():
        A.equal(args[key], expected, "fixed control argument " + key)
    A.equal(meta["parent_checkpoint_sha256"], PARENT_HASH, "fixed original control parent")
    A.equal(meta["source_commit"], expected_commit, "control source commit")
    A.equal(meta["primary_parent_init_seed"], 68101, "control original seed")
    A.equal(meta["attention_segment_len"], geometry, "recorded attention geometry")
    A.equal(meta["parent_config"], parent["config"], "recorded parent configuration")
    A.equal(meta["control_config"], dict(parent["config"], segment_len=geometry), "recorded control configuration")
    A.equal(meta["changed_config_fields"], [] if geometry == 32 else ["segment_len"], "only changed config field")
    A.require(meta["initial_state_tensors_identical"] is True, "Missing exact-copy initialization attestation")
    A.checked_hash(run/"initial_state.pt", meta["initial_state_sha256"], "saved actual initial tensors")
    initial_state = torch.load(run/"initial_state.pt", map_location="cpu", weights_only=True)
    A.finite_state(initial_state)
    A.equal(set(initial_state), set(parent["model_state"]), "initial state tensor names match parent")
    for name, value in initial_state.items():
        A.equal(tuple(value.shape), tuple(parent["model_state"][name].shape), "initial state tensor shape")
        A.equal(value.dtype, parent["model_state"][name].dtype, "initial state tensor dtype")
        A.require(torch.equal(value, parent["model_state"][name]), "Initial tensor differs from original parent: " + name)
    A.equal(meta["parent_parameter_count"], 839248, "parent parameter count")
    A.equal(meta["parameter_count"], 839248, "control parameter count")
    A.equal(meta["training_input_length"], 95, "control training input length")
    A.equal(meta["record_bytes"], 32, "unchanged serialization record size")
    for field in ("local_attention_contains_entire_training_example", "memory_reset_expected_noop_on_training_length"):
        A.equal(meta[field], geometry == 128, field)
    A.require(meta["conditional_supplemental_diagnostic"] and not meta["replaces_primary_result"]
              and not meta["v1_memory_gate_applied"], "Control improperly replaces or applies main v1 gate")
    A.equal(meta["driver_source_snapshot"], "scripts/fact_attention_control.py", "driver source snapshot")
    A.equal(meta["driver_source_sha256"], meta["source_sha256"]["scripts/fact_attention_control.py"], "driver source hash")
    A.checked_hash(run/"checkpoint.pt", meta["output_checkpoint_sha256"], "fixed final control checkpoint")
    checkpoint = load_control_checkpoint(run/"checkpoint.pt", parent, geometry, meta, expected_commit)
    training = validate_training(run, meta, checkpoint)
    evaluation = validate_evaluation(run, meta)
    return dict(meta=meta, training=training, evaluation=evaluation, initial_state=initial_state)


def statistics(arms):
    groups, estimates, contrasts, gates = defaultdict(list), [], [], {}
    for arm, data in arms.items():
        for row in data["evaluation"]["rows"]:
            groups[arm, row["cell"], row["family"], row["category"], row["condition"]].append(row)
    metrics = ("correct", "forced_correct", "nll", "forced_nll", "legal_probability_mass",
               "last_observed_correct", "last_presented_correct")
    for key, rows in groups.items():
        arm, cell, family, category, condition = key
        for metric in metrics + (("obsolete",) if category == "revised" else ()):
            summary = A.cluster_estimate(rows, metric, 5000, A.analysis_seed(78124, repr((key, metric))))
            estimates.append(dict(arm=arm, cell=cell, family=family, category=category, condition=condition,
                                  metric=metric, **summary, scope="one fixed parent; episode-conditional exploratory interval"))
        if condition == "online":
            accuracy = float(np.mean([r["correct"] for r in rows]))
            gates.setdefault(arm, []).append(dict(cell=cell, family=family, category=category,
                full256_accuracy=accuracy, descriptive_threshold=.8, threshold_met=accuracy >= .8,
                v1_normal_minus_reset_gate_applied=False))
    primary = None
    for cell in A.SHORT:
        for family in ("seen", "heldout"):
            category = "revised" if cell == "revise_k1" else "stable"
            for condition in ("online", "reset_memory"):
                a = groups["attention128", cell, family, category, condition]
                b = groups["attention32", cell, family, category, condition]
                for metric in metrics + (("obsolete",) if category == "revised" else ()):
                    summary = A.paired_estimate(a, b, metric, 5000,
                        A.analysis_seed(78124, repr((cell, family, category, condition, metric))))
                    is_primary = (cell, family, condition, metric) == ("acquire_k2", "seen", "online", "correct")
                    row = dict(cell=cell, family=family, category=category, condition=condition,
                        left="attention128", right="attention32", metric=metric, **summary,
                        role="supplemental primary diagnostic" if is_primary else "supplemental descriptive")
                    contrasts.append(row)
                    if is_primary:
                        primary = row
    def k2(arm):
        return next(row for row in gates[arm] if row["cell"] == "acquire_k2" and row["family"] == "seen")
    hit32, hit128 = k2("attention32")["threshold_met"], k2("attention128")["threshold_met"]
    if hit128 and not hit32 and primary["ci_low"] > 0:
        pattern = "Only attention128 meets the descriptive acquisition target, with a positive paired contrast."
    elif hit128 and hit32:
        pattern = "Both K2-only controls meet the descriptive acquisition target."
    elif not hit128 and not hit32:
        pattern = "Neither control meets the descriptive acquisition target; learnability remains unresolved."
    else:
        pattern = "Mixed descriptive outcomes; inspect the paired contrast and intervals."
    decisions = dict(supplemental_certified=True, main_v1_certified=False, replaces_main_v1=False,
        evidence_scope=SCOPE, primary_contrast=primary, descriptive_acquisition=gates, pattern=pattern,
        uncertainty="5000 whole-episode bootstrap draws; both K2 questions clustered; one fixed parent",
        initialization_evidence="Every saved initial tensor independently compared exactly with the original parent and the other control, including shape, dtype, and finite values.",
        limits=["No inference about training-seed variability from one parent.",
            "This diagnostic was selected after the main experiment's acquisition failure and cannot rescue its replay comparison.",
            "Attention128 retains neural memory but supplies direct attention to all short inputs; it is not an independently trained memory-free model.",
            "The normal-minus-reset v1 gate is not applied; reset must be exactly a no-op for attention128 short inputs.",
            "32-byte raw-data record indices describe serialization units, not the attention128 model's actual attention boundaries.",
            "Training and inference were not rerun by this auditor."])
    return estimates, contrasts, decisions


def run(args, out):
    import torch
    A.require(args.root is not None and args.expected_commit is not None, "Provide --root and --expected-commit")
    A.require(re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is not None, "Expected control commit must be a full SHA")
    root = args.root.resolve()
    design_path = root/"attention-control-design.json"
    design, design_hash = A.read_json(design_path), A.sha(design_path)
    A.equal(design["experiment_id"], "fact-acquisition-attention-control-v1", "supplemental design")
    A.equal(design["parent"]["sha256"], PARENT_HASH, "supplemental parent specification")
    A.equal(design["parent"]["fixed_seed"], 68101, "supplemental fixed parent seed")
    for key, value in dict(steps=1000, batch_size=16, train_seed=78122, facts=2, delay=0,
                            revision_fraction=0, lr=.0005).items():
        A.equal(design["training"][key], value, "supplemental fixed design " + key)
    for key, value in dict(dev_seed=78123, episodes=64, suite="gate").items():
        A.equal(design["evaluation"][key], value, "supplemental fixed evaluation " + key)
    A.equal(design["analysis"]["bootstrap_draws"], 5000, "supplemental bootstrap draws")
    A.equal(design["analysis"]["bootstrap_seed"], 78124, "supplemental bootstrap seed")
    parent_path = args.parent or root.parent/"clmm-shared-pretrain-2026-11-14"/"lm-68101"/"checkpoint.pt"
    A.checked_hash(parent_path, PARENT_HASH, "original primary checkpoint")
    parent = A.load_checkpoint_evidence(parent_path)
    A.equal(parent["training_info"]["init_seed"], 68101, "original parent initialization seed")
    A.require(parent["training_info"].get("task") is None and parent["training_info"].get("parent_checkpoint_sha256") is None,
              "Supplement starts from a task-adapted checkpoint")
    arms = {}
    for geometry in (32, 128):
        name = f"attention{geometry}"
        print("Auditing " + name, flush=True)
        arms[name] = validate_arm(root/name, geometry, parent_path, parent, args.expected_commit, design_hash)
    a, b = arms["attention32"], arms["attention128"]
    A.equal(set(a["initial_state"]), set(b["initial_state"]), "paired initial tensor names")
    for name, value in a["initial_state"].items():
        A.require(torch.equal(value, b["initial_state"][name]), "Initial tensor differs between controls: " + name)
    A.equal(a["meta"]["source_sha256"], b["meta"]["source_sha256"], "same control source snapshot")
    A.equal(a["training"]["plan"], b["training"]["plan"], "same complete control training plan")
    for key, array in a["training"]["data"].items():
        A.require(np.array_equal(array, b["training"]["data"][key]), "Training data differs across attention controls: " + key)
    A.equal(a["training"]["canonical_hash"], b["training"]["canonical_hash"], "matched canonical training bytes")
    A.equal(a["evaluation"]["canonical_hash"], b["evaluation"]["canonical_hash"], "matched full evaluation data")
    A.equal(a["evaluation"]["ledger"], b["evaluation"]["ledger"], "matched evaluation sample identities")
    estimates, contrasts, decisions = statistics(arms)
    A.write_csv(out/"estimates.csv", estimates)
    A.write_csv(out/"contrasts.csv", contrasts)
    A.write_json(out/"decisions.json", decisions)
    A.write_json(out/"source-manifest.json", dict(expected_commit=args.expected_commit, design_sha256=design_hash,
        auditor_sha256=A.sha(__file__), main_helper_sha256=A.sha(A.__file__),
        parent=dict(path=str(parent_path), sha256=PARENT_HASH), artifacts=A.manifest(root, ("attention32", "attention128"))))
    return dict(status="verified_supplement", supplemental_certified=True, main_v1_certified=False, replaces_main_v1=False,
        evidence_scope=SCOPE, expected_commit=args.expected_commit, steps_per_arm=1000, answers_per_arm=16000,
        evaluation_queries_per_arm=1536, training_canonical_sha256=a["training"]["canonical_hash"],
        evaluation_canonical_sha256=a["evaluation"]["canonical_hash"],
        reset_diagnostic={name: arm["evaluation"]["no_op"] for name, arm in arms.items()},
        initialization_evidence=decisions["initialization_evidence"],
        output_sha256={name: A.sha(out/name) for name in ("estimates.csv", "contrasts.csv", "decisions.json", "source-manifest.json")})


def selfcheck(out):
    checks = []
    def reject(name, operation):
        try:
            operation()
        except (A.AuditError, KeyError, ValueError):
            checks.append(name)
        else:
            raise A.AuditError("Did not reject: " + name)
    rows = [dict(cell="acquire_k2", family="seen", episode=0, episode_seed=9, query_index=q,
                 condition=arm, key=f"{q:03d}", target=97, category="stable", correct=1)
            for arm in ("online", "reset_memory") for q in (0, 1)]
    logits = np.zeros((4, 256), dtype=np.float32)
    A.require(verify_noop(logits, rows, 128)["exact_noop_verified"], "no-op fixture")
    checks.append("128 short logits must be exactly paired under reset")
    logits[2, 97] = 1e-7
    reject("even small 128 no-op violation", lambda: verify_noop(logits, rows, 128))
    A.require(not verify_noop(logits, rows, 32)["exact_noop_verified"], "32 diagnostic should not require no-op")
    checks.append("32 geometry does not falsely require no-op")
    left = [dict(row, correct=1) for row in rows if row["condition"] == "online"]
    right = [dict(row, correct=0) for row in left]
    contrast = A.paired_estimate(left, list(reversed(right)), "correct", 5000, 78124)
    A.equal((contrast["estimate"], contrast["ci_low"], contrast["ci_high"], contrast["episodes"], contrast["queries"]),
            (1., 1., 1., 1, 2), "whole episode paired contrast")
    checks.append("K2 two-query paired contrast clusters whole episodes")
    reject("mismatched control query identity", lambda: A.pair_rows(left, right[:-1]))
    data = dict(input_bytes=np.zeros((2, 2, 95), dtype=np.uint8), targets=np.full((2, 2), 97, dtype=np.uint8),
                episode_seeds=np.ones((2, 2), dtype=np.int64), query_indices=np.zeros((2, 2), dtype=np.uint8))
    baseline = canonical_training(data)
    data["targets"][0, 0] = 98
    A.require(canonical_training(data) != baseline, "Training canonical hash ignores labels")
    checks.append("canonical control training hash includes target labels")
    key = A.POOLS["train"][0]
    other = A.POOLS["train"][1]
    raw = A.RECORDS["seen"][0].format(key=key, value="a").encode().rjust(32, b" ")
    raw += A.RECORDS["seen"][1].format(key=other, value="b").encode().rjust(32, b" ")
    raw += A.QUERIES["seen"][0].format(key=key).encode().rjust(31, b" ")
    A.parse_training_prefix(raw, 97, 2, 0, 0., 0)
    reject("control label copied from last unrelated fact", lambda: A.parse_training_prefix(raw, 98, 2, 0, 0., 0))
    reject("control query answer fed into input", lambda: A.parse_training_prefix(raw+b"a", 97, 2, 0, 0., 0))
    result = dict(status="selfchecks_passed", supplemental_certified=False, main_v1_certified=False,
                  fixture_scope="Synthetic fixtures only", checks=checks, count=len(checks))
    A.write_json(out/"selfchecks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    A.equal(A.sha(A.__file__), MAIN_ANALYZER_SHA256, "frozen main analyzer helper")
    out = args.out.resolve()
    A.require(out.is_relative_to(HERE) and out != HERE, "Output must be a new subdirectory under analysis")
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = selfcheck(out) if args.selfcheck else run(args, out)
        code = 0
    except BaseException as error:
        result = dict(status="failed", supplemental_certified=False, main_v1_certified=False,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        A.write_json(out/"decisions.json", dict(supplemental_certified=False, main_v1_certified=False,
            conclusion="Supplemental artifact audit failed; no diagnostic conclusion certified.", error=str(error)))
        print(result["traceback"], file=sys.stderr)
        code = 1
    result.update(auditor_sha256=A.sha(__file__), main_helper_sha256=A.sha(A.__file__),
                  command=[sys.executable, *sys.argv])
    A.write_json(out/"verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
