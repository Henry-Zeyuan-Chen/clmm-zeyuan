"""Synthetic failures only: no downloaded runs, models, GPU, or prior experiment."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import audit_paired_acquisition as audit


def strict_fixture():
    rows = []
    context = b"x" * 64
    for condition in ("online", "reset_memory"):
        for rank, (key, target) in enumerate((("123", 97), ("987", 98))):
            prefix = context + f"value of key {key}: ".encode().rjust(31)
            rows.append(dict(family="seen", condition=condition, episode="0", rank=str(rank),
                             key=key, target=str(target), prefix=prefix.decode(),
                             prediction=str(target), correct="1"))
    logits = np.zeros((4, 256), dtype=np.float32)
    for i, row in enumerate(rows):
        logits[i, int(row["target"])] = 3
    return rows, logits


class CorruptionSelfchecks(unittest.TestCase):
    def test_invalid_full_vocabulary_prediction_is_not_hidden_by_forced_choice(self):
        logits = np.zeros((1, 256)); logits[0, 97] = 2; logits[0, 33] = 20
        result = audit.score(logits, [97])
        self.assertEqual(result["prediction"].tolist(), [33])
        self.assertEqual(result["forced_prediction"].tolist(), [97])
        self.assertEqual(result["correct"].tolist(), [0])
        self.assertGreater(result["nll"][0], 17)
        self.assertLess(result["legal_probability_mass"][0], 1e-6)

    def test_uniform_logits_have_known_nll_and_mass(self):
        result = audit.score(np.zeros((2, 256)), [97, 112])
        np.testing.assert_allclose(result["nll"], np.log(256))
        np.testing.assert_allclose(result["forced_nll"], np.log(16))
        np.testing.assert_allclose(result["legal_probability_mass"], 16 / 256)

    def test_nonfinite_logits_rejected(self):
        logits = np.zeros((1, 256)); logits[0, 14] = np.nan
        with self.assertRaises(audit.AuditError): audit.score(logits, [97])

    def test_illegal_target_and_incorrect_shape_rejected(self):
        for logits, target in ((np.zeros((1, 16)), [97]), (np.zeros((1, 256)), [33])):
            with self.assertRaises(audit.AuditError): audit.score(logits, target)

    def test_logit_row_permutation_detected(self):
        rows, logits = strict_fixture()
        with self.assertRaises(audit.AuditError): audit.validate_scores(rows, logits[::-1], strict=True)

    def test_forged_correct_field_detected(self):
        rows, logits = strict_fixture(); rows[0]["correct"] = "0"
        with self.assertRaises(audit.AuditError): audit.validate_scores(rows, logits, strict=True)

    def test_shifted_labels_detected(self):
        rows, logits = strict_fixture(); rows[0]["target"] = "98"
        with self.assertRaises(audit.AuditError): audit.validate_scores(rows, logits, strict=True)

    def test_prefix_extra_answer_or_changed_context_detected(self):
        episode = SimpleNamespace(events=[SimpleNamespace(encoded=b"x" * 32)])
        query = SimpleNamespace(answer_byte=97)
        streams = SimpleNamespace(render_query=lambda q: b"query".rjust(31))
        prefix = b"x" * 32 + b"query".rjust(31)
        audit.validate_prefix_target(prefix, 97, episode, query, streams, "fixture")
        for bad in (prefix + b"a", b"z" + prefix[1:]):
            with self.assertRaises(audit.AuditError):
                audit.validate_prefix_target(bad, 97, episode, query, streams, "fixture")
        with self.assertRaises(audit.AuditError):
            audit.validate_prefix_target(prefix, 98, episode, query, streams, "fixture")

    def test_forced_choice_and_obsolete_corruption_detected(self):
        logits = np.zeros((1, 256)); logits[0, 98] = 3
        computed = audit.score(logits, [97])
        row = {key: str(value[0]) for key, value in computed.items()}
        row.update(target="97", old_target="98", obsolete="1")
        audit.validate_scores([row], logits)
        for field in ("forced_correct", "obsolete"):
            corrupted = dict(row); corrupted[field] = str(1 - int(row[field]))
            with self.assertRaises(audit.AuditError): audit.validate_scores([corrupted], logits)

    def test_duplicate_query_identity_is_not_silently_deduplicated(self):
        rows, logits = strict_fixture()
        rows[1] = deepcopy(rows[0]); logits[1] = logits[0]
        with self.assertRaises(audit.AuditError): audit.strict_pairs(rows, logits)

    def test_incomplete_pair_rejected(self):
        rows, logits = strict_fixture()
        with self.assertRaises(audit.AuditError): audit.strict_pairs(rows[:-1], logits[:-1])

    def test_key_only_pair_is_valid(self):
        rows, logits = strict_fixture()
        pairs = audit.strict_pairs(rows, logits)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["both_correct"] == pair["prediction_changes"] == 1 for pair in pairs.values()))

    def test_query_wording_switch_rejected(self):
        rows, logits = strict_fixture()
        rows[1]["prefix"] = rows[1]["prefix"].replace("value of", "value on")
        with self.assertRaises(audit.AuditError): audit.strict_pairs(rows, logits)

    def test_context_switch_rejected(self):
        rows, logits = strict_fixture(); rows[1]["prefix"] = "z" + rows[1]["prefix"][1:]
        with self.assertRaises(audit.AuditError): audit.strict_pairs(rows, logits)

    def test_one_seen_value_does_not_count_as_both_correct(self):
        rows, logits = strict_fixture()
        for row in rows: row["prediction"] = "97"; row["correct"] = str(int(row["target"] == "97"))
        logits.fill(0); logits[:, 97] = 3
        pairs = audit.strict_pairs(rows, logits)
        self.assertTrue(all(pair["accuracy"] == .5 and pair["both_correct"] == 0 and
                            pair["prediction_changes"] == 0 for pair in pairs.values()))

    def test_pairing_joins_by_episode_identity_after_reordering(self):
        left = {0: dict(identity="a", value=0), 1: dict(identity="b", value=1)}
        right = {1: dict(identity="b", value=1), 0: dict(identity="a", value=0)}
        a, b = audit.paired_vectors(left, right, "value")
        self.assertEqual(a, b)

    def test_pairing_rejects_same_ids_with_different_semantics(self):
        left = {0: dict(identity="a", value=1)}; right = {0: dict(identity="b", value=1)}
        with self.assertRaises(audit.AuditError): audit.paired_vectors(left, right, "value")

    def test_pairing_rejects_missing_episode(self):
        left = {0: dict(identity="a", value=1)}
        with self.assertRaises(audit.AuditError): audit.paired_vectors(left, {}, "value")

    def test_bootstrap_preserves_complete_episode_differences(self):
        result = audit.bootstrap_difference([0, 1], [1, 0])
        self.assertEqual((result["difference"], result["lower"], result["upper"]), (0., -1., 1.))
        self.assertEqual(result, audit.bootstrap_difference([0, 1], [1, 0]))

    def test_bootstrap_constant_positive_difference(self):
        result = audit.bootstrap_difference([1] * 8, [0] * 8)
        self.assertEqual((result["difference"], result["lower"], result["upper"]), (1., 1., 1.))

    def test_128_noop_checks_every_logit_not_only_argmax(self):
        rows, logits = strict_fixture()
        self.assertEqual(audit.require_noop(rows, logits, ("family", "episode", "rank"), "fixture"), 2)
        logits[-1, 233] += 1e-4
        with self.assertRaises(audit.AuditError):
            audit.require_noop(rows, logits, ("family", "episode", "rank"), "fixture")

    def test_initial_or_stage_tensor_mutation_detected(self):
        import torch
        left = {"weight": torch.tensor([1., 2.])}; right = {"weight": left["weight"].clone()}
        audit.tensor_state_equal(left, right, "fixture")
        right["weight"][0] += 1e-6
        with self.assertRaises(audit.AuditError): audit.tensor_state_equal(left, right, "fixture")

    def test_source_corruption_and_incomplete_status_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary); (root / "sources").mkdir(); source = root / "sources/example.py"
            source.write_bytes(b"value = 1\n")
            meta = dict(status="succeeded", repo_commit="abc", source_sha256={"example.py": audit.sha(source)})
            audit.source_audit(root, meta, expected_commit="abc")
            source.write_bytes(b"value = 2\n")
            with self.assertRaises(audit.AuditError): audit.source_audit(root, meta, expected_commit="abc")
            meta["status"] = "running"
            with self.assertRaises(audit.AuditError): audit.source_audit(root, meta)

    def test_wrong_commit_and_unsafe_source_path_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            meta = dict(status="succeeded", repo_commit="abc", source_sha256={"../escape.py": "0" * 64})
            with self.assertRaises(audit.AuditError): audit.source_audit(root, meta, expected_commit="def")
            with self.assertRaises(audit.AuditError): audit.source_audit(root, meta)

    def test_incomplete_main_download_fails_without_certification_or_output_directory(self):
        design = dict(runs=[dict(parent=seed, arm=arm, segment_len=segment) for seed, arm, segment in audit.RUNS],
            training=dict(steps=1600, answers_per_batch=16, unique_contexts_per_batch=8,
                          train_seed=89102, lr=.0003, optimizer="AdamW", weight_decay=.01,
                          gradient_clip=1, checkpoints=list(audit.STEPS)),
            evaluation=dict(dev_seed=89103, gate_episodes=64, strict_episodes=128),
            analysis=dict(bootstrap_draws=5000, bootstrap_seed=89104))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary); audit.jwrite(root / "design.json", design)
            output = root / "must-not-exist"
            args = SimpleNamespace(root=root, out=output, parents=root,
                                   expected_commit=audit.FROZEN_COMMIT, repo=None)
            with self.assertRaisesRegex(audit.AuditError, "Download incomplete"):
                audit.run_audit(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
