"""CPU-only synthetic checks; no experiment model or checkpoint is loaded."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import unittest
from types import SimpleNamespace

import disjoint_probe as probe


REPO = Path(r"C:\Users\henry\Documents\Codex\clmm-replay")


class DisjointSelectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = probe.bootstrap(REPO)
        cls.runtime.torch.set_num_threads(2)
        cls.bank = cls.runtime.association.build_bank()
        cls.candidates = cls.runtime.association.fresh_same(cls.bank, probe.SELECTION_SEED,
                                                          count=probe.CANDIDATES)
        cls.blocked = probe.context_bytes(cls.candidates[0])
        cls.selected, cls.counters = probe.select_episodes(cls.bank, {cls.blocked}, cls.runtime)

    def test_known_training_context_is_rejected_and_four_per_pair_are_deterministic(self):
        again, counters = probe.select_episodes(self.bank, {self.blocked}, self.runtime)
        self.assertEqual(again, self.selected)
        self.assertEqual(counters, self.counters)
        self.assertGreaterEqual(counters["rejected_training"], 1)
        self.assertEqual(Counter(record["base_pair"] for record in self.selected),
                         Counter({index: 4 for index in range(32)}))
        contexts = set()
        for record in self.selected:
            episode = self.runtime.facts.Episode.from_dict(record["episode"])
            self.assertEqual(episode, self.candidates[record["source_candidate_index"]])
            self.assertEqual(record["base_pair"], record["source_candidate_index"] % 32)
            context = probe.context_bytes(episode)
            self.assertNotEqual(context, self.blocked)
            contexts.add(context)
        self.assertEqual(len(contexts), 128)

    def test_all_accepted_prefixes_use_valid_namespace_and_exclude_answer_suffix(self):
        samples = probe.make_samples(self.selected, self.runtime)
        self.assertEqual(len(samples), 256)
        train = set(self.runtime.facts.KEY_POOLS["train"])
        other = set(self.runtime.facts.KEY_POOLS["dev"]) | set(self.runtime.facts.KEY_POOLS["test"])
        for index, record in enumerate(self.selected):
            episode = self.runtime.facts.Episode.from_dict(record["episode"])
            original = self.runtime.facts.Episode.from_dict(self.bank["episodes"][record["base_pair"]])
            old_values = {event.value for event in original.events}
            self.assertFalse(old_values & {event.value for event in episode.events})
            self.assertTrue({event.key for event in episode.events} <= train)
            self.assertFalse({event.key for event in episode.events} & other)
            for rank, query in enumerate(episode.queries):
                sample = samples[index * 2 + rank]
                prefix = sample["prefix"].encode("ascii")
                self.assertEqual(len(prefix), 95)
                self.assertEqual(prefix[:64], probe.context_bytes(episode))
                self.assertEqual(prefix[64:], self.runtime.facts.render_query(query))
                self.assertEqual(sample["target"], query.answer_byte)
                for value in self.runtime.facts.VALUES:
                    self.assertEqual(self.runtime.facts.render_query(replace(query, target=value)),
                                     prefix[64:])

    def test_candidate_cap_fails_closed(self):
        excluded = {probe.context_bytes(episode) for episode in self.candidates}
        with self.assertRaisesRegex(RuntimeError, "Candidate cap exhausted"):
            probe.select_episodes(self.bank, excluded, self.runtime)

    def test_training_exclusion_uses_the_full64_fact_context(self):
        np = self.runtime.np
        episode = self.candidates[0]
        prefixes = [self.runtime.facts.encode_example(episode, query, "online")[0]
                    for query in episode.queries]
        array = np.tile(np.stack([np.frombuffer(prefix, dtype=np.uint8) for prefix in prefixes]),
                        (2000, 8, 1))
        self.assertEqual(array.shape, (2000, 16, 95))
        self.assertEqual(probe.training_contexts(array, np), {probe.context_bytes(episode)})
        array[0, 1, 63] ^= 1
        with self.assertRaisesRegex(ValueError, "share their fact context"):
            probe.training_contexts(array, np)

    def test_scoring_alignment_and_frozen_tensors_with_a_synthetic_model(self):
        torch, np = self.runtime.torch, self.runtime.np

        class Synthetic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.linspace(-1, 1, 256))

            def forward(self, x):
                return self.bias.view(1, 1, 256).expand(x.shape[0], x.shape[1], 256)

        artifact = dict(checkpoint_path=str(Path(__file__).resolve()),
                        checkpoint_sha256=probe.sha256(__file__),
                        parent_checkpoint_sha256="synthetic-parent", source_sha256={})
        samples = probe.make_samples(self.selected, self.runtime)
        rows, logits = [], []
        for run in probe.RUNS:
            artifact.update(segment_len=32 if "32-" in run else 128,
                            regime="fresh" if run.startswith("fresh") else "fixed")
            payload = dict(source_info=dict(source_sha256={}, repo_commit=probe.FROZEN_COMMIT),
                training_info=dict(parent_init_seed=68101, steps=2000, regime=artifact["regime"],
                    task="fixed-set-generalization", fixed_final_checkpoint=True,
                    parent_checkpoint_sha256="synthetic-parent"))
            config = SimpleNamespace(segment_len=artifact["segment_len"], memory_segment_len=8,
                                     memory_batch_size=32)
            runtime = SimpleNamespace(**vars(self.runtime))
            runtime.pretrain = SimpleNamespace(
                load_checkpoint=lambda *args, **kwargs: (Synthetic(), config, payload),
                reset_memory_at_segments=lambda model: nullcontext())
            checks = probe.evaluate_run(run, artifact, samples, runtime, torch.device("cpu"),
                                        None, rows, logits)
            self.assertTrue(all(checks.values()))
        matrix = np.concatenate(logits)
        self.assertEqual(matrix.shape, (2048, 256))
        self.assertEqual(len(rows), len(matrix))
        self.assertTrue(all(row["prediction"] == 255 for row in rows))
        self.assertTrue(all(row["forced_prediction"] == ord("p") for row in rows))
        scores = self.runtime.common.score_logits(torch.tensor(matrix),
                                                  torch.tensor([row["target"] for row in rows]))
        for field, values in scores.items():
            np.testing.assert_allclose([row[field] for row in rows], values.numpy(), rtol=1e-6)
        summary = probe.summarize(rows)
        self.assertEqual(len(summary), 8)
        self.assertTrue(all(row["both_correct"] == 0 for row in summary))
        altered = [dict(row) for row in rows]
        for row in altered:
            row["correct"] = 1
        altered[1]["correct"] = 0
        changed = probe.summarize(altered)[0]
        self.assertEqual(changed["accuracy"], 255 / 256)
        self.assertEqual(changed["both_correct"], 127 / 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
