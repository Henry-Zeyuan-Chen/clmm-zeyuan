# CLMM project: modified Titans

Our class project (Arjun, Bobby, Zeyuan) builds a paper-faithful reproduction of Titans ([arXiv 2501.00663](https://arxiv.org/abs/2501.00663)) and then studies a modification of it.

We build on lucidrains' [titans-pytorch](https://github.com/lucidrains/titans-pytorch) memory engine at v0.6.0 (tag `baseline-v0.6.0`). `results/` records why we chose it over the alternative we evaluated.

## Status

The [K16 multiple-revision replay comparison](docs/SLEEP_MULTIREVISION.md) evaluates all twelve shared local-delta checkpoints with exact matched continuous/periodic samples and write budgets. At D128, revised accuracy is 84.22% continuous, 81.76% periodic, versus 93.89% repeat-current/online. All checkpoints pass neutral acquisition, but both replay schedules damage replacement at D0. This is a controlled revision/interference result and motivates testing replay freshness rather than claiming a sleep benefit.

The latest [key-local revision study](docs/REVISION_LOCAL.md) repairs existing-checkpoint revisions and supplies 24 matched training runs with initial controls. Trained local delta reaches 95.5–97.1% late stable accuracy versus EMA's 24.0–28.5%; both solve short revisions at 100%. These qualified component baselines retain fixed random routing controls and distinguish the ceiling-limited revision test from the stable gain.

The latest [routing/decay factorial](docs/ROUTING_DECAY.md) freezes four dense-trained parents across48 conditions. Inactive-trace protection strongly changes late stable scores, while cosine allocation has no advantage over matched fixed random maps. Hebbian revision failures expose a stability/revision tradeoff; the gradient replication passes admission.

The local [reference adapter integration](docs/ADAPTER_INTEGRATION.md) connects bounded sleep sessions and Hebbian/gradient associative components to the shared correlated-key baseline, with persistent checkpoints, a common evaluator and a finite four-system pilot. Selective access helps in the component study, while the Hebbian rule does not beat its matched gradient control. These are reference modules; full Titans/sequence-model and teammate-final method integration remain separate.

We are working through [docs/ROADMAP.md](docs/ROADMAP.md). Until the `paper-baseline-v1` tag exists, the paper-style models are in progress.

Local exploratory evidence on this branch: [memory-pressure experiments](docs/MEMORY_PRESSURE.md) extend the [replay pilot](docs/REPLAY_PILOT.md) with matched update budgets, three training seeds, and longer retention delays. These experiments do not mark the paper-style baseline complete.

Following the group sync, [shared pretraining](docs/SHARED_PRETRAIN.md) adds a compact sequence model, common checkpoint format, disjoint data splits, and memory-context diagnostics under `clmm/pretrain`.

Member C's [fact/revision evaluation](docs/FACT_REVISION.md) adapts the shared checkpoints and compares causal continuous/periodic replay schedules with matched record budgets, alongside acquisition and revision diagnostics. Both main adaptations fail the acquisition gate; the saved results motivate fixing key-specific recall before drawing retention conclusions.

The follow-up [paired-key acquisition diagnostic](docs/PAIRED_ACQUISITION.md) tests balanced supervision against a matched duplicate-query control and saves fixed curriculum-stage checkpoints for both shared parents. Single-fact recall and revisions succeed, while two-key selection remains unresolved even with the wider attention control.

[Fixed-set fitting and transfer](docs/FIXED_SET_GENERALIZATION.md) establishes perfect fitting of 32 contexts but failed transfer to new values, including a bounded fresh-assignment follow-up with a training-disjoint probe. [Member C status](docs/MEMBER_C_STATUS.md) connects these controls, checkpoint qualifications and remaining baseline work to the sleep and Hebbian project tracks.

## Getting started

Follow [SETUP.md](SETUP.md). After pulling a change to `pyproject.toml`, rerun `pip install -e ".[test]"`.

## Repository layout

```text
clmm/              our code: reference implementation, paper-style models, baselines
titans_pytorch/    lucidrains' engine (NeuralMemory); we change it only behind flags
tests/             test_titans.py is upstream's; test_clmm_*.py are ours
scripts/           verification and utility scripts
results/           saved outputs, each with its config and commit hash
docs/              roadmap, ambiguity register, deviations from the paper
data/              enwik8 sample from upstream, for quick smoke tests
upstream/          lucidrains' original README, figures and example training scripts
```

Upstream's example scripts read `./data/enwik8.gz`, so run them from the repo root, e.g. `python upstream/train_mac.py`.

## Documents

- [docs/ROADMAP.md](docs/ROADMAP.md): what we still need to do before starting our modification
- [docs/ambiguity-register.md](docs/ambiguity-register.md): every choice the paper leaves open, and what we decided
- `docs/DEVIATIONS.md` (to come): every place we knowingly differ from the paper

## Attribution and license

`titans_pytorch/`, `tests/test_titans.py`, `data/` and `upstream/` come from lucidrains/titans-pytorch, copyright (c) 2025 Phil Wang, under the MIT license in `LICENSE`.

If you use this work, cite the Titans paper:

```bibtex
@article{behrouz2024titans,
  title   = {Titans: Learning to Memorize at Test Time},
  author  = {Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2501.00663},
  year    = {2024}
}
```
