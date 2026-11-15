# CLMM project: modified Titans

Our class project (Arjun, Bobby, Zeyuan) builds a paper-faithful reproduction of Titans ([arXiv 2501.00663](https://arxiv.org/abs/2501.00663)) and then studies a modification of it.

We build on lucidrains' [titans-pytorch](https://github.com/lucidrains/titans-pytorch) memory engine at v0.6.0 (tag `baseline-v0.6.0`). `results/` records why we chose it over the alternative we evaluated.

## Status

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
