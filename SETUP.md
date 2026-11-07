# CLMM project setup

Baseline: lucidrains/titans-pytorch v0.6.0, commit 35a549f (tag `baseline-v0.6.0`).

## One-time setup

1. Accept the GitHub org invite and make sure your SSH key is on your GitHub account (`ssh -T git@github.com` should greet you by name).
2. Install Python 3.13 if you don't have it (python.org). `requirements-lock.txt` was made with 3.13. Check with `python3 --version`.
3. Clone and create the environment:

```bash
git clone git@github.com:CLMM-group-Arjun-Bobby-Zeyuan/clmm-project.git
cd clmm-project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e ".[test]"
```

If you use conda, run `conda deactivate` first so the venv isn't built on conda's Python.

NVIDIA GPU users: first install PyTorch with the command from pytorch.org for your CUDA version, then run the two `pip install` lines.

## Check your install

```bash
pytest tests/ -q
python scripts/verify_titans.py --repo lucidrains --path .
```

All tests should pass. The 5 CUDA-only tests skip on machines without an NVIDIA GPU. The verify script should report PASS for every causality check.

## Baseline verification

`results/verify_lucidrains.txt` and `results/verify_neuranox.txt` record why this project builds on lucidrains: lucidrains passes every causality check, and every Neuranox variant fails them.

## Every work session

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
git switch main && git pull
git switch -c <task-branch-name>
```

Work on task branches, push them, and merge into `main` through a pull request reviewed by someone else.
Don't move or delete the `baseline-v0.6.0` tag.
