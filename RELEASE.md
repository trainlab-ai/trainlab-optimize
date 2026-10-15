# Releasing trainlab-optimize

## 1. Check

```bash
pip install -e ".[dev]" && pip install openai          # openai: the sample app's tests
python -m pytest -q                                     # also on the oldest supported Python, 3.10
python bench/experiments/run.py WORK && python bench/experiments/report.py   # all X checks pass
python bench/run.py --lock --fresh && python bench/freeze.py check           # scanner changes only; ~25 min
```

## 2. Version

Set the same version in `pyproject.toml` and `src/trainlab_optimize/__init__.py`, update the
roadmap line in `README.md`, commit, and tag `vX.Y.Z`.

## 3. Build

```bash
rm -rf dist build src/*.egg-info
python -m build                  # dist/trainlab_optimize-X.Y.Z-py3-none-any.whl and .tar.gz
twine check dist/*
```

Try the wheel the way users will get it, from a clean shell:

```bash
TRAINLAB_PACKAGE="$PWD/dist/trainlab_optimize-X.Y.Z-py3-none-any.whl" sh scripts/install.sh
```

```powershell
$env:TRAINLAB_PACKAGE = "$PWD\dist\trainlab_optimize-X.Y.Z-py3-none-any.whl"; .\scripts\install.ps1
```

Then in `examples/support_desk`: `trainlab trace --capture -- python run.py --mock`,
`trainlab experiment run --max-cost 1` (with `python mock_openai.py --port 8787` running and
`OPENAI_BASE_URL=http://127.0.0.1:8787/v1 OPENAI_API_KEY=mock`), and `trainlab ui`.

## 4. Publish

First time only: create the project on PyPI (an account with 2FA, then an API token scoped to
`trainlab-optimize`, or a trusted publisher for the GitHub repository).

```bash
twine upload -r testpypi dist/*      # optional dry run on test.pypi.org
twine upload dist/*
```

## 5. Install scripts

Serve `scripts/install.sh` and `scripts/install.ps1` as plain text at
`https://trainlab.ai/install.sh` and `https://trainlab.ai/install.ps1`. They install the latest
version from PyPI, so they only change when the install steps change.

## 6. Announce

GitHub release from the tag, with the commit log since the last tag as notes.
