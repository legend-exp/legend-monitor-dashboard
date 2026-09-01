This repo contains the code for the legend monitoring dashboard.
The paths to the relevant data are contained in the `dashboard-config.yaml` file at the the project root.
The dashboard uses the panel framework : https://panel.holoviz.org/, and
consists of a series of classes which inherit from each other to share underlying parameters.
It runs on the spin system at NERSC inside a lightweight docker image:
https://docs.nersc.gov/services/spin/.
It should be servable to multiple people, giving them independent dashboards they can
interact with. It should be fast and responsive for users with minimal spin up time.
All code is formatted using pre-commit with `pre-commit run -a`.

## General code guidelines

Prefer short, targeted changes. Inline comments should fit on the line next to
the code they refer to; if code needs a long comment, make the code clearer instead.
Docstrings follow numpy convention:

```python
def func(a, b):
    """
    One-line summary.

    Parameters
    ----------
    a : str
        description
    b : float
        description

    Returns
    -------
    int
        description
    """
```

## Dev commands

- Venv: uv-managed `.venv` (Python 3.11), package installed editable.
  Recreate: `uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e ".[test]"`
- Tests: `.venv/bin/python -m pytest -q` (single test: `pytest tests/<dir>/<file>.py::<test> -q`)
- pytest is strict: warnings are errors, `--strict-markers --strict-config`. A new
  FutureWarning from pandas/numpy fails the suite. Use `1h` not `1H` offset aliases.

## Github guidelines

Always run `pre-commit run -a` before committing. Keep commit messages, PR bodies,
and history short and clean; avoid too many commits. Push to the `ggmarshall` fork
first; upstream is `legend-exp/legend-data-monitor`. Substantial AI contributions
must be disclosed in the PR (see `AI_POLICY.md`).
