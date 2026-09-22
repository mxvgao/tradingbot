# Legacy stochastic execution experiment

This directory preserves the original stochcontrol package, scripts, AAPL fixture
and generated outputs. It is not imported or installed by the ETF pairs project.
The duplicate run_single_path implementation was removed; the previously active
implementation (including its gamma price-pressure behavior) was preserved.

Use the root locked environment if revisiting this experiment:

```sh
uv sync --frozen
uv pip install --no-deps -e experiments/stochcontrol
cd experiments/stochcontrol
../../.venv/bin/python scripts/verify_setup.py
```

A subsequent root `uv sync --frozen` removes this optional installation.
Dependencies for normal research remain in the root pyproject.toml and uv.lock.
