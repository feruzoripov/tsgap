# TSGap

[![Tests](https://github.com/feruzoripov/tsgap/actions/workflows/python-package.yml/badge.svg)](https://github.com/feruzoripov/tsgap/actions/workflows/python-package.yml)
[![Coverage](https://codecov.io/gh/feruzoripov/tsgap/branch/main/graph/badge.svg)](https://codecov.io/gh/feruzoripov/tsgap)
[![Docs](https://github.com/feruzoripov/tsgap/actions/workflows/docs.yml/badge.svg)](https://feruzoripov.github.io/tsgap/)
[![PyPI](https://img.shields.io/pypi/v/tsgap)](https://pypi.org/project/tsgap/)
[![Python versions](https://img.shields.io/pypi/pyversions/tsgap)](https://pypi.org/project/tsgap/)
[![DOI](https://zenodo.org/badge/1163640402.svg)](https://doi.org/10.5281/zenodo.21365453)

A Python library for simulating realistic missingness in time-series data for imputation benchmarking.

TSGap separates **mechanisms** (why data is missing: MCAR, MAR, MNAR) from
**patterns** (how data is missing: pointwise, block, monotone, decay, markov,
Gilbert-Elliott).
Most patterns can be combined with any mechanism, making it easy to create
controlled missing-data scenarios for evaluating imputation methods.
`gilbert_elliott` is MCAR-only because it models an independent burst-loss
channel rather than value- or driver-dependent missingness.

## Statement of Need

Time-series imputation studies need reproducible missing-data conditions, but
lightweight tools do not always separate the statistical reason values are
missing from their temporal arrangement or expose how requested rates are
measured. TSGap provides a NumPy-native interface for constructing controlled,
seeded masks while preserving target dimensions and pre-existing gaps.

The package is designed for researchers and practitioners evaluating
time-series imputation methods. It is especially useful in healthcare
monitoring, wearable sensing, environmental sensing, longitudinal studies, and
machine-learning workflows where observed data are intentionally masked to
benchmark reconstruction quality under explicit missing-data assumptions.

## How TSGap Differs From Similar Tools

Related tools include PyGrinder and BenchPOTS from the PyPOTS ecosystem.
PyGrinder provides several Python-native missingness generators, and BenchPOTS
supports benchmarking workflows for partially observed time-series datasets.
TSGap focuses specifically on lightweight missingness simulation for arbitrary
NumPy arrays: it exposes missingness mechanisms and temporal patterns as
composable axes in one API, supports scale-aware block lengths, preserves
pre-existing NaNs and target dimensions, and documents how each
mechanism-pattern combination controls the final mask.

![Complete data (left) vs. five mechanism+pattern combinations at 20% missing rate.](https://raw.githubusercontent.com/feruzoripov/tsgap/main/assets/before_after.png)

## Features

- MCAR, MAR, and MNAR missingness mechanisms
- Pointwise, block, monotone, temporal decay, Markov-chain, and Gilbert-Elliott patterns
- 2D `(time, features)` and 3D `(samples, time, features)` arrays
- Exact or calibrated missing-rate control
- Weighted multi-driver MAR
- Reproducible masks with NumPy random seeds
- Existing NaNs and target dimensions preserved across patterns

## Installation

```bash
pip install tsgap
```

For development:

```bash
git clone https://github.com/feruzoripov/tsgap.git
cd tsgap
pip install -e ".[dev]"
```

Requires Python 3.11 or newer and NumPy 1.19 or newer.

## Quick Start

```python
import numpy as np
from tsgap import simulate_missingness

X = np.random.default_rng(42).standard_normal((1000, 6))

# MCAR: 15% scattered missing values
X_miss, mask = simulate_missingness(X, "mcar", 0.15, seed=42)

# MAR: dimension 0 drives missingness in non-driver dimensions, then block
# reshapes the selected missingness budget
X_miss, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    pattern="block", driver_dims=[0], block_len=10
)

# Scale block length to the sequence length for long series
# The default block pattern uses block_len=10 and block_density=1.0.
# block_frac is recommended
# for long wearable-style series where fixed 10-sample blocks are too short.
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="block", block_frac=0.01
)

# Sample each block length between 1% and 5% of the time axis
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="block", block_frac=(0.01, 0.05)
)

# MNAR: extreme values missing with monotone dropout
X_miss, mask = simulate_missingness(
    X, "mnar", 0.20, seed=42,
    pattern="monotone", mnar_mode="extreme"
)

# Gilbert-Elliott: bursty packet-loss with leaky good/bad periods.
# Burst length is controlled by `persist` (not block_frac): higher persist
# means longer bursts. `bad_loss` < 1 lets some values through during a bad
# period, and `good_loss` > 0 drops the occasional value during a good period.
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="gilbert_elliott", persist=0.9,
    bad_loss=0.8, good_loss=0.02
)

eligible = ~np.isnan(X)
artificial_rate = ((~mask) & eligible).sum() / eligible.sum()
print(f"Artificial missing rate: {artificial_rate:.4f}")
```

The returned `mask` uses:

```text
mask == True  -> observed
mask == False -> missing
```

## Scientific Interpretation and Rate Semantics

`missing_rate` is measured over eligible entries: originally observed values
in the requested target dimensions. Pre-existing NaNs and non-target dimensions
are excluded from its denominator. The achieved artificial rate is:

```python
eligible = ~np.isnan(X)  # additionally restrict this mask for target dimensions
achieved_rate = ((~mask) & eligible).sum() / eligible.sum()
```

Pointwise MAR and MNAR directly assign value- or driver-dependent
probabilities to individual entries. With `block`, `monotone`, `decay`, or
`markov`, the mechanism supplies a missingness budget or influence signal that
the temporal pattern rearranges. These outputs should be described as
temporally structured, mechanism-informed masks, not automatically as strict
final-mask MAR or MNAR processes.

Rate guarantees depend on the mechanism and pattern. MCAR pointwise differs
from the requested rate by at most `0.5 / number_of_eligible_entries` because
its count is rounded. Block, monotone, and decay preserve the mechanism count
when feasible. MAR, MNAR, Markov, and Gilbert-Elliott are stochastic and target
the rate in expectation, so no deterministic per-run tolerance is guaranteed;
report the achieved rate and use multiple seeds when comparing methods.
Infeasible Markov and Gilbert-Elliott configurations raise `ValueError`.

## Documentation

The documentation is built with Sphinx from the Markdown files in `docs/`.
After the documentation workflow runs on GitHub, the online documentation is
available at:

- [https://feruzoripov.github.io/tsgap/](https://feruzoripov.github.io/tsgap/)

- [Installation](https://feruzoripov.github.io/tsgap/installation.html)
- [Core concepts](https://feruzoripov.github.io/tsgap/concepts.html)
- [Mathematical details](https://feruzoripov.github.io/tsgap/mathematical_details.html)
- [Mechanisms](https://feruzoripov.github.io/tsgap/mechanisms.html)
- [Patterns](https://feruzoripov.github.io/tsgap/patterns.html)
- [API reference](https://feruzoripov.github.io/tsgap/api.html)
- [Benchmarking workflow](https://feruzoripov.github.io/tsgap/benchmarking.html)

To build the documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Example Use Cases

- Random telemetry loss in sensor networks
- Activity-dependent dropout in wearable devices
- Participant dropout in longitudinal or clinical studies
- Gradual signal loss from sensor degradation
- Intermittent connectivity failures with bursty missingness

## Evaluation Workflow

```python
import numpy as np
from tsgap import simulate_missingness

X = np.random.default_rng(42).standard_normal((1000, 6))
X_miss, mask = simulate_missingness(X, "mar", 0.20, seed=42, driver_dims=[0])

X_imputed = your_imputation_method(X_miss)
evaluation_idx = ~mask & ~np.isnan(X)

rmse = np.sqrt(np.mean((X[evaluation_idx] - X_imputed[evaluation_idx]) ** 2))
mae = np.mean(np.abs(X[evaluation_idx] - X_imputed[evaluation_idx]))
```

Use `~mask` when the original data are complete. If the original data already
contain NaNs, `~mask & ~np.isnan(X)` evaluates only values that TSGap
artificially hid.

See the [benchmarking workflow](https://feruzoripov.github.io/tsgap/benchmarking.html)
for a complete baseline workflow.

The repository also includes a runnable benchmark example:

```bash
python examples/benchmark_imputation.py
```

## Testing

```bash
pytest tsgap/tests/ -v
```

GitHub Actions runs the test suite on Python 3.11 through 3.14 and builds the
Sphinx documentation on every pull request to `main`.

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/feruzoripov/tsgap/blob/main/CONTRIBUTING.md)
for development setup, testing, and pull request guidelines.

## Citation

```bibtex
@software{tsgap,
  author = {Oripov, Feruz and Korchagina, Kseniia and Bonsu, Enock Adu and Bilgin, Ali and Aras, Shravan},
  title = {TSGap: A Python Library for Composable Time-Series Missingness Simulation},
  year = {2026},
  version = {0.8.0},
  doi = {10.5281/zenodo.21365453},
  url = {https://github.com/feruzoripov/tsgap}
}
```

## License

MIT

## AI Usage Disclosure

Generative AI tools, including OpenAI Codex/ChatGPT and Anthropic Claude, have
been used to assist with code review, test generation, documentation
organization, and manuscript drafting during development. All AI-assisted code
and text are reviewed, tested, and validated by the package authors, who retain
full responsibility for the project.
