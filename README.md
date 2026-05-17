# TSGap

A Python library for simulating realistic missingness in time-series data for imputation benchmarking.

TSGap separates **mechanisms** (why data is missing: MCAR, MAR, MNAR) from
**patterns** (how data is missing: pointwise, block, monotone, decay, markov).
Any mechanism can be combined with any pattern, making it easy to create
controlled missing-data scenarios for evaluating imputation methods.

![Complete data (left) vs. five mechanism+pattern combinations at 20% missing rate.](assets/before_after.png)

## Features

- MCAR, MAR, and MNAR missingness mechanisms
- Pointwise, block, monotone, temporal decay, and Markov-chain patterns
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

Requires Python 3.9 or newer and NumPy 1.19 or newer.

## Quick Start

```python
import numpy as np
from tsgap import simulate_missingness

X = np.random.default_rng(42).standard_normal((1000, 6))

# MCAR: 15% scattered missing values
X_miss, mask = simulate_missingness(X, "mcar", 0.15, seed=42)

# MAR: missingness driven by dimension 0 and arranged in blocks
X_miss, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    pattern="block", driver_dims=[0], block_len=10
)

# MNAR: extreme values missing with monotone dropout
X_miss, mask = simulate_missingness(
    X, "mnar", 0.20, seed=42,
    pattern="monotone", mnar_mode="extreme"
)

print(f"Actual missing rate: {(~mask).mean():.4f}")
```

The returned `mask` uses:

```text
mask == True  -> observed
mask == False -> missing
```

## Documentation

- [Installation](docs/installation.md)
- [Core concepts](docs/concepts.md)
- [Mechanisms](docs/mechanisms.md)
- [Patterns](docs/patterns.md)
- [API reference](docs/api.md)
- [Benchmarking workflow](docs/benchmarking.md)

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
missing_idx = ~mask

rmse = np.sqrt(np.mean((X[missing_idx] - X_imputed[missing_idx]) ** 2))
mae = np.mean(np.abs(X[missing_idx] - X_imputed[missing_idx]))
```

See [docs/benchmarking.md](docs/benchmarking.md) for a complete baseline workflow.

The repository also includes a runnable benchmark example:

```bash
python examples/benchmark_imputation.py
```

## Testing

```bash
pytest tsgap/tests/ -v
```

## Citation

```bibtex
@software{tsgap,
  author = {Oripov, Feruz and Korchagina, Kseniia and Bonsu, Enock Adu and Bilgin, Ali and Aras, Shravan},
  title = {TSGap: A Python Library for Composable Time-Series Missingness Simulation},
  year = {2026},
  url = {https://github.com/feruzoripov/tsgap}
}
```

## License

MIT
