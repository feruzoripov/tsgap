# Benchmarking Workflow

TSGap is designed for controlled imputation benchmarking: start with complete
data, inject known missingness, run an imputation method, then evaluate only on
the artificially masked values.

## Basic Workflow

```python
import numpy as np
from tsgap import simulate_missingness

rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 6))

X_missing, mask = simulate_missingness(
    X, "mar", 0.20, seed=42,
    pattern="block", driver_dims=[0], block_len=12
)

X_imputed = X_missing.copy()
for d in range(X.shape[1]):
    fill = np.nanmean(X_imputed[:, d])
    X_imputed[np.isnan(X_imputed[:, d]), d] = fill

missing_idx = ~mask
rmse = np.sqrt(np.mean((X[missing_idx] - X_imputed[missing_idx]) ** 2))
mae = np.mean(np.abs(X[missing_idx] - X_imputed[missing_idx]))

print({"rmse": rmse, "mae": mae})
```

## Comparing Conditions

```python
conditions = [
    ("mcar", "pointwise", {}),
    ("mcar", "block", {"block_len": 10}),
    ("mar", "block", {"driver_dims": [0], "block_len": 10}),
    ("mnar", "monotone", {"mnar_mode": "extreme"}),
    ("mcar", "markov", {"persist": 0.8}),
]

for mechanism, pattern, kwargs in conditions:
    X_missing, mask = simulate_missingness(
        X, mechanism, 0.20, seed=42, pattern=pattern, **kwargs
    )
    print(mechanism, pattern, (~mask).mean())
```

## Reporting Results

For reproducible benchmark reports, record:

- dataset or synthetic data generation procedure
- mechanism and pattern
- missing rate
- random seed
- imputation method
- metric and evaluation subset

The most common evaluation subset is `~mask`, which contains values that TSGap
artificially hid.
