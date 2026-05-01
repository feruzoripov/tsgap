# Core Concepts

TSGap separates missing-data simulation into two composable parts:

- **Mechanisms** describe why data is missing.
- **Patterns** describe how missingness is arranged in time.

This separation lets users combine a statistical assumption with a temporal
structure. For example, MAR missingness can be simulated as scattered points,
contiguous blocks, monotone dropout, late-series decay, or bursty Markov
segments.

## Mechanisms

| Mechanism | Meaning | Rate control |
|-----------|---------|--------------|
| `mcar` | Missing Completely At Random. Missingness is independent of data values. | Exact |
| `mar` | Missing At Random. Missingness depends on observed driver variables. | Calibrated |
| `mnar` | Missing Not At Random. Missingness depends on the value itself. | Calibrated |

## Patterns

| Pattern | Aliases | Description |
|---------|---------|-------------|
| `pointwise` | `point`, `scattered` | Individual scattered points |
| `block` | `contiguous` | Contiguous sensor-dropout segments |
| `monotone` | `dropout` | Once missing, a series remains missing |
| `decay` | `degradation` | Missingness increases over time |
| `markov` | `flickering` | Bursty temporal dependence |

## Mask Convention

TSGap returns `(X_missing, mask)`.

```text
mask == True  -> observed
mask == False -> missing
```

This convention supports evaluation on the artificially masked entries:

```python
missing_idx = ~mask
rmse = np.sqrt(np.mean((X[missing_idx] - X_imputed[missing_idx]) ** 2))
```

## Data Shapes

TSGap supports 2D and 3D arrays:

```python
# 2D: one series, shape (time, features)
X_2d = np.random.default_rng(42).standard_normal((500, 6))

# 3D: multiple samples, shape (samples, time, features)
X_3d = np.random.default_rng(42).standard_normal((50, 500, 6))
```

For 3D data, MAR normalizes driver signals per sample and MNAR normalizes each
sample-feature series independently.

## Existing NaNs And Targets

Pre-existing NaNs are preserved and excluded from the eligible pool. If
`target` dimensions are provided, missingness is injected only into those
dimensions.

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.2, seed=42, target=[1, 3], pattern="block"
)
```
