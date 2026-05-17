# Mechanisms

Mechanisms describe the relationship between the data and the probability that
a value is missing.

## MCAR

`mcar` means Missing Completely At Random. Every eligible position has the same
chance of being masked, independent of values in the array.

```python
X_missing, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
```

MCAR samples without replacement, so the achieved missing count is exact up to
rounding.

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target` | `"all"` | Dimensions to mask, or `"all"` |

## MAR

`mar` means Missing At Random. Missingness depends on observed driver
dimensions, not on the value being masked.

```python
X_missing, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    driver_dims=[0], strength=2.0
)
```

With multiple driver dimensions, `driver_weights` controls their relative
contribution. Weights are normalized automatically.

```python
X_missing, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    driver_dims=[0, 1],
    driver_weights=[0.8, 0.2],
    strength=2.0
)
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `driver_dims` | `[0]` | Dimensions that drive missingness |
| `driver_weights` | `None` | Optional non-negative weights for drivers |
| `target` | `"all"` | Dimensions to mask, or `"all"` |
| `strength` | `2.0` | Dependency strength, must be non-negative |
| `base_rate` | `0.01` | Minimum probability floor |
| `direction` | `"positive"` | `"positive"` or `"negative"` driver relationship |

## MNAR

`mnar` means Missing Not At Random. Missingness depends on the value itself.

```python
X_missing, mask = simulate_missingness(
    X, "mnar", 0.20, seed=42,
    mnar_mode="extreme", strength=3.0
)
```

Modes:

| Mode | Effect |
|------|--------|
| `"high"` | High values are more likely to be missing |
| `"low"` | Low values are more likely to be missing |
| `"extreme"` | Values far from the mean are more likely to be missing |

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mnar_mode` | `"extreme"` | `"high"`, `"low"`, or `"extreme"` |
| `target` | `"all"` | Dimensions to mask, or `"all"` |
| `strength` | `2.0` | Dependency strength, must be non-negative |

## Rate Calibration

MAR and MNAR use a logistic probability model and calibrate an offset by binary
search so the expected missing rate over eligible positions matches the target
rate. Because they sample Bernoulli outcomes, achieved rates are approximate and
vary more on small arrays.
