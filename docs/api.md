# API Reference

## `simulate_missingness`

```python
simulate_missingness(
    X,
    mechanism,
    missing_rate,
    seed=None,
    pattern="pointwise",
    **kwargs,
)
```

Simulate missingness in a 2D `(time, features)` or 3D
`(samples, time, features)` NumPy array.

Returns:

```python
X_missing, mask
```

where `X_missing` is a copy of `X` with NaNs inserted and `mask` is a boolean
array with `True` for observed values and `False` for missing values.

Core parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Input data, shape `(T, D)` or `(N, T, D)` |
| `mechanism` | `str` | `"mcar"`, `"mar"`, or `"mnar"` |
| `missing_rate` | `float` | Target fraction missing, clipped to `[0, 1]` |
| `seed` | `int \| None` | Random seed |
| `pattern` | `str` | Missingness pattern |

Mechanism parameters:

| Parameter | Mechanisms | Default | Description |
|-----------|------------|---------|-------------|
| `target` | all | `"all"` | Dimensions to mask |
| `driver_dims` | MAR | `[0]` | Driver dimensions |
| `driver_weights` | MAR | `None` | Per-driver weights |
| `strength` | MAR, MNAR | `2.0` | Dependency strength |
| `base_rate` | MAR | `0.01` | Minimum probability floor |
| `direction` | MAR | `"positive"` | Driver direction |
| `mnar_mode` | MNAR | `"extreme"` | MNAR scoring mode |

Pattern parameters:

| Parameter | Patterns | Default | Description |
|-----------|----------|---------|-------------|
| `block_len` | block | `10` | Block length in timesteps |
| `block_density` | block | `0.7` | Fraction of missingness in blocks |
| `decay_rate` | decay | `3.0` | Decay ramp steepness |
| `decay_center` | decay | `0.7` | Normalized ramp center |
| `persist` | markov | `0.8` | Missing-state persistence |

## `simulate_many_rates`

```python
from tsgap import simulate_many_rates

rates = [0.05, 0.15, 0.25]
results = simulate_many_rates(X, "mcar", rates, seed=42)
```

Returns a dictionary mapping each rate to `(X_missing, mask)`. When a seed is
provided, each rate gets a deterministic offset from the base seed.

## `MissingnessSimulator`

```python
from tsgap import MissingnessSimulator

sim = MissingnessSimulator(
    "mar", missing_rate=0.25, seed=42,
    driver_dims=[0], pattern="block", block_len=10
)

X_missing, mask = sim.generate(X)
```

This object-oriented wrapper is useful when the same missingness configuration
is applied to multiple arrays.

## Registries

```python
from tsgap import MECHANISMS, PATTERNS

print(MECHANISMS.keys())
print(PATTERNS.keys())
```

The registries expose the currently available mechanism and pattern names,
including aliases.
