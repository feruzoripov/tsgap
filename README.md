# ts_missingness

A Python library for simulating realistic missingness in time-series data for imputation benchmarking.

Explicitly separates **mechanisms** (why data is missing: MCAR, MAR, MNAR) from **patterns** (how data is missing: pointwise, block, monotone, decay, markov). Any mechanism can be combined with any pattern. Supports 2D and 3D arrays, exact or calibrated rate control, weighted multi-driver MAR, and full reproducibility.

![Complete data (left) vs. five mechanism+pattern combinations at 20% missing rate.](assets/before_after.png)

Each panel shows a (200 timesteps × 8 features) array as a heatmap. The color scale represents data values: yellow = low values, green = mid-range values, blue = high values. Red cells are missing values overlaid on top of the data. The first panel shows the complete dataset with no missingness. The remaining five panels each apply a different mechanism+pattern combination at 20% missing rate, so you can see exactly how each configuration removes data:

- MCAR + Pointwise: red cells scattered uniformly at random across the entire array.
- MAR + Block: red cells appear in contiguous horizontal bands, concentrated in regions where the driver variable (dim 0, which ramps up over time) has high values.
- MNAR + Monotone: each feature column has a clean cutoff point — everything after the dropout time is solid red, modeling permanent sensor failure.
- MCAR + Decay: red cells are sparse at the top (early timesteps) and dense at the bottom (late timesteps), modeling gradual sensor degradation.
- MAR + Markov: bursty red streaks that flicker on and off along the time axis, with burst locations influenced by the driver variable.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9 and NumPy ≥ 1.19.

---

## Quick Start

```python
import numpy as np
from ts_missingness import simulate_missingness

X = np.random.randn(1000, 6)  # (T, D) — 1000 timesteps, 6 features

# MCAR: 15% scattered missing (default pattern)
X_miss, mask = simulate_missingness(X, "mcar", 0.15, seed=42)

# MAR: 25% missing in contiguous blocks, driven by dimension 0
X_miss, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    pattern="block", driver_dims=[0], block_len=10
)

# MNAR: 20% extreme values missing with monotone dropout
X_miss, mask = simulate_missingness(
    X, "mnar", 0.20, seed=42,
    pattern="monotone", mnar_mode="extreme"
)

# MAR: weighted multi-driver with temporal decay
X_miss, mask = simulate_missingness(
    X, "mar", 0.30, seed=42,
    pattern="decay", driver_dims=[0, 1], driver_weights=[0.8, 0.2],
    decay_rate=5.0, decay_center=0.6
)

# MCAR: intermittent flickering with Markov chain dependence
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="markov", persist=0.8
)

print(f"Actual missing rate: {(~mask).mean():.4f}")
```

---

## Core Concept: Mechanisms × Patterns

The library separates two orthogonal concerns:

**Mechanisms** answer *why* data is missing — the probabilistic relationship between missingness and data values:

| Mechanism | Description | Rate Control |
|-----------|-------------|--------------|
| `mcar` | Independent of all data | Exact |
| `mar` | Depends on other observed variables | Calibrated |
| `mnar` | Depends on the missing value itself | Calibrated |

**Patterns** answer *how* data is missing — the temporal/spatial structure:

| Pattern | Aliases | Description |
|---------|---------|-------------|
| `pointwise` | `point`, `scattered` | Individual scattered points (default) |
| `block` | `contiguous` | Contiguous segments (sensor dropout) |
| `monotone` | `dropout` | Once missing, stays missing (participant dropout) |
| `decay` | `degradation` | Missingness increases over time (sensor degradation) |
| `markov` | `flickering` | Temporally dependent bursts (intermittent sensor failure) |

Any mechanism can be combined with any pattern:

| Combination | Real-World Scenario |
|-------------|---------------------|
| MCAR + pointwise | Random transmission errors |
| MCAR + block | Random battery depletion periods |
| MAR + block | Activity-dependent sensor dropout |
| MAR + decay | Sensor degrades faster under load |
| MNAR + monotone | Extreme readings cause permanent sensor failure |
| MNAR + pointwise | Sensor saturates at extreme values |
| MCAR + markov | Random intermittent connectivity drops |
| MAR + markov | Load-dependent flickering sensor |

![3×5 grid showing all mechanism × pattern combinations at 20% missing rate. Blue = missing, white = observed.](assets/mechanism_pattern_grid.png)

The grid above shows all 15 possible mechanism × pattern combinations. Each cell is a (200 × 8) missingness mask where blue cells are missing and white cells are observed. Rows vary the mechanism (MCAR, MAR, MNAR) and columns vary the pattern (pointwise, block, monotone, decay, markov). The percentage in each cell's corner is the actual achieved missing rate. Notice how MCAR rows look uniform across features, MAR rows concentrate missingness where the driver is high, and MNAR rows target extreme data values — while the column patterns independently control the temporal shape.

---

## Mask Convention

```
mask == True  → observed
mask == False → missing
```

This allows direct evaluation:

```python
missing_idx = ~mask
rmse = np.sqrt(np.mean((X[missing_idx] - X_imputed[missing_idx]) ** 2))
```

---

## Mechanisms

### MCAR — Missing Completely At Random

Missingness is independent of all data. Uses uniform sampling without replacement for exact rate control.

$$P(M_{ij} = 1) = \rho$$

```python
X_miss, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
```

**Parameters:**
- `target`: `"all"` (default) or list of dimension indices to mask, e.g. `[0, 2]`

---

### MAR — Missing At Random

Missingness depends on observed driver variables via a logistic model. The offset is automatically calibrated via binary search to match the target rate.

$$P(M_{ij} = 1 \mid X) = \sigma\!\left(\alpha \cdot z_i + \beta\right)$$

where $z_i$ is the normalized driver signal, $\alpha$ is the strength, and $\beta$ is the calibrated offset.

When multiple drivers are specified, the driver signal is a weighted linear combination:

$$z_i = \sum_k w_k \cdot \frac{X_{i,k} - \mu_k}{\sigma_k}$$

where $w_k$ are the normalized `driver_weights`. If omitted, all drivers contribute equally (simple mean).

```python
# Single driver
X_miss, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    driver_dims=[0], strength=2.0
)

# Weighted multi-driver: 80% activity, 20% temperature
X_miss, mask = simulate_missingness(
    X, "mar", 0.25, seed=42,
    driver_dims=[0, 1], driver_weights=[0.8, 0.2], strength=2.0
)
```

**Parameters:**
- `driver_dims`: list of dimension indices that drive missingness (default: `[0]`)
- `driver_weights`: weights for each driver (auto-normalized, default: equal)
- `target`: `"all"` or list of dimension indices to mask
- `strength`: dependency strength, ≥ 0 (default: `2.0`)
- `base_rate`: minimum probability floor (default: `0.01`)
- `direction`: `"positive"` (high driver → high missing) or `"negative"`

---

### MNAR — Missing Not At Random

Missingness depends on the value itself. Three modes control which values are more likely to be missing:

$$P(M_{ij} = 1 \mid X_{ij}) = \sigma\!\left(\alpha \cdot f(z_{ij}) + \beta\right)$$

where $f(z)$ depends on the mode:

| Mode | $f(z)$ | Interpretation |
|------|--------|----------------|
| `"high"` | $z$ | High values more likely missing |
| `"low"` | $-z$ | Low values more likely missing |
| `"extreme"` | $\|z\|$ | Extreme values (both tails) more likely missing |

```python
X_miss, mask = simulate_missingness(
    X, "mnar", 0.15, seed=42,
    mnar_mode="extreme", strength=3.0
)
```

**Parameters:**
- `mnar_mode`: `"extreme"` (default), `"high"`, or `"low"`
- `target`: `"all"` or list of dimension indices to mask
- `strength`: dependency strength, ≥ 0 (default: `2.0`)

---

## Patterns

### Pointwise (default)

Individual points are missing independently. This is the default — no additional parameters needed.

```python
X_miss, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
# equivalent to: pattern="pointwise"
```

---

### Block

Converts scattered missingness into contiguous segments. Simulates sensor dropout periods where a device goes offline for multiple consecutive timesteps.

```python
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="block", block_len=10, block_density=0.7
)
```

**Parameters:**
- `block_len`: length of each missing block in timesteps (default: `10`)
- `block_density`: fraction of total missingness allocated to blocks, 0.0–1.0 (default: `0.7`)

**Use cases:** battery depletion, device removal, connectivity loss.

---

### Monotone

Once a dimension goes missing at time $t$, it stays missing for all $t' > t$. Models participant dropout in longitudinal studies and clinical trials.

The mechanism mask determines which dimensions drop out and approximately when. The pattern enforces the monotone constraint and adjusts dropout times to match the target missing count.

```python
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42, pattern="monotone"
)

# Verify: no observed values after first missing in each dimension
for d in range(X.shape[1]):
    missing_t = np.where(~mask[:, d])[0]
    if len(missing_t) > 0:
        assert not mask[missing_t[0]:, d].any()  # Monotone guaranteed
```

**Use cases:** patient dropout, study withdrawal, permanent sensor failure.

---

### Temporal Decay

Missingness probability increases over time via a sigmoid ramp, modeling sensor degradation, battery drain, or participant fatigue. Early timesteps have low missingness; later timesteps have high missingness.

$$w(t) = \sigma\!\left(\gamma \cdot (t_{\text{norm}} - c)\right)$$

where $\gamma$ is `decay_rate`, $c$ is `decay_center`, and $t_{\text{norm}} \in [0, 1]$.

```python
X_miss, mask = simulate_missingness(
    X, "mcar", 0.25, seed=42,
    pattern="decay", decay_rate=5.0, decay_center=0.6
)
```

**Parameters:**
- `decay_rate`: steepness of the temporal ramp (default: `3.0`). Higher = sharper transition.
- `decay_center`: normalized time position (0–1) where missingness reaches 50% (default: `0.7`). Lower values shift missingness earlier.

**Use cases:** sensor degradation, battery drain, participant fatigue, aging equipment.

---

### Markov Chain

Missingness at time $t$ depends on whether $t-1$ was missing, creating realistic "flickering" on/off patterns. Governed by a 2-state Markov chain per (sample, dimension) series:

$$P(\text{missing at } t \mid \text{observed at } t\!-\!1) = p_{\text{onset}}$$
$$P(\text{missing at } t \mid \text{missing at } t\!-\!1) = p_{\text{persist}}$$

The `persist` parameter controls stickiness — how likely a missing state is to continue. `p_onset` is automatically calibrated from the target missing rate using the stationary distribution:

$$\pi_{\text{missing}} = \frac{p_{\text{onset}}}{p_{\text{onset}} + 1 - p_{\text{persist}}}$$

```python
# Moderate bursts (avg ~5 steps)
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="markov", persist=0.8
)

# Long bursts (avg ~20 steps)
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="markov", persist=0.95
)

# Rapid flickering (avg ~1.5 steps)
X_miss, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="markov", persist=0.3
)
```

**Parameters:**
- `persist`: probability of staying missing once entered, range [0, 1) (default: `0.8`). Higher = longer bursts.

**Use cases:** intermittent connectivity, unstable sensor connections, WiFi/Bluetooth dropouts, flickering wearable sensors.

---

## Data Shapes

The library supports both 2D and 3D arrays:

```python
# 2D: single subject, (time × features)
X_2d = np.random.randn(500, 6)       # 500 timesteps, 6 features
X_miss, mask = simulate_missingness(X_2d, "mcar", 0.15, seed=42)

# 3D: multiple subjects, (subjects × time × features)
X_3d = np.random.randn(50, 500, 6)   # 50 subjects, 500 timesteps, 6 features
X_miss, mask = simulate_missingness(X_3d, "mcar", 0.15, seed=42)
```

For 3D data, MAR normalizes the driver signal per-participant and MNAR normalizes per-sample per-dimension, ensuring consistent behavior across subjects with different scales.

---

## Additional API

### Multiple Rates

```python
from ts_missingness import simulate_many_rates

rates = [0.05, 0.15, 0.25, 0.50]
results = simulate_many_rates(X, "mcar", rates, seed=42)

for rate, (X_miss, mask) in results.items():
    print(f"Rate {rate}: actual {(~mask).mean():.4f}")
```

Each rate gets a different seed derived from the base seed for independence.

### Object-Oriented Interface

```python
from ts_missingness import MissingnessSimulator

sim = MissingnessSimulator(
    "mar", missing_rate=0.25, seed=42,
    driver_dims=[0], pattern="block", block_len=10
)
X_miss, mask = sim.generate(X)
```

### Registries

```python
from ts_missingness import MECHANISMS, PATTERNS

print(list(MECHANISMS.keys()))  # ['mcar', 'mar', 'mnar']
print(list(PATTERNS.keys()))    # ['pointwise', 'point', 'scattered', 'block', 'contiguous', 'monotone', 'dropout', 'decay', 'degradation', 'markov', 'flickering']
```

---

## Evaluation Workflow

```python
import numpy as np
from ts_missingness import simulate_missingness

# Ground truth
X = np.random.randn(1000, 6)

# Simulate missingness
X_miss, mask = simulate_missingness(X, "mar", 0.20, seed=42, driver_dims=[0])

# Your imputation method
X_imputed = your_imputation_method(X_miss)

# Evaluate only on artificially masked entries
missing_idx = ~mask
rmse = np.sqrt(np.mean((X[missing_idx] - X_imputed[missing_idx]) ** 2))
mae = np.mean(np.abs(X[missing_idx] - X_imputed[missing_idx]))

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
```

---

## Full Parameter Reference

### `simulate_missingness(X, mechanism, missing_rate, seed=None, pattern="pointwise", **kwargs)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Input data, shape `(T, D)` or `(N, T, D)` |
| `mechanism` | `str` | `"mcar"`, `"mar"`, or `"mnar"` |
| `missing_rate` | `float` | Target fraction missing, 0.0–1.0 (clipped automatically) |
| `seed` | `int \| None` | Random seed for reproducibility |
| `pattern` | `str` | `"pointwise"`, `"block"`, `"monotone"`, `"decay"`, or `"markov"` |

**Mechanism-specific kwargs:**

| Parameter | Mechanisms | Default | Description |
|-----------|-----------|---------|-------------|
| `target` | all | `"all"` | Dimensions to mask: `"all"` or `list[int]` |
| `driver_dims` | MAR | `[0]` | Driver dimensions |
| `driver_weights` | MAR | `None` | Per-driver weights (auto-normalized) |
| `strength` | MAR, MNAR | `2.0` | Dependency strength (≥ 0) |
| `base_rate` | MAR | `0.01` | Minimum probability floor |
| `direction` | MAR | `"positive"` | `"positive"` or `"negative"` |
| `mnar_mode` | MNAR | `"extreme"` | `"high"`, `"low"`, or `"extreme"` |

**Pattern-specific kwargs:**

| Parameter | Patterns | Default | Description |
|-----------|---------|---------|-------------|
| `block_len` | block | `10` | Block length in timesteps |
| `block_density` | block | `0.7` | Fraction of missingness in blocks |
| `decay_rate` | decay | `3.0` | Steepness of temporal ramp |
| `decay_center` | decay | `0.7` | Normalized time of 50% crossover |
| `persist` | markov | `0.8` | Probability of staying missing [0, 1) |

**Returns:** `(X_missing, mask)` — data with NaNs inserted and boolean mask.

---

## Reproducibility

- Uses NumPy's `default_rng(seed)` — no global RNG state
- Same seed → identical masks
- Different seeds → independent results

---

## Existing NaNs

Pre-existing NaNs in the input are preserved and excluded from the eligible pool. The `missing_rate` is applied only to non-NaN entries.

```python
X = np.random.randn(100, 5)
X[:10, 0] = np.nan  # Pre-existing NaNs

X_miss, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
assert np.isnan(X_miss[:10, 0]).all()  # Preserved
```

---

## Testing

```bash
pytest ts_missingness/tests/ -v
```

77 tests covering all mechanisms, patterns, edge cases, extreme rates, numerical stability, and validation.

---

## Citation

```bibtex
@software{ts_missingness,
  author = {Feruz Oripov},
  title = {Time-Series Missingness Simulation Library},
  year = {2026},
  url = {https://github.com/feruzoripov/ts_missingness}
}
```

---

## License

MIT
