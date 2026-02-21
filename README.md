# Time-Series Missingness Simulation Library

A Python library for simulating realistic missingness patterns in time-series data for imputation benchmarking.

## Features

- Three standard missingness mechanisms: MCAR, MAR, MNAR
- Block (contiguous) missingness patterns for realistic sensor dropout
- Support for 2D (T, D) and 3D (N, T, D) arrays
- Exact missing rate control for MCAR, approximate for MAR/MNAR
- Reproducible with seed control
- Handles existing NaNs and edge cases

## Missingness Mechanisms

### MCAR (Missing Completely At Random)

**Definition**: Missingness is independent of both observed and unobserved data.

**Mathematical formulation**:
```
P(M_ij = 1) = ρ    for all i,j
```
where M_ij is the missingness indicator and ρ is the target missing rate.

**Implementation**:
- Uniform random sampling without replacement
- Exactly ⌊n × ρ⌋ positions are masked from n eligible entries
- Each eligible position has equal probability of being selected

**Use case**: Random sensor failures, data transmission errors

---

### MAR (Missing At Random)

**Definition**: Missingness depends on observed data, but not on the missing values themselves.

**Mathematical formulation**:
```
P(M_ij = 1 | X) = σ(α · z_i + β)

where:
  z_i = (X_i,driver - μ) / σ     [normalized driver signal]
  σ(x) = 1 / (1 + exp(-x))       [sigmoid function]
  α = strength parameter
  β = calibrated offset
```

**Implementation**:
1. Compute driver signal from specified dimensions (e.g., activity level)
2. Normalize: z = (driver - mean) / std
3. Apply sigmoid to get missingness probability
4. Calibrate offset β via binary search to achieve target rate ρ
5. Sample Bernoulli(p_ij) for each position

**Use case**: Sensor failures correlated with other measurements (e.g., heart rate sensor fails during high activity)

---

### MNAR (Missing Not At Random)

**Definition**: Missingness depends on the unobserved (missing) values themselves.

**Mathematical formulation**:
```
P(M_ij = 1 | X_ij) = σ(α · f(z_ij) + β)

where:
  z_ij = (X_ij - μ_j) / σ_j      [per-dimension normalization]
  
  f(z) = { z      if mode = "high"     [high values missing]
         { -z     if mode = "low"      [low values missing]
         { |z|    if mode = "extreme"  [extreme values missing]
  
  σ(x) = 1 / (1 + exp(-x))       [sigmoid function]
  α = strength parameter
  β = calibrated offset
```

**Implementation**:
1. Normalize each dimension independently: z_ij = (X_ij - μ_j) / σ_j
2. Compute score based on mode (high/low/extreme)
3. Apply sigmoid to get missingness probability
4. Calibrate offset β to achieve target rate ρ
5. Sample Bernoulli(p_ij) for each position

**Use case**: 
- **Extreme**: Sensor saturation at extreme values
- **High**: Ceiling effect (values above detection limit)
- **Low**: Floor effect (values below detection limit)

---

### Block Missingness (Optional)

**Definition**: Contiguous segments of missing data, simulating sensor dropout periods.

**Mathematical formulation**:
```
Given initial missingness M, create blocks:
  1. n_block = ⌊n_missing × density⌋
  2. Restore (n_missing - n_block) random points
  3. Add contiguous blocks of length L until n_block points masked
```

**Implementation**:
- Works with any base mechanism (MCAR/MAR/MNAR)
- `block_density`: fraction of missingness in contiguous blocks
- `block_len`: length of each missing segment

**Use case**: Sensor battery depletion, connectivity loss, device removal

---

## Key Parameters Summary

| Parameter | MCAR | MAR | MNAR | Description |
|-----------|------|-----|------|-------------|
| `missing_rate` | ✓ | ✓ | ✓ | Target fraction of missing values (0.0-1.0) |
| `target` | ✓ | ✓ | ✓ | Dimensions to mask: "all" or list of indices |
| `driver_dims` | - | ✓ | - | Dimensions that drive missingness |
| `mnar_mode` | - | - | ✓ | "high", "low", or "extreme" |
| `strength` | - | ✓ | ✓ | Dependency strength (α in formulas) |
| `block` | ✓ | ✓ | ✓ | Enable block missingness |
| `seed` | ✓ | ✓ | ✓ | Random seed for reproducibility |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from ts_missingness import simulate_missingness

# Generate sample data
X = np.random.randn(100, 5)  # 100 timesteps, 5 dimensions

# MCAR: 15% missing completely at random
X_missing, mask = simulate_missingness(X, "mcar", missing_rate=0.15, seed=42)

# MAR: 25% missing depending on dimension 0
X_missing, mask = simulate_missingness(
    X, "mar", missing_rate=0.25, seed=42,
    driver_dims=[0], strength=2.0
)

# MNAR: 10% extreme values missing
X_missing, mask = simulate_missingness(
    X, "mnar", missing_rate=0.10, seed=42,
    mnar_mode="extreme", strength=2.0
)

# Block missingness: 60-timestep blocks
X_missing, mask = simulate_missingness(
    X, "mcar", missing_rate=0.20, seed=42,
    block=True, block_len=60, block_density=0.7
)
```

## Usage Examples

### Multiple Missing Rates

```python
from ts_missingness import simulate_many_rates

rates = [0.05, 0.15, 0.25]
results = simulate_many_rates(X, "mcar", rates, seed=42)

for rate, (X_missing, mask) in results.items():
    print(f"Rate {rate}: {(~mask).sum()} missing values")
```

### Object-Oriented Interface

```python
from ts_missingness import MissingnessSimulator

sim = MissingnessSimulator("mcar", missing_rate=0.15, seed=42)
X_missing, mask = sim.generate(X)
```

### Evaluation for Imputation

```python
# Simulate missingness
X_missing, mask = simulate_missingness(X, "mcar", 0.20, seed=42)

# Your imputation method
X_imputed = your_imputation_method(X_missing)

# Compute metrics only on masked entries
missing_indices = ~mask
rmse = np.sqrt(np.mean((X[missing_indices] - X_imputed[missing_indices])**2))
mae = np.mean(np.abs(X[missing_indices] - X_imputed[missing_indices]))

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
```

## API Reference

### `simulate_missingness(X, mechanism, missing_rate, seed=None, **kwargs)`

Main function to simulate missingness.

Parameters:
- `X`: numpy array of shape (T, D) or (N, T, D)
- `mechanism`: "mcar", "mar", or "mnar"
- `missing_rate`: float between 0.0 and 1.0
- `seed`: optional random seed
- `**kwargs`: mechanism-specific parameters

Returns:
- `X_missing`: array with NaNs inserted
- `mask`: boolean array (True=observed, False=missing)

### Mechanism-Specific Parameters

MCAR:
- `target`: "all" (default) or list of dimension indices to mask

MAR:
- `driver_dims`: list of dimensions that drive missingness (required)
- `strength`: dependency strength (default: 2.0)
- `base_rate`: minimum probability (default: 0.01)
- `direction`: "positive" or "negative" (default: "positive")

MNAR:
- `mnar_mode`: "high", "low", or "extreme" (default: "extreme")
- `strength`: dependency strength (default: 2.0)

Block Missingness (all mechanisms):
- `block`: enable block patterns (default: False)
- `block_len`: length of each block in timesteps (default: 10)
- `block_density`: fraction of missingness in blocks (default: 0.7)

## Testing

```bash
pytest ts_missingness/tests/
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{ts_missingness,
  author = {Feruz Oripov},
  title = {Time-Series Missingness Simulation Library},
  year = {2026},
  url = {https://github.com/feruzoripov/ts_missingness}
}
```

## License

MIT
