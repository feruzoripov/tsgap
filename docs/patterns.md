# Patterns

Patterns describe the temporal arrangement of missing values. Every pattern can
be combined with any mechanism.

![3x5 grid showing mechanism and pattern combinations.](../assets/mechanism_pattern_grid.png)

## Pointwise

Individual values are missing as scattered points. This is the default.

```python
X_missing, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
```

Aliases: `point`, `scattered`.

## Block

Missingness is arranged into contiguous segments, modeling sensor dropout or
connectivity loss.

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="block", block_len=10, block_density=0.7
)
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_len` | `10` | Length of each block in timesteps |
| `block_density` | `0.7` | Fraction of missingness allocated to blocks |

Alias: `contiguous`.

## Monotone

Once a feature series becomes missing, all later eligible timesteps remain
missing. This models participant dropout, study withdrawal, or permanent sensor
failure.

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42, pattern="monotone"
)
```

Alias: `dropout`.

## Temporal Decay

Missingness is shifted toward later timesteps using a sigmoid time ramp. This
models gradual degradation, battery drain, or participant fatigue.

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.25, seed=42,
    pattern="decay", decay_rate=5.0, decay_center=0.6
)
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decay_rate` | `3.0` | Steepness of the temporal ramp |
| `decay_center` | `0.7` | Normalized time of the 50% crossover |

Alias: `degradation`.

## Markov

Missingness follows a two-state Markov chain for each sample-feature series.
The `persist` parameter controls how likely missingness is to continue once it
starts.

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="markov", persist=0.8
)
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist` | `0.8` | Probability of staying missing, in `[0, 1)` |

Alias: `flickering`.

## Eligibility Guarantees

All patterns preserve:

- pre-existing NaNs as missing
- non-target dimensions as observed
- shape of the input data
- consistency between `X_missing` and `mask`
