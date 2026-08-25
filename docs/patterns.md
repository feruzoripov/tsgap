# Patterns

Patterns describe the temporal arrangement of missing values. Every pattern can
be combined with any mechanism.

For step-by-step formulas for each pattern, see
[Mathematical details](mathematical_details.md).

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
    pattern="block", block_len=10
)
```

By default, the block pattern uses a fixed `block_len=10` and
`block_density=1.0`, so all requested missingness is allocated to blocks. For
long wearable-style time series, prefer `block_frac` to define block length
relative to the time axis:

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="block", block_frac=0.01
)
```

For a series with 30,000 timesteps, `block_frac=0.01` creates blocks of about
300 timesteps. If both `block_len` and `block_frac` are provided, `block_frac`
takes precedence.

To simulate variable-length dropout episodes, pass a `(min_frac, max_frac)`
range. A new block length is sampled uniformly from the range for each block:

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="block", block_frac=(0.01, 0.05)
)
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_len` | `10` | Length of each block in timesteps |
| `block_frac` | `None` | Relative block length as a fraction of the time axis, or `(min_frac, max_frac)` for variable-length blocks. Recommended for long time series. |
| `block_density` | `1.0` | Fraction of missingness allocated to blocks. Set below `1.0` to retain some pointwise missingness. |

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

## Gilbert-Elliott

A two-state burst-loss model widely used to describe bursty packet loss in
telecommunications. Each sample-feature series alternates between a *good* state
(low loss) and a *bad* state (high loss). Unlike the `markov` pattern---where the
bad state is always missing and the good state is never missing---Gilbert-Elliott
produces *ragged* bursts: bad periods still let some values through, and good
periods can have occasional dropouts.

```python
X_missing, mask = simulate_missingness(
    X, "mcar", 0.20, seed=42,
    pattern="gilbert_elliott", persist=0.9,
    bad_loss=0.8, good_loss=0.02
)
```

The overall missing rate is calibrated automatically from `bad_loss`,
`good_loss`, and `persist` to match the requested `missing_rate`. With the
defaults `bad_loss=1.0` and `good_loss=0.0`, the model reduces to the clean
on/off behavior of the `markov` pattern.

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist` | `0.8` | Probability of staying in the bad state, in `[0, 1)`. Higher values create longer bursts. |
| `bad_loss` | `1.0` | Probability a value is missing while in the bad state (`h`), in `(0, 1]`. |
| `good_loss` | `0.0` | Probability a value is missing while in the good state (`k`), in `[0, 1)`. Must be strictly less than `bad_loss`. |

Aliases: `gilbert`, `burst`.

## Eligibility Guarantees

All patterns preserve:

- pre-existing NaNs as missing
- non-target dimensions as observed
- shape of the input data
- consistency between `X_missing` and `mask`
