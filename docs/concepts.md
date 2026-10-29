# Core Concepts

[Installation](installation.md) |
[Core concepts](concepts.md) |
[Mechanisms](mechanisms.md) |
[Patterns](patterns.md) |
[Mathematical details](mathematical_details.md) |
[Benchmarking workflow](benchmarking.md) |
[API reference](api.md)

TSGap separates missing-data simulation into two composable parts:

- **Mechanisms** describe why data is missing.
- **Patterns** describe how missingness is arranged in time.

This separation lets users combine a statistical influence with a temporal
structure. For example, a MAR-informed missingness budget can be represented as
scattered points, contiguous blocks, monotone dropout, late-series decay, or
bursty Markov segments.

For `pointwise`, MAR and MNAR dependencies are preserved at the individual
timestamp-feature level. For temporal patterns such as `block`, `decay`,
`markov`, and `monotone`, the mechanism first determines the missingness budget
or relative series-level missingness, and the pattern then reshapes that budget
in time. In those cases, describe the result as a temporally structured,
mechanism-informed mask. The final individual missing positions should not be
interpreted as a strict MAR or MNAR data-generating process unless that property
has been established separately.

The Gilbert-Elliott pattern is the exception: it is MCAR-only because it models
an independent burst-loss channel.

For formulas and step-by-step probability calculations, see
[Mathematical details](mathematical_details.md).

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
| `gilbert_elliott` | `gilbert-elliott`, `gilbert`, `burst` | MCAR-only ragged bursts with leaky good/bad periods |

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

If the original input contains NaNs, evaluate only artificially hidden values:

```python
evaluation_idx = ~mask & ~np.isnan(X)
```

## Missing-Rate Denominator and Guarantees

`missing_rate` is the requested artificial missing fraction among eligible
entries. Eligible entries are originally observed values in target dimensions;
pre-existing NaNs and non-target dimensions are excluded. For eligible set `E`:

```{math}
\hat{r} = \frac{\#\{i \in E : mask_i = False\}}{|E|}
```

| Configuration | Rate guarantee |
|---------------|----------------|
| MCAR pointwise | Exact rounded count; rate error is at most `0.5 / \|E\|` |
| Block, monotone, decay | Preserve the mechanism-assigned count when feasible |
| MAR/MNAR pointwise | Calibrated in expectation; realized rate is stochastic |
| Markov, Gilbert-Elliott | Calibrated in expectation when feasible; realized rate is stochastic |

For stochastic configurations there is no deterministic per-run tolerance.
Benchmark reports should record `hat(r)`, the random seed, and preferably
results across multiple seeds. Infeasible temporal-model configurations raise a
clear error instead of silently returning a substantially different rate.

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

## Behavioral Test Coverage

The test suite checks both API invariants and statistical behavior, including:

- MAR positive and negative driver directions
- MNAR high, low, and extreme value targeting
- block patterns producing longer missing runs than pointwise patterns
- decay patterns shifting missingness later in time
- Markov persistence increasing burst lengths
