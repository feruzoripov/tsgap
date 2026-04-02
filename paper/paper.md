---
title: 'TSGap: A Python Library for Composable Time-Series Missingness Simulation'
tags:
  - Python
  - time series
  - missing data
  - imputation
  - benchmarking
  - simulation
authors:
  - name: Feruz Oripov
    orcid: 0009-0001-4303-0512
    affiliation: "1, 3"
  - name: Kseniia Korchagina
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Enock Adu Bonsu
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Ali Bilgin
    orcid: 0000-0003-4196-4036
    affiliation: "3, 4, 5"
  - name: Shravan Aras
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Center for Biomedical Informatics and Biostatistics, University of Arizona, USA
    index: 1
    ror: 03m2x1q45
  - name: Department of Epidemiology and Biostatistics, University of Arizona, USA
    index: 2
    ror: 03m2x1q45
  - name: Department of Electrical and Computer Engineering, University of Arizona, USA
    index: 3
    ror: 03m2x1q45
  - name: Department of Biomedical Engineering, University of Arizona, USA
    index: 4
    ror: 03m2x1q45
  - name: Radiology and Imaging Sciences, University of Arizona, USA
    index: 5
    ror: 03m2x1q45
date: 1 April 2026
bibliography: paper.bib
---

# Summary

Missing data is pervasive in time-series applications, particularly in healthcare
monitoring, wearable sensors, and environmental sensing
[@kazijevs2023deep; @bent2020investigating; @austin2021missing], where data
loss arises from device failures, connectivity drops, patient dropout, and sensor
degradation. Evaluating imputation algorithms requires generating controlled
missingness in complete datasets, yet most benchmarking studies rely on
simplistic random masking that fails to capture the structured, temporally
correlated missingness observed in practice [@kazijevs2023deep].

`tsgap` is a Python library that provides composable, reproducible missingness
simulation for time-series data. Its core design contribution is the explicit
separation of *mechanisms* (why data is missing) from *patterns* (how data is
missing) as two orthogonal, independently configurable axes. This enables
researchers to systematically evaluate imputation methods across realistic
combinations--for example, testing whether an algorithm that performs well under
random scattered missingness also handles activity-dependent sensor dropout or
gradual sensor degradation.

# Statement of Need

The missing data literature distinguishes three canonical mechanisms
[@rubin1976inference; @little2019statistical]: Missing Completely At Random
(MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). In
time-series data, the *temporal structure* of missingness is equally important:
data may be missing as scattered individual points, contiguous blocks (sensor
dropout), monotone tails (participant dropout), gradually increasing gaps (sensor
degradation), or intermittent bursts (flickering connections).

Existing tools address these concerns only partially. The `ampute` function in
the R package `mice` [@vanbuuren2011mice] provides multivariate amputation with
weighted sum scores but lacks temporal pattern awareness and is unavailable in
Python. PyGrinder [@du2023pypots], part of the PyPOTS ecosystem, implements MCAR,
MAR, and MNAR in Python but conflates mechanisms with patterns and does not offer
rate calibration for non-MCAR mechanisms. Most published imputation benchmarks
[@cao2018brits; @du2023saits; @fortuin2020gpvae] use ad-hoc MCAR-only masking
with `numpy.random`, providing no control over temporal structure, no
reproducibility guarantees, and no support for MAR or MNAR evaluation.

`tsgap` addresses these gaps by providing:

- **Mechanism--pattern composability**: 3 mechanisms $\times$ 5 patterns = 15
  distinct missingness configurations, all accessible through a single function
  call.
- **Automatic rate calibration**: Binary search offset calibration for MAR and
  MNAR ensures researchers can conduct controlled experiments at specific missing
  rates, not just whatever rate a sigmoid happens to produce.
- **Temporal pattern diversity**: Block, monotone, temporal decay, and Markov
  chain patterns capture real-world missingness structures absent from existing
  Python tools.
- **Weighted multi-driver MAR**: A weighted linear combination of multiple
  observed variables drives missingness probability, enabling realistic
  multi-factor dependency modeling.
- **Native 3D support**: Operates natively on longitudinal panel data of shape
  $(N, T, D)$, where $N$ denotes the number of subjects, $T$ the number of
  timesteps, and $D$ the number of features.

# State of the Field

Existing approaches to missingness simulation--dedicated R packages, Python libraries, and manual scripting--each address the problem only partially, as summarized in \autoref{comparison}.

The `ampute` function in `mice` [@vanbuuren2011mice] is the most established
dedicated tool. It generates multivariate missingness using weighted sum scores
and supports all three Rubin mechanisms [@rubin1976inference] (MCAR, MAR, MNAR). However, it operates on tabular data
without temporal awareness--it cannot produce contiguous blocks, monotone
dropout, or other time-dependent structures--and is unavailable in Python, which
limits its use in deep learning imputation pipelines that are predominantly
Python-based.

PyGrinder [@du2023pypots] provides Python-native MCAR, MAR, and MNAR generation
as part of the PyPOTS ecosystem. However, it conflates mechanisms with patterns
(e.g., its "block missing" is MCAR-only) and does not offer rate calibration for
MAR or MNAR, making controlled experiments at specific missing rates difficult.
Contributing temporal patterns to PyGrinder would require restructuring its API
around the mechanism--pattern separation that is central to `tsgap`'s design, as
the two libraries are built on fundamentally different abstractions.

The most common practice in imputation benchmarks remains ad-hoc masking with
`numpy.random` [@cao2018brits; @du2023saits; @fortuin2020gpvae], which typically
supports only MCAR with no temporal structure, no rate calibration, and no
reproducibility guarantees beyond manual seed management.

: Comparison of missingness simulation tools. \label{comparison}

| Feature | TSGap | PyGrinder | mice | Ad-hoc |
|---------|:-:|:-:|:-:|:-:|
| MCAR / MAR / MNAR | $\checkmark$ | $\checkmark$ | $\checkmark$ | MCAR only |
| Mechanism--pattern separation | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| Block pattern | $\checkmark$ | $\times$ | $\times$ | Rare |
| Monotone pattern | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| Temporal decay pattern | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| Markov chain pattern | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| Rate calibration (MAR/MNAR) | $\checkmark$ | $\times$ | Partial | $\times$ |
| Weighted multi-driver | $\checkmark$ | $\times$ | $\checkmark$ | $\times$ |
| 3D $(N, T, D)$ native | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| Python | $\checkmark$ | $\checkmark$ | $\times$ (R) | $\checkmark$ |
| Reproducible (seeded RNG) | $\checkmark$ | $\checkmark$ | $\checkmark$ | Varies |

# Software Design

The central design decision in `tsgap` is the strict separation of mechanisms
and patterns into independent, composable modules. This separation reflects a
conceptual distinction that is well-established in the missing data literature
but not enforced in existing software: *why* data is missing (the probabilistic
relationship between values and missingness) is orthogonal to *how* it is missing
(the temporal structure of the gaps). By making these two axes independently
configurable, `tsgap` enables systematic evaluation across all 15
mechanism--pattern combinations through a single function call.

The library's architecture consists of three modules:

**Mechanisms** (`mechanisms.py`) implement the probabilistic models. MCAR uses
uniform sampling without replacement for exact rate control. MAR and MNAR use
logistic probability models of the form $P(M_{ij} = 1) = \sigma(\alpha \cdot s_{ij} + \beta)$,
where $M_{ij}$ is the missingness indicator at timestep $i$ and feature $j$,
$\sigma$ is the sigmoid function, $\alpha$ is a user-specified strength parameter,
$s_{ij}$ is a normalized score (derived from a driver signal for MAR or from the
value itself for MNAR), and $\beta$ is an offset automatically calibrated via
binary search to match the target missing rate. MAR supports weighted multi-driver
signals computed as
$z_i = \sum_k w_k \cdot (X_{i,k} - \mu_k) / \sigma_k$, where $w_k$ are
user-specified weights, $X_{i,k}$ is the value of driver dimension $k$ at
timestep $i$, and $\mu_k$ and $\sigma_k$ are its mean and standard deviation.

**Patterns** (`patterns.py`) reshape the temporal structure of the
mechanism-generated mask. Five patterns are implemented: *pointwise* (scattered
individual points), *block* (contiguous missing segments), *monotone* (once
missing, stays missing), *temporal decay* (missingness increases over time via a
sigmoid ramp), and *Markov chain* (a 2-state chain per series with transition
probabilities calibrated from the stationary distribution). Patterns receive the
mechanism's binary mask and redistribute its missing positions according to the
desired temporal structure, preserving the total missing count.

**Core API** (`core.py`) composes mechanisms and patterns through a single entry
point:

```python
X_miss, mask = simulate_missingness(
    X,                          # (T, D) or (N, T, D) array
    mechanism="mar",            # WHY: depends on driver
    missing_rate=0.25,          # calibrated to target
    pattern="markov",           # HOW: intermittent bursts
    driver_dims=[0, 1],         # multi-driver
    driver_weights=[0.8, 0.2],  # weighted combination
    persist=0.8,                # Markov stickiness
    seed=42                     # reproducible
)
```

The library depends only on NumPy, avoiding heavy dependencies on deep learning
frameworks. This keeps installation lightweight and ensures compatibility with
any downstream imputation pipeline. All randomness flows through NumPy's
`Generator` API with explicit seed propagation, ensuring full reproducibility
without reliance on global RNG state.

# Research Impact Statement

`tsgap` was developed at the University of Arizona to investigate the sensitivity
of time-series imputation algorithms to different missingness structures. The library is pip-installable (`pip install tsgap`),
includes comprehensive documentation with mathematical descriptions of all
mechanisms and patterns, and provides 77 automated tests covering all
mechanism--pattern combinations, edge cases, extreme rate calibration accuracy
(1%--90%), numerical stability, and reproducibility verification. The test suite
runs in under 0.4 seconds. The package is released under the MIT license and
hosted on GitHub with an open issue tracker to facilitate community adoption and
contribution.

# AI Usage Disclosure

Generative AI (Claude Opus 4.6, Anthropic) was used to assist with code
generation during the development of `tsgap`. All AI-generated code was carefully
reviewed, tested, and validated by the authors to ensure correctness and
adherence to the library's design principles. The authors assume full
responsibility for the final implementation.

# Acknowledgements

This work was supported by the Center for Biomedical Informatics and
Biostatistics at the University of Arizona, which provided computational
resources and infrastructure support.

# References
