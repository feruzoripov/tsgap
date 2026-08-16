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
    affiliation: 1
  - name: Enock Adu Bonsu
    affiliation: 2
  - name: Ali Bilgin
    orcid: 0000-0003-4196-4036
    affiliation: "3, 4, 5"
  - name: Shravan Aras
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
date: 16 August 2026
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
missing) as two orthogonal, independently configurable axes. Researchers can
therefore evaluate whether an imputation method that performs well under random
scattered missingness also handles activity-dependent sensor dropout or gradual
sensor degradation.

# Statement of Need

The missing data literature distinguishes three canonical mechanisms
[@rubin1976inference; @little2019statistical]: Missing Completely At Random
(MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). In
time-series data, the *temporal structure* of missingness is equally important:
data may be missing as scattered individual points, contiguous blocks (sensor
dropout), monotone tails (participant dropout), gradually increasing gaps (sensor
degradation), or intermittent bursts (flickering connections).

Existing tools cover parts of this problem.
The `ampute` function in the R package `mice` [@vanbuuren2011mice] provides
multivariate amputation but lacks temporal pattern awareness. PyGrinder
[@du2023pypots] implements Python-native MCAR, MAR, MNAR, and selected
sequential or block missingness generators, but exposes them as separate
functions rather than as one mechanism-pattern composition API. Its
documentation also notes that some final missing rates, such as MCAR with
pre-existing missing values and block missingness, are not strictly controlled.
BenchPOTS, the benchmarking suite in the PyPOTS ecosystem [@du2023pypots],
provides shared preprocessing pipelines for partially observed time-series
datasets and can introduce artificial point, subsequence, or block missingness
during preprocessing. However, BenchPOTS is dataset- and benchmark-oriented
rather than a standalone simulator for composing missingness mechanisms with
temporal patterns on arbitrary user arrays.
Most published imputation benchmarks
[@cao2018brits; @du2023saits; @fortuin2020gpvae] use ad-hoc MCAR-only masking,
providing no control over temporal structure and no support for MAR or MNAR
evaluation. A detailed comparison with these tools is provided in the State of
the Field section.

`tsgap` provides:

- **Mechanism--pattern composability**: 3 mechanisms $\times$ 5 patterns = 15
  distinct missingness configurations, all accessible through a single function
  call.
- **Automatic rate calibration**: Binary search offset calibration for MAR and
  MNAR ensures that researchers can conduct controlled experiments at precise
  target missing rates, rather than accepting the uncontrolled rates produced
  by uncalibrated sigmoid models.
- **Temporal pattern diversity**: Block, monotone, temporal decay, and Markov
  chain patterns capture real-world missingness structures absent from existing
  Python tools.
- **Scale-aware block gaps**: Block lengths can be specified as absolute sample
  counts or as fractions of the time axis, including `(min_frac, max_frac)`
  ranges for variable-length dropout episodes in long wearable-style recordings.
- **Weighted multi-driver MAR**: A weighted linear combination of multiple
  observed variables drives missingness probability, enabling realistic
  multi-factor dependency modeling.
- **Native 3D support**: Operates natively on longitudinal panel data of shape
  $(N, T, D)$, where $N$ denotes the number of subjects, $T$ the number of
  timesteps, and $D$ the number of features.

# State of the Field

Existing approaches to missingness simulation include dedicated R packages,
Python libraries, and manual scripting. Each covers part of the problem, as
summarized in \autoref{comparison}.

The `ampute` function in `mice` [@vanbuuren2011mice] is the most
established dedicated tool. It generates multivariate missingness using
weighted sum scores and supports all three Rubin mechanisms
[@rubin1976inference] (MCAR, MAR, MNAR). However, it operates on tabular data
without temporal awareness: it cannot produce contiguous blocks, monotone
dropout, or other time-dependent structures. It is also unavailable in Python,
which limits its use in predominantly Python-based deep learning imputation
pipelines.

PyGrinder [@du2023pypots] provides Python-native MCAR, MAR, MNAR, sequential,
and block-missing generators as part of the PyPOTS ecosystem. However, these are
separate generator functions with different parameterizations and rate-control
semantics. `tsgap` instead exposes mechanism and pattern as orthogonal arguments
to one API, so the same MCAR, MAR, or MNAR mechanism can be evaluated under any
supported temporal pattern while preserving target-feature constraints and
pre-existing missing values.

BenchPOTS [@du2023pypots] works at a different layer of the workflow. It
standardizes dataset loading, train/validation/test preparation, and task
conversion for partially observed time-series benchmarks. Its missingness
interface supports point, subsequence, and block patterns during preprocessing,
delegating the actual masking operations to PyGrinder. This is useful when
working inside the PyPOTS benchmarking ecosystem, while `tsgap` is intended for
controlled missingness simulation on arbitrary NumPy arrays, including
mechanism-pattern combinations, target-dimension constraints, preservation of
pre-existing missing values, and scale-aware block lengths.

The most common practice in imputation benchmarks remains ad-hoc masking with
`numpy.random` [@cao2018brits; @du2023saits; @fortuin2020gpvae], which typically
supports only MCAR with no temporal structure, no rate calibration, and no
reproducibility guarantees beyond manual seed management.

: Comparison of missingness simulation tools. \label{comparison}

| Feature | TSGap | PyGrinder | BenchPOTS | mice | Ad-hoc |
|---------|:-:|:-:|:-:|:-:|:-:|
| MCAR / MAR / MNAR generators | $\checkmark$ | $\checkmark$ | Via PyGrinder | $\checkmark$ | MCAR only |
| Mechanism--pattern separation | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Point / subsequence / block patterns | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ | Rare |
| Scale-aware block fractions | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Monotone pattern | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Temporal decay pattern | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Markov chain pattern | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Unified mechanism--pattern API | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Target-rate control over eligible entries | $\checkmark$ | Partial | Partial | Partial | $\times$ |
| Weighted multi-driver MAR | $\checkmark$ | $\times$ | $\times$ | $\checkmark$ | $\times$ |
| Arbitrary 2D/3D user arrays | $\checkmark$ | Varies | Dataset pipelines | $\times$ | Varies |
| Python | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ (R) | $\checkmark$ |
| Reproducible explicit RNG | $\checkmark$ | Varies | Varies | $\checkmark$ | Varies |

# Software Design

The central design decision in `tsgap` is the strict separation of mechanisms
and patterns into independent, composable modules. This separation reflects a
conceptual distinction that is well-established in the missing data literature
but not enforced in existing software: *why* data is missing (the probabilistic
relationship between values and missingness) is orthogonal to *how* it is missing
(the temporal structure of the gaps). By making these two axes independently
configurable, `tsgap` supports evaluation across all 15 mechanism-pattern
combinations through a single function call.

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
desired temporal structure while preserving pre-existing missing values,
target-feature eligibility, and consistency between the returned data and mask.
For the block pattern, users may request fixed sample lengths with `block_len`
or relative lengths with `block_frac`; passing a range such as
`block_frac=(0.02, 0.10)` samples a new block length uniformly within that range
for each block, which is more appropriate for long recordings where a fixed
10-sample default would behave like scattered point missingness at the scale of
the full series.

**Core API** (`core.py`) composes mechanisms and patterns through a single entry
point:

```python
X_miss, mask = simulate_missingness(
    X,                          # (T, D) or (N, T, D) array
    mechanism="mar",            # WHY: depends on driver
    missing_rate=0.25,          # calibrated to target
    pattern="block",            # HOW: contiguous dropout
    driver_dims=[0, 1],         # multi-driver
    driver_weights=[0.8, 0.2],  # weighted combination
    block_frac=(0.02, 0.10),    # variable-length blocks
    seed=42                     # reproducible
)
```

The library depends only on NumPy, avoiding heavy dependencies on deep learning
frameworks. This keeps installation lightweight and ensures compatibility with
any downstream imputation pipeline. All randomness flows through NumPy's
`Generator` API with explicit seed propagation, ensuring full reproducibility
without reliance on global RNG state.

# Research Impact Statement

`tsgap` was developed at the University of Arizona to investigate the
sensitivity of time-series imputation algorithms to different missingness
structures. By providing reproducible missingness generation across all
mechanism-pattern combinations, `tsgap` lets researchers benchmark statistical,
machine learning, and deep learning imputation methods under controlled
conditions. This responds to a gap in the imputation literature, where
evaluations are typically limited to MCAR-only masking at low missing rates
[@kazijevs2023deep; @cao2018brits]. The library is pip-installable
(`pip install tsgap`), includes focused documentation with mathematical
descriptions of all mechanisms and patterns, and provides a runnable imputation
benchmark comparing simple baselines across representative missingness
scenarios. The current release is archived with a Zenodo DOI [@tsgapzenodo].
Its 118 automated tests cover mechanism-pattern combinations, edge cases,
extreme rate calibration accuracy (1%--90%), numerical stability,
reproducibility, eligibility guarantees, and behavioral checks such as MAR
direction, MNAR tail targeting, block run lengths, scale-aware block fractions,
variable-length blocks, decay timing, and Markov burst persistence. Continuous
integration runs on Python 3.9--3.13 with Ruff linting and coverage reporting.
The package is released under the MIT license and hosted on GitHub with an open
issue tracker for community use and contribution.

# AI Usage Disclosure

Generative AI tools, including Anthropic Claude and OpenAI Codex/ChatGPT, were
used to assist with code review, test generation, documentation organization,
and manuscript drafting during the development of `tsgap`. All AI-assisted code
and text were reviewed, tested, and validated by the authors to ensure
correctness and adherence to the library's design principles. The authors
assume full responsibility for the final implementation and manuscript.

# Acknowledgements

This work was supported by the Center for Biomedical Informatics and
Biostatistics at the University of Arizona, which provided computational
resources and infrastructure support.

# References
