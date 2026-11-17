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
    corresponding: true
    affiliation: "1, 2"
  - name: Kseniia Korchagina
    affiliation: 1
  - name: Ali Bilgin
    orcid: 0000-0003-4196-4036
    affiliation: "2, 3, 4"
  - name: Shravan Aras
    affiliation: 1
affiliations:
  - name: Center for Biomedical Informatics and Biostatistics, University of Arizona, United States
    index: 1
    ror: 03m2x1q45
  - name: School of Electrical, Computing and Software Engineering, University of Arizona, United States
    index: 2
    ror: 03m2x1q45
  - name: Department of Biomedical Engineering, University of Arizona, United States
    index: 3
    ror: 03m2x1q45
  - name: Radiology and Imaging Sciences, University of Arizona, United States
    index: 4
    ror: 03m2x1q45
date: 17 November 2026
bibliography: paper.bib
---

# Summary

A common way to evaluate a method that estimates missing measurements is to
hide observed values and compare the estimates with those held-out
measurements [@schouten2018amputation]. In time-series health data, the
resulting assessment can depend on the proportion and type of missingness, as
well as the dataset [@kazijevs2023deep].

`TSGap` is a Python package for generating missing values in numerical time
series while retaining the original array for evaluation. Users select a rule
for choosing values to hide and a pattern for arranging gaps over time,
including scattered points, contiguous intervals, and dropout tails. The
package returns an incomplete array and a Boolean observation mask, supports
individual series and collections of series, and accepts a random seed to
reproduce a simulation in a fixed software environment [@tsgap2026].

# Statement of need

Missingness mechanisms describe how the probability of an observation being
missing relates to observed and unobserved data: missing completely at random
(MCAR) is independent of data values; missing at random (MAR) permits
dependence on observed information but not additional dependence on unobserved
values; and missing not at random (MNAR) involves such additional dependence
[@rubin1976inference]. Generating missing values under specified assumptions is
an established component of evaluating missing-data methods
[@schouten2018amputation]. For time-series imputation, considering more than
one missingness condition allows an evaluation to examine sensitivity to the
chosen masking procedure [@kazijevs2023deep].

`TSGap` addresses this experimental task for researchers working with
numerical arrays, including longitudinal and sensor measurements, by exposing
selection mechanisms and temporal patterns as separate configuration choices.
Its scope is artificial masking and the associated observation mask; it does
not estimate missingness mechanisms from collected data or perform imputation.
The implemented patterns specify simulation assumptions rather than establish
how a particular instrument or participant actually produces missing
observations.

# State of the field

The `ampute` procedure in the R package `mice` [@vanbuuren2011mice] generates
multivariate missingness using weighted scores and user-specified combinations
of incomplete variables [@miceampute; @schouten2018amputation]. The Python
package `pyampute` implements this approach through a scikit-learn-compatible
transformer with configurable mechanisms and score-to-probability functions
[@pyampute]. These tools provide alternatives when the experimental unit is a
row of a data matrix and the principal configuration concerns which variables
are jointly missing.

PyGrinder provides Python functions for MCAR, logistic MAR, self-dependent
MNAR, and sequential or block missingness [@du2023pypots; @pygrinder].
BenchPOTS provides dataset preprocessing pipelines with point, subsequence,
and block masking options as part of time-series benchmarking workflows
[@du2023pypots; @benchpots].

`TSGap` instead takes user-supplied NumPy arrays and selects entries before
applying a pattern, using explicit target-feature eligibility. The separate
NumPy implementation keeps mask construction independent of dataset loading
and learning frameworks, while treating sample, time, and feature axes
explicitly. This scope differs from the dataset pipelines in BenchPOTS and the
row-amputation interface in `pyampute` [@benchpots; @pyampute]. For patterns
that relocate selected entries, the design trades preservation of the initial
entry-level selection model for control over the temporal arrangement of gaps;
a mechanism label therefore does not by itself characterize the final mask.

# Software design

Version 0.8.0 accepts real numerical arrays with shape $(T,D)$ or $(N,T,D)$,
where $T$ is the number of time steps, $D$ the number of features, and $N$ the
number of samples; NumPy [@harris2020numpy] is its only required runtime
dependency [@tsgap2026]. Separate modules implement selection mechanisms, temporal
patterns, and their composition, allowing pattern construction to use a shared
definition of eligible entries. An entry is eligible when it was originally
observed and belongs to a selected target feature; existing `NaN` values remain
missing and are excluded from the artificial missing-rate denominator. Let $r$
denote the requested fraction of eligible entries to hide. For a nonempty
eligible set $E$, the achieved artificial missing rate is

$$
\widehat r =
\frac{\#\{i\in E:\mathrm{mask}_i=\mathrm{False}\}}{|E|}.
$$

The returned mask marks both original and artificial missing values as false,
so imputation errors should be calculated only at newly hidden, originally
observed entries.

The MCAR selector samples a fixed number of eligible entries uniformly without
replacement. MAR combines available values from specified driver features and
transforms a standardized score into logistic missingness probabilities. MNAR
similarly uses standardized feature values, their negatives, or their absolute
values so that high, low, or extreme observations can receive higher
missingness probabilities. A bounded search calibrates the logistic offset to
the requested mean probability over eligible entries. The subsequent Bernoulli
sampling means that the realized MAR or MNAR count can vary between runs even
when the expected rate is calibrated.

The pointwise pattern retains the selector's mask. Block placement instead uses
fixed lengths or fractions of the time axis, including ranges from which block
lengths are sampled. Monotone dropout masks the final eligible observations of
selected sample-feature series, so pre-existing gaps do not consume the
artificial dropout budget. Temporal decay preserves the selected count while
sampling without replacement from increasing sigmoid time weights.

The Markov pattern generates observed and missing states, while the
Gilbert--Elliott pattern uses hidden good and bad states with separate loss
probabilities [@gilbert1960capacity; @elliott1963estimates]. Both target the
selector's realized eligible-entry rate in expectation rather than preserving
an exact realized count, and infeasible partial-rate configurations are
rejected. Gilbert--Elliott is restricted to MCAR; the other five patterns
accept all three selectors. Temporal rearrangement can discard dependence on
individual driver or target values, so these combinations are structured masks
informed by the initial selection rather than a guarantee that the final
process satisfies MAR or MNAR.

The package requires Python 3.11 or newer, promotes integer input to floating
point, and rejects unsupported array types, infinite values, non-finite
numerical configuration, and unrecognized arguments. Randomness is passed
through a local NumPy generator, avoiding changes to global random state. The
test suite covers reproducibility, target-feature preservation, pre-existing
gaps, missing-rate behavior, invalid configurations, and both two- and
three-dimensional inputs. Continuous integration tests Python 3.11--3.14,
runs Ruff, reports coverage, and builds the Sphinx documentation.

# Research impact statement

`TSGap` was developed for research evaluating how statistical, machine
learning, and deep learning imputation methods respond to different
missingness structures in wearable-derived time series. The repository
supplies a reproducible synthetic benchmark comparing mean imputation, forward
filling, and linear interpolation under six selected mechanism-pattern
conditions, together with scripts for visualizing generated masks. These
materials demonstrate an end-to-end evaluation workflow rather than claim
comparative performance on a population or application dataset.

The package's explicit mechanism-pattern separation supports sensitivity
analyses that would otherwise require researchers to maintain several
experiment-specific masking implementations. Installation instructions,
mathematical descriptions, 184 automated tests, an MIT license, and
contribution guidelines support inspection and reuse of the workflow. Version
0.8.0 is archived on Zenodo with a version-specific DOI [@tsgap2026] and has
completed pyOpenSci software peer review.

# AI usage disclosure

As disclosed in the package README, generative AI tools, including OpenAI
Codex/ChatGPT and Anthropic Claude, assisted with code review, test generation,
documentation organization, and manuscript drafting during development.
OpenAI Codex also assisted with this manuscript revision, including comparison
with version 0.8.0, conversion to the JOSS format, and reference checking.
AI-assisted code and text were reviewed, tested, and validated by the package
authors, who retain full responsibility for the project and manuscript.

# Acknowledgements

The authors acknowledge computational resources and infrastructure support
provided by the Center for Biomedical Informatics and Biostatistics at the
University of Arizona.

# References
