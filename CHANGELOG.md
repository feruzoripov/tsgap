# Changelog

## Version 0.5.0 - Scale-Aware Block Missingness

### Added

- Add `block_frac` for relative block lengths based on the time axis
- Support `block_frac` for both 2D `(T, D)` and 3D `(N, T, D)` arrays
- Add validation for invalid `block_frac` values
- Document `block_frac` in README, API docs, and pattern docs

### Changed

- Update benchmark example to use scale-aware block missingness

### Testing

- Add tests for `block_frac` scaling, 3D behavior, precedence over `block_len`, and validation

---

## Version 0.4.0 - JOSS Readiness and Pattern Reliability

### Bug Fixes

- Preserve pre-existing NaNs across all missingness patterns
- Respect `target` dimensions across block, monotone, decay, and Markov patterns
- Keep `X_missing` and returned `mask` consistent for non-pointwise patterns

### Documentation and Examples

- Add focused documentation pages for installation, concepts, mechanisms, patterns, API usage, and benchmarking
- Add a runnable imputation benchmark example in `examples/benchmark_imputation.py`
- Add and update JOSS paper draft materials under `paper/`

### Testing and Tooling

- Expand behavioral tests for MAR direction, MNAR value targeting, block run lengths, decay timing, and Markov bursts
- Increase the test suite to 103 tests
- Add Ruff linting configuration
- Add coverage reporting and Python 3.9-3.13 CI coverage

---

## Version 0.1.1 - Critical Fixes

### Critical Bug Fixes

#### 1. Calibration Bracketing Logic (Fixed) ⚠️
- **Before**: Bound expansion logic was reversed, causing calibration to fail for extreme rates
- **After**: Correct bracketing - expands bounds in proper direction
- **Impact**: Now handles extreme missing rates (1%, 90%) correctly

#### 2. MAR Normalization for 3D Data (Improved)
- **Before**: Global normalization across all participants
- **After**: Per-participant normalization for 3D data
- **Impact**: More consistent MAR behavior across subjects with different scales

#### 3. Base Rate Handling (Fixed)
- **Before**: `base_rate` could conflict with low `missing_rate`
- **After**: Automatically capped at `missing_rate * 0.5`
- **Impact**: Prevents calibration issues with very low missing rates

#### 4. Probability Zeroing (Improved)
- **Before**: Non-eligible positions handled after sampling
- **After**: Probabilities zeroed before sampling for cleaner semantics
- **Impact**: Slightly faster and more explicit logic

### Testing

Added extreme rate testing:
- Low rates: 1%, 2%, 5% - all within 0.4% of target
- High rates: 50%, 70%, 90% - all within 1.3% of target
- MCAR remains exact at all rates

---

## Version 0.1.0 - Initial Release

### Key Improvements

#### 1. Reproducibility (Fixed)
- **Before**: Used global `np.random.seed()` and direct calls to `np.random.choice()`, `np.random.rand()`
- **After**: Uses `np.random.Generator` with seed passed through all functions
- **Impact**: Fully reproducible results with same seed across all mechanisms

#### 2. Consistent Mask Semantics (Fixed)
- **Before**: MCAR didn't mark existing NaNs as False in returned mask
- **After**: All mechanisms consistently set `mask[existing_nans] = False`
- **Impact**: Uniform mask interpretation across all mechanisms

#### 3. Target Dimension Support (Added)
- **Before**: Only MCAR supported `target` parameter
- **After**: MAR and MNAR now support `target` parameter
- **Impact**: Can selectively mask specific dimensions while others drive missingness

#### 4. Improved Calibration (Enhanced)
- **Before**: Fixed bounds [-10, 10] for binary search
- **After**: Automatic bound expansion with bracketing
- **Impact**: Handles extreme missing rates (e.g., 0.01, 0.99) correctly

#### 5. Eligible Position Handling (Fixed)
- **Before**: MAR/MNAR calibrated over all positions including existing NaNs
- **After**: Calibration only considers eligible (non-NaN) positions
- **Impact**: More accurate missing rate achievement on real datasets with existing NaNs

#### 6. Performance Optimization (Improved)
- **Before**: Nested loops for MNAR normalization (slow on large datasets)
- **After**: Vectorized normalization using numpy broadcasting
- **Impact**: ~10-100x faster on large 3D arrays

#### 7. Code Organization (Refactored)
- Added helper functions:
  - `_get_eligible_mask()`: Unified eligible position logic
  - `_calibrate_offset()`: Reusable calibration with auto-bracketing
- **Impact**: Cleaner code, easier to maintain and extend

### API Changes

All changes are backward compatible. New optional parameters:
- `rng`: `np.random.Generator` for explicit RNG control
- `target`: Now supported in MAR and MNAR (was MCAR-only)

### Testing

All 17 unit tests pass:
- MCAR exact rate control
- MAR/MNAR approximate rate control
- Reproducibility with seeds
- Edge cases (constant signals, existing NaNs)
- Block missingness patterns

### Documentation

- Updated docstrings with clearer parameter descriptions
- Added notes about missing rate being applied to eligible entries
- Clarified mask semantics (True=observed, False=missing)
- Added mathematical formulations for each mechanism in README
