"""Missing data patterns (HOW data is missing)."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

BlockFrac = Union[float, Tuple[float, float]]


def _pattern_context(
    mask: np.ndarray,
    eligible_mask: np.ndarray | None = None,
    forced_missing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pattern eligibility and forced-missing masks."""
    if eligible_mask is None:
        eligible_mask = np.ones_like(mask, dtype=bool)
    else:
        eligible_mask = eligible_mask.astype(bool, copy=False)

    if forced_missing is None:
        forced_missing = np.zeros_like(mask, dtype=bool)
    else:
        forced_missing = forced_missing.astype(bool, copy=False)

    return eligible_mask, forced_missing


def _finalize_pattern_mask(
    mask: np.ndarray,
    eligible_mask: np.ndarray | None = None,
    forced_missing: np.ndarray | None = None,
) -> np.ndarray:
    """Preserve non-eligible observed positions and forced missing positions."""
    eligible_mask, forced_missing = _pattern_context(
        mask, eligible_mask, forced_missing
    )
    finalized = np.ones_like(mask, dtype=bool)
    finalized[eligible_mask] = mask[eligible_mask]
    finalized[forced_missing] = False
    return finalized


def _resolve_block_len(
    shape: tuple[int, ...],
    block_len: int,
    block_frac: BlockFrac | None = None,
) -> int:
    """Resolve absolute block length from block_len or block_frac."""
    if len(shape) == 2:
        T = shape[0]
    else:
        T = shape[1]

    if block_frac is not None:
        if isinstance(block_frac, tuple):
            block_len = _block_len_from_frac(T, block_frac[1])
        else:
            block_len = _block_len_from_frac(T, block_frac)

    if block_len < 1:
        raise ValueError("block_len must be >= 1")

    return min(block_len, T)


def _validate_block_frac(block_frac: object | None) -> BlockFrac | None:
    """Validate and normalize scalar or ranged block_frac values."""
    if block_frac is None:
        return None

    if np.isscalar(block_frac):
        try:
            frac = float(block_frac)
        except (TypeError, ValueError) as exc:
            raise ValueError("block_frac must be a float or a length-2 range") from exc
        if not 0.0 < frac <= 1.0:
            raise ValueError("block_frac must be in (0.0, 1.0]")
        return frac

    try:
        frac_min, frac_max = block_frac
    except (TypeError, ValueError) as exc:
        raise ValueError("block_frac must be a float or a length-2 range") from exc

    try:
        frac_min = float(frac_min)
        frac_max = float(frac_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("block_frac range values must be numeric") from exc

    if not 0.0 < frac_min <= frac_max <= 1.0:
        raise ValueError("block_frac range must satisfy 0.0 < min <= max <= 1.0")

    return frac_min, frac_max


def _block_len_from_frac(T: int, block_frac: float) -> int:
    """Convert a relative block length to an absolute length."""
    return min(max(1, int(np.round(T * block_frac))), T)


def _sample_block_len(
    T: int,
    block_len: int,
    block_frac: BlockFrac | None,
    rng: np.random.Generator,
) -> int:
    """Sample the next block length from fixed or ranged parameters."""
    if block_frac is None:
        return block_len
    if isinstance(block_frac, tuple):
        block_frac = rng.uniform(block_frac[0], block_frac[1])
    return _block_len_from_frac(T, block_frac)


def apply_pointwise_pattern(
    mask: np.ndarray,
    shape: tuple[int, ...] | None = None,
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply point-wise (scattered) missingness pattern.
    
    This is the default pattern - no modification needed.
    Individual points are missing independently.
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    shape : tuple, optional
        Shape of the data (not used, for API consistency)
    rng : np.random.Generator, optional
        Random number generator (not used, for API consistency)
    
    Returns
    -------
    mask : np.ndarray
        Unmodified mask (point-wise is the default)
    """
    return mask


def apply_block_pattern(
    mask: np.ndarray,
    shape: tuple[int, ...],
    block_len: int = 10,
    block_frac: BlockFrac | None = None,
    block_density: float = 1.0,
    eligible_mask: np.ndarray | None = None,
    forced_missing: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply block (contiguous) missingness pattern.
    
    Converts some point-wise missingness into contiguous blocks.
    Simulates realistic sensor dropout periods.
    
    Parameters
    ----------
    mask : np.ndarray
        Initial boolean mask (True=observed, False=missing)
    shape : tuple
        Shape of the data
    block_len : int
        Length of each missing block (in time steps)
    block_frac : float or tuple[float, float], optional
        Relative block length as a fraction of the time axis (0.0, 1.0].
        If a tuple is provided, a new fraction is sampled uniformly from
        ``(min_frac, max_frac)`` for each block. If provided, overrides
        block_len.
    block_density : float
        Fraction of total missingness allocated to blocks (0.0 to 1.0)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    mask : np.ndarray
        Modified mask with block patterns
    """
    if rng is None:
        rng = np.random.default_rng()
    
    eligible_mask, forced_missing = _pattern_context(
        mask, eligible_mask, forced_missing
    )

    block_frac = _validate_block_frac(block_frac)
    block_len = _resolve_block_len(shape, block_len, block_frac)
    if not 0.0 <= block_density <= 1.0:
        raise ValueError("block_density must be between 0.0 and 1.0")
    
    # Count current missing points
    n_missing = ((~mask) & eligible_mask).sum()
    if n_missing == 0:
        return _finalize_pattern_mask(mask, eligible_mask, forced_missing)
    
    # Determine how many missing points should be in blocks
    n_block_missing = int(n_missing * block_density)
    n_point_missing = n_missing - n_block_missing
    
    # Start fresh: restore some points, then add blocks
    new_mask = mask.copy()
    
    # Get currently missing indices
    missing_indices = np.where((~mask) & eligible_mask)
    
    # Restore some points (keep only n_point_missing as point-wise)
    if n_point_missing < n_missing:
        # Randomly select which missing points to restore
        n_to_restore = n_missing - n_point_missing
        restore_idx = rng.choice(
            len(missing_indices[0]), size=n_to_restore, replace=False
        )
        
        if mask.ndim == 2:
            new_mask[
                missing_indices[0][restore_idx],
                missing_indices[1][restore_idx]
            ] = True
        else:  # 3D
            new_mask[
                missing_indices[0][restore_idx],
                missing_indices[1][restore_idx],
                missing_indices[2][restore_idx]
            ] = True
    
    # Add block missingness
    if n_block_missing > 0:
        new_mask = _add_blocks(
            new_mask, shape, block_len, n_block_missing, rng, eligible_mask,
            block_frac=block_frac
        )
    
    return _finalize_pattern_mask(new_mask, eligible_mask, forced_missing)


def _add_blocks(
    mask: np.ndarray,
    shape: tuple[int, ...],
    block_len: int,
    n_target: int,
    rng: np.random.Generator,
    eligible_mask: np.ndarray,
    block_frac: BlockFrac | None = None,
) -> np.ndarray:
    """Add contiguous missing blocks to mask."""
    
    if len(shape) == 2:
        T, D = shape
        N = 1
    else:  # 3D
        N, T, D = shape
    
    if len(shape) == 2:
        eligible_series = [(0, d) for d in range(D) if eligible_mask[:, d].any()]
    else:
        eligible_series = [
            (n, d)
            for n in range(N)
            for d in range(D)
            if eligible_mask[n, :, d].any()
        ]

    if not eligible_series:
        return mask

    n_added = 0
    max_attempts = 1000
    attempts = 0
    
    while n_added < n_target and attempts < max_attempts:
        attempts += 1
        
        # Randomly select eligible sample/dimension and start time
        n_idx, d_idx = eligible_series[rng.integers(0, len(eligible_series))]
        current_block_len = _sample_block_len(T, block_len, block_frac, rng)
        t_start = rng.integers(0, max(1, T - current_block_len + 1))
        t_end = min(t_start + current_block_len, T)
        
        # Limit block length to not overshoot target
        remaining = n_target - n_added
        
        if len(shape) == 2:
            block_slice = (
                mask[t_start:t_end, d_idx]
                & eligible_mask[t_start:t_end, d_idx]
            )
        else:
            block_slice = (
                mask[n_idx, t_start:t_end, d_idx]
                & eligible_mask[n_idx, t_start:t_end, d_idx]
            )
        
        n_can_add = block_slice.sum()  # Count currently observed
        
        # If this block would overshoot, truncate it
        if n_can_add > remaining:
            observed_positions = np.where(block_slice)[0]
            keep_count = n_can_add - remaining  # How many to keep observed
            # Only mask 'remaining' of the observed positions
            to_mask = observed_positions[keep_count:]
            if len(shape) == 2:
                mask[t_start + to_mask, d_idx] = False
            else:
                mask[n_idx, t_start + to_mask, d_idx] = False
            n_added += remaining
        else:
            if len(shape) == 2:
                positions = np.where(block_slice)[0]
                mask[t_start + positions, d_idx] = False
            else:
                positions = np.where(block_slice)[0]
                mask[n_idx, t_start + positions, d_idx] = False
            n_added += n_can_add
    
    return mask


def apply_monotone_pattern(
    mask: np.ndarray,
    shape: tuple[int, ...],
    eligible_mask: np.ndarray | None = None,
    forced_missing: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply monotone missingness pattern.

    Once a dimension goes missing at time t, it stays missing for all t' > t.
    Models participant dropout in longitudinal studies and clinical trials.

    The mechanism mask is used to determine *how much* each dimension should
    be missing (its missing density). Dimensions that the mechanism targeted
    more heavily get earlier dropout times. This preserves the mechanism's
    influence: under MAR, dimensions driven by high-valued drivers drop out
    earlier; under MNAR, dimensions with more extreme values drop out earlier;
    under MCAR, dropout times are roughly uniform.

    The total missing count is preserved by distributing the budget across
    dimensions proportionally to their mechanism-assigned missing densities.

    Parameters
    ----------
    mask : np.ndarray
        Initial boolean mask from mechanism (True=observed, False=missing)
    shape : tuple
        Shape of the data
    rng : np.random.Generator, optional
        Random number generator (not used, kept for API consistency)

    Returns
    -------
    mask : np.ndarray
        Modified mask with monotone constraint enforced
    """
    eligible_mask, forced_missing = _pattern_context(
        mask, eligible_mask, forced_missing
    )

    n_target_missing = int(((~mask) & eligible_mask).sum())

    if n_target_missing == 0:
        return _finalize_pattern_mask(mask, eligible_mask, forced_missing)

    is_2d = len(shape) == 2

    if is_2d:
        T, D = shape
        N = 1
        mask_3d = mask[np.newaxis, :, :]
        eligible_3d = eligible_mask[np.newaxis, :, :]
    else:
        N, T, D = shape
        mask_3d = mask
        eligible_3d = eligible_mask

    # Compute per-(sample, dimension) missing density from the mechanism mask.
    # This tells us how much the mechanism *wanted* each dimension to be missing.
    # density[n, d] = fraction of timesteps missing in that series.
    density = np.zeros((N, D), dtype=float)
    capacity = np.zeros((N, D), dtype=int)
    for n in range(N):
        for d in range(D):
            capacity[n, d] = int(eligible_3d[n, :, d].sum())
            if capacity[n, d] > 0:
                density[n, d] = (
                    ((~mask_3d[n, :, d]) & eligible_3d[n, :, d]).sum()
                    / capacity[n, d]
                )

    # Allocate the total missing budget across dimensions proportionally
    # to their mechanism densities. This preserves the mechanism's influence.
    total_density = density.sum()
    if total_density < 1e-10:
        # No mechanism signal — fall back to equal allocation across eligible
        density = (capacity > 0).astype(float)
        total_density = density.sum()
        if total_density < 1e-10:
            return _finalize_pattern_mask(mask, eligible_mask, forced_missing)

    # For each (n, d), compute how many timesteps should be missing
    # (i.e., T - dropout_time = allocated missing count for that series)
    dropout_times = np.full((N, D), T, dtype=int)
    allocated = 0

    for n in range(N):
        for d in range(D):
            # Proportional share of the total missing budget
            share = density[n, d] / total_density
            n_missing_nd = int(np.round(share * n_target_missing))
            # Clamp to valid range
            n_missing_nd = max(0, min(capacity[n, d], n_missing_nd))
            dropout_times[n, d] = T - n_missing_nd
            allocated += n_missing_nd

    # Fix rounding errors: adjust to hit exact target
    diff = allocated - n_target_missing
    if diff != 0:
        # Sort dimensions by density (descending) for adjustment priority
        indices = []
        for n in range(N):
            for d in range(D):
                indices.append((n, d, density[n, d]))
        indices.sort(key=lambda x: x[2], reverse=True)

        if diff > 0:
            # Allocated too many — push dropout times later (reduce missing)
            for n_i, d_i, _ in indices:
                if diff <= 0:
                    break
                if dropout_times[n_i, d_i] < T:
                    dropout_times[n_i, d_i] += 1
                    diff -= 1
        else:
            # Allocated too few — push dropout times earlier (add missing)
            for n_i, d_i, _ in reversed(indices):
                if diff >= 0:
                    break
                if dropout_times[n_i, d_i] > T - capacity[n_i, d_i]:
                    dropout_times[n_i, d_i] -= 1
                    diff += 1

    # Build the monotone mask
    new_mask = np.ones((N, T, D), dtype=bool)
    for n in range(N):
        for d in range(D):
            if dropout_times[n, d] < T:
                eligible_tail = eligible_3d[n, dropout_times[n, d]:, d]
                tail_positions = np.where(eligible_tail)[0]
                new_mask[n, dropout_times[n, d] + tail_positions, d] = False

    if is_2d:
        return _finalize_pattern_mask(new_mask[0], eligible_mask, forced_missing)
    return _finalize_pattern_mask(new_mask, eligible_mask, forced_missing)


def apply_temporal_decay_pattern(
    mask: np.ndarray,
    shape: tuple[int, ...],
    decay_rate: float = 3.0,
    decay_center: float = 0.7,
    eligible_mask: np.ndarray | None = None,
    forced_missing: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply temporal decay missingness pattern.

    Missingness probability increases over time, modeling sensor degradation,
    battery drain, or participant fatigue. Early timesteps have low
    missingness; later timesteps have high missingness.

    Uses a sigmoid ramp over the time axis:

        w(t) = σ(decay_rate × (t_norm - decay_center))

    where t_norm ∈ [0, 1] is the normalized time position. The mechanism
    mask is then resampled with time-weighted probabilities to preserve
    the overall missing count.

    Parameters
    ----------
    mask : np.ndarray
        Initial boolean mask from mechanism (True=observed, False=missing)
    shape : tuple
        Shape of the data
    decay_rate : float
        Steepness of the temporal ramp (higher = sharper transition).
        Default 3.0 gives a smooth S-curve.
    decay_center : float
        Normalized time position (0-1) where missingness reaches 50%.
        Default 0.7 means most missingness concentrates in the last 30%.
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    mask : np.ndarray
        Modified mask with temporally increasing missingness
    """
    if rng is None:
        rng = np.random.default_rng()

    eligible_mask, forced_missing = _pattern_context(
        mask, eligible_mask, forced_missing
    )

    if decay_rate < 0:
        raise ValueError("decay_rate must be >= 0")
    if not 0.0 <= decay_center <= 1.0:
        raise ValueError("decay_center must be between 0.0 and 1.0")

    n_target_missing = ((~mask) & eligible_mask).sum()
    if n_target_missing == 0:
        return _finalize_pattern_mask(mask, eligible_mask, forced_missing)

    if len(shape) == 2:
        T, D = shape
    else:
        _, T, D = shape

    # Build time weights using sigmoid ramp
    t_norm = np.linspace(0, 1, T)
    weights_1d = 1 / (1 + np.exp(-decay_rate * (t_norm - decay_center)))
    # Ensure minimum weight so early timesteps aren't completely immune
    weights_1d = np.maximum(weights_1d, 0.01)

    # Broadcast weights to full data shape
    if len(shape) == 2:
        weights = np.broadcast_to(weights_1d[:, np.newaxis], shape).copy()
    else:  # 3D
        weights = np.broadcast_to(
            weights_1d[np.newaxis, :, np.newaxis], shape
        ).copy()

    # Start with all observed, then sample n_target_missing eligible positions
    # weighted by temporal decay.
    new_mask = np.ones_like(mask, dtype=bool)

    # Flatten for weighted sampling
    flat_weights = weights.ravel() * eligible_mask.ravel().astype(float)

    # Normalize
    weight_sum = flat_weights.sum()
    if weight_sum < 1e-10:
        return mask  # Can't resample, return original

    flat_probs = flat_weights / weight_sum

    # Sample without replacement
    n_to_sample = min(int(n_target_missing), len(flat_probs))
    chosen = rng.choice(
        len(flat_probs), size=n_to_sample, replace=False, p=flat_probs
    )

    flat_mask = new_mask.ravel()
    flat_mask[chosen] = False
    new_mask = flat_mask.reshape(shape)

    return _finalize_pattern_mask(new_mask, eligible_mask, forced_missing)


def apply_markov_pattern(
    mask: np.ndarray,
    shape: tuple[int, ...],
    persist: float = 0.8,
    eligible_mask: np.ndarray | None = None,
    forced_missing: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply Markov chain temporal dependence pattern.

    Missingness at time t depends on whether t-1 was missing, creating
    realistic "flickering" on/off patterns common in wearable sensor data.

    Governed by a 2-state Markov chain per (sample, dimension) series:

        P(missing at t | observed at t-1) = p_onset
        P(missing at t | missing  at t-1) = p_persist

    The persist parameter controls "stickiness" — how likely a missing
    state is to continue. Higher values create longer missing bursts;
    lower values create rapid flickering.

    p_onset is automatically calibrated from the target missing count
    using the stationary distribution:

        π_missing = p_onset / (p_onset + 1 - p_persist)

    Solving for p_onset:

        p_onset = π_missing × (1 - p_persist) / (1 - π_missing)

    The chain is simulated independently for each (sample, dimension)
    series. The mechanism mask's total missing count is preserved
    approximately.

    Parameters
    ----------
    mask : np.ndarray
        Initial boolean mask from mechanism (True=observed, False=missing)
    shape : tuple
        Shape of the data
    persist : float
        Probability of staying in the missing state once entered.
        Range [0, 1). Higher = longer missing bursts.
        Default 0.8 creates moderate-length bursts.
        Must be strictly less than 1.0.
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    mask : np.ndarray
        Modified mask with Markov-chain temporal dependence
    """
    if rng is None:
        rng = np.random.default_rng()

    if not 0.0 <= persist < 1.0:
        raise ValueError("persist must be in [0, 1), got {:.4f}".format(persist))

    eligible_mask, forced_missing = _pattern_context(
        mask, eligible_mask, forced_missing
    )

    n_target_missing = int(((~mask) & eligible_mask).sum())
    if n_target_missing == 0:
        return _finalize_pattern_mask(mask, eligible_mask, forced_missing)

    is_2d = len(shape) == 2

    if is_2d:
        T, D = shape
        N = 1
    else:
        N, T, D = shape

    total_elements = int(eligible_mask.sum())
    if total_elements == 0:
        return _finalize_pattern_mask(mask, eligible_mask, forced_missing)

    # Target missing rate (global)
    pi_missing = n_target_missing / total_elements

    # Calibrate p_onset from stationary distribution:
    # π = p_onset / (p_onset + 1 - persist)
    # => p_onset = π × (1 - persist) / (1 - π)
    if pi_missing >= 1.0:
        # Everything missing
        new_mask = np.ones_like(mask, dtype=bool)
        new_mask[eligible_mask] = False
        return _finalize_pattern_mask(new_mask, eligible_mask, forced_missing)
    if pi_missing <= 0.0:
        return _finalize_pattern_mask(
            np.ones_like(mask, dtype=bool), eligible_mask, forced_missing
        )

    p_onset = pi_missing * (1.0 - persist) / (1.0 - pi_missing)
    # Clamp to valid probability
    p_onset = float(np.clip(p_onset, 0.0, 1.0))

    # Simulate Markov chain for each (sample, dimension) series
    new_mask = np.ones(shape, dtype=bool)

    if is_2d:
        for d in range(D):
            if not eligible_mask[:, d].any():
                continue
            # Initialize: use mechanism's first timestep state
            is_missing = rng.random() < pi_missing
            for t in range(T):
                if not eligible_mask[t, d]:
                    new_mask[t, d] = True
                    continue
                if is_missing:
                    new_mask[t, d] = False
                    # Transition: stay missing with prob persist
                    is_missing = rng.random() < persist
                else:
                    # Transition: become missing with prob p_onset
                    is_missing = rng.random() < p_onset
    else:  # 3D
        for n in range(N):
            for d in range(D):
                if not eligible_mask[n, :, d].any():
                    continue
                is_missing = rng.random() < pi_missing
                for t in range(T):
                    if not eligible_mask[n, t, d]:
                        new_mask[n, t, d] = True
                        continue
                    if is_missing:
                        new_mask[n, t, d] = False
                        is_missing = rng.random() < persist
                    else:
                        is_missing = rng.random() < p_onset

    return _finalize_pattern_mask(new_mask, eligible_mask, forced_missing)


# Pattern registry
PATTERNS = {
    "pointwise": apply_pointwise_pattern,
    "point": apply_pointwise_pattern,       # Alias
    "scattered": apply_pointwise_pattern,   # Alias
    "block": apply_block_pattern,
    "contiguous": apply_block_pattern,      # Alias
    "monotone": apply_monotone_pattern,
    "dropout": apply_monotone_pattern,      # Alias
    "decay": apply_temporal_decay_pattern,
    "degradation": apply_temporal_decay_pattern,  # Alias
    "markov": apply_markov_pattern,
    "flickering": apply_markov_pattern,     # Alias
}
