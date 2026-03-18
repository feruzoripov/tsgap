"""Missingness mechanism implementations."""

from __future__ import annotations

import numpy as np


def _get_eligible_mask(
    X: np.ndarray,
    existing_nans: np.ndarray,
    target: str | list[int] = "all"
) -> np.ndarray:
    """Get mask of eligible positions for missingness injection.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
    target : str or list[int]
        "all" or list of dimension indices to mask
        (also accepts tuple or numpy array)
    
    Returns
    -------
    eligible : np.ndarray
        Boolean mask (True=eligible for masking)
    """
    # Check if target is "all" (handle string comparison safely)
    if isinstance(target, str) and target == "all":
        return ~existing_nans
    
    # Convert to list for uniform handling (accepts tuple, array, etc.)
    target = list(target)
    
    # Validate target dimensions
    n_dims = X.shape[-1]
    if any(d < 0 or d >= n_dims for d in target):
        raise ValueError(
            f"target dimensions out of range. "
            f"Got {target}, but data has {n_dims} dimensions (0-{n_dims-1})"
        )
    
    # Mask only specified dimensions
    eligible = np.zeros_like(X, dtype=bool)
    if X.ndim == 2:
        eligible[:, target] = ~existing_nans[:, target]
    else:  # 3D
        eligible[:, :, target] = ~existing_nans[:, :, target]
    
    return eligible


def _calibrate_offset(
    compute_rate_fn,
    target_rate: float,
    initial_low: float = -10.0,
    initial_high: float = 10.0,
    max_iterations: int = 30
) -> float:
    """Calibrate offset to achieve target missing rate using binary search.
    
    Automatically expands bounds if needed. Guards against bracketing failure.
    
    Assumes compute_rate_fn is monotonically increasing with offset:
    higher offset → higher probability → higher missing rate
    
    Note: Caller should handle edge cases (target_rate <= 0 or >= 1) before
    calling this function.
    
    Parameters
    ----------
    compute_rate_fn : callable
        Function that takes offset and returns achieved rate
    target_rate : float
        Target missing rate (should be in (0, 1))
    initial_low : float
        Initial lower bound
    initial_high : float
        Initial upper bound
    max_iterations : int
        Maximum iterations for binary search
    
    Returns
    -------
    offset : float
        Calibrated offset value (clamped to [-50, 50] for numerical stability)
    """
    offset_low, offset_high = initial_low, initial_high
    
    # Expand bounds if needed (bracketing)
    rate_low = compute_rate_fn(offset_low)
    rate_high = compute_rate_fn(offset_high)
    
    # If low rate is too HIGH, push offset_low down (decreases rate)
    while rate_low > target_rate and offset_low > -50:
        offset_low = max(-50, offset_low - 10)
        rate_low = compute_rate_fn(offset_low)
    
    # If high rate is too LOW, push offset_high up (increases rate)
    while rate_high < target_rate and offset_high < 50:
        offset_high = min(50, offset_high + 10)
        rate_high = compute_rate_fn(offset_high)
    
    # Guard against bracketing failure
    if rate_low > target_rate and rate_high > target_rate:
        # Both bounds give rates higher than target → return lowest achievable
        return np.clip(offset_low, -50, 50)
    if rate_low < target_rate and rate_high < target_rate:
        # Both bounds give rates lower than target → return highest achievable
        return np.clip(offset_high, -50, 50)
    
    # Binary search: rate_low <= target <= rate_high
    for _ in range(max_iterations):
        offset_mid = (offset_low + offset_high) / 2
        rate = compute_rate_fn(offset_mid)
        if rate < target_rate:
            offset_low = offset_mid  # Need higher offset for more missing
        else:
            offset_high = offset_mid
    
    # Clamp final offset for numerical stability
    return np.clip((offset_low + offset_high) / 2, -50, 50)


def apply_mcar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    target: str | list[int] = "all",
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply MCAR (Missing Completely At Random) mechanism.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate (applied to eligible non-NaN entries)
        Will be clipped to [0.0, 1.0]
    existing_nans : np.ndarray
        Boolean mask of existing NaNs (should be np.isnan(X) from original data)
    target : str or list[int]
        "all" (default) or list of dimension indices to mask
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Clip missing_rate to valid range
    missing_rate = float(np.clip(missing_rate, 0.0, 1.0))
    
    mask = np.ones_like(X, dtype=bool)
    
    # Early return for edge cases
    if missing_rate <= 0:
        mask[existing_nans] = False
        return mask
    
    # Get eligible positions
    eligible = _get_eligible_mask(X, existing_nans, target)
    
    if missing_rate >= 1:
        # Mask all eligible positions
        mask[eligible] = False
        mask[existing_nans] = False
        return mask
    
    # Count eligible elements
    n_eligible = eligible.sum()
    n_to_mask = int(np.round(n_eligible * missing_rate))
    
    if n_to_mask == 0:
        mask[existing_nans] = False
        return mask
    
    # Sample without replacement for exact rate
    eligible_indices = np.where(eligible.ravel())[0]
    if n_to_mask > len(eligible_indices):
        n_to_mask = len(eligible_indices)
    
    masked_indices = rng.choice(
        eligible_indices, size=n_to_mask, replace=False
    )
    
    # Apply mask
    mask_flat = mask.ravel()
    mask_flat[masked_indices] = False
    mask = mask_flat.reshape(X.shape)
    
    # Mark existing NaNs as missing
    mask[existing_nans] = False
    
    return mask


def apply_mar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    driver_dims: list[int] | None = None,
    driver_weights: list[float] | None = None,
    target: str | list[int] = "all",
    strength: float = 2.0,
    base_rate: float = 0.01,
    direction: str = "positive",
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply MAR (Missing At Random) mechanism.
    
    Missingness depends on driver dimensions (other observed variables).
    
    Masking probability is defined per time step as a logistic function of
    a driver variable; masking is then sampled independently across eligible
    features at each time step.
    
    When multiple driver dimensions are specified, the driver signal is
    computed as a weighted linear combination:
    
        driver_t = Σ_k  w_k × X_{t,k}
    
    where w_k are the (normalized) driver_weights. This allows different
    observed variables to contribute differently to missingness probability.
    If driver_weights is None, all drivers contribute equally (simple mean).
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate (applied to eligible entries)
        Will be clipped to [0.0, 1.0]
    existing_nans : np.ndarray
        Boolean mask of existing NaNs (should be np.isnan(X) from original data)
    driver_dims : list[int], optional
        Dimensions that drive missingness (default: first dimension)
    driver_weights : list[float], optional
        Weights for each driver dimension. Must have same length as
        driver_dims. Weights are normalized to sum to 1.
        Default: equal weights (simple mean).
    target : str or list[int]
        "all" (default) or list of dimension indices to mask
    strength : float
        Dependency strength (higher = stronger dependency, must be >= 0)
    base_rate : float
        Minimum probability to avoid all-zeros (should be < missing_rate)
    direction : str
        "positive" (high driver -> high missing) or "negative"
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if driver_dims is None:
        driver_dims = [0]
    
    # Validate and normalize driver_weights
    if driver_weights is not None:
        if len(driver_weights) != len(driver_dims):
            raise ValueError(
                f"driver_weights length ({len(driver_weights)}) must match "
                f"driver_dims length ({len(driver_dims)})"
            )
        if any(w < 0 for w in driver_weights):
            raise ValueError("driver_weights must be non-negative")
        w_sum = sum(driver_weights)
        if w_sum < 1e-10:
            raise ValueError("driver_weights must not all be zero")
        driver_weights = [w / w_sum for w in driver_weights]
    
    # Validate strength
    if strength < 0:
        raise ValueError("strength must be >= 0")
    
    # Validate direction
    if direction not in ("positive", "negative"):
        raise ValueError(
            f"direction must be 'positive' or 'negative', got '{direction}'"
        )
    
    # Clip missing_rate to valid range
    missing_rate = float(np.clip(missing_rate, 0.0, 1.0))
    
    # Validate driver dimensions
    n_dims = X.shape[-1]
    if any(d < 0 or d >= n_dims for d in driver_dims):
        raise ValueError(
            f"driver_dims out of range. "
            f"Got {driver_dims}, but data has {n_dims} dimensions (0-{n_dims-1})"
        )
    
    # Early return for edge cases
    if missing_rate <= 0:
        mask = np.ones_like(X, dtype=bool)
        mask[existing_nans] = False
        return mask
    
    if missing_rate >= 1:
        # Mask all eligible positions
        mask = np.zeros_like(X, dtype=bool)
        eligible = _get_eligible_mask(X, existing_nans, target)
        mask[~eligible] = True  # Keep non-eligible as observed
        mask[existing_nans] = False  # Mark existing NaNs as missing
        return mask
    
    # Cap base_rate to avoid conflicts (must be < missing_rate)
    base_rate = min(base_rate, max(1e-6, 0.5 * missing_rate))
    
    mask = np.ones_like(X, dtype=bool)
    
    # Get eligible positions
    eligible = _get_eligible_mask(X, existing_nans, target)
    n_eligible = eligible.sum()
    
    if n_eligible == 0:
        mask[existing_nans] = False
        return mask
    
    # Compute driver signal (weighted linear combination)
    if X.ndim == 2:
        if driver_weights is None:
            driver = X[:, driver_dims].mean(axis=1, keepdims=True)
        else:
            w = np.array(driver_weights)
            driver = (X[:, driver_dims] * w[np.newaxis, :]).sum(
                axis=1, keepdims=True
            )
        # Normalize globally for 2D
        driver_std = np.nanstd(driver)
        if driver_std > 1e-10:
            driver_norm = (driver - np.nanmean(driver)) / driver_std
        else:
            driver_norm = np.zeros_like(driver)
    else:  # 3D (N, T, D)
        if driver_weights is None:
            driver = X[:, :, driver_dims].mean(axis=2, keepdims=True)
        else:
            w = np.array(driver_weights)
            driver = (X[:, :, driver_dims] * w[np.newaxis, np.newaxis, :]).sum(
                axis=2, keepdims=True
            )
        # Vectorized per-participant normalization for 3D
        driver_means = np.nanmean(driver, axis=1, keepdims=True)
        driver_stds = np.nanstd(driver, axis=1, keepdims=True)
        driver_stds = np.where(driver_stds > 1e-10, driver_stds, 1.0)
        driver_norm = (driver - driver_means) / driver_stds
    
    # Compute probabilities using sigmoid
    if direction == "negative":
        driver_norm = -driver_norm
    
    # Calibrate to achieve target missing rate over eligible positions
    def compute_rate(offset):
        # Clip logits for numerical stability
        logits = np.clip(strength * driver_norm + offset, -50, 50)
        probs = 1 / (1 + np.exp(-logits))
        probs = np.maximum(probs, base_rate)
        probs_full = np.broadcast_to(probs, X.shape).copy()
        probs_full[~eligible] = 0  # Only consider eligible positions
        return probs_full[eligible].mean()  # Rate over eligible only
    
    offset = _calibrate_offset(compute_rate, missing_rate)
    
    # Compute final probabilities with numerical stability
    logits = np.clip(strength * driver_norm + offset, -50, 50)
    probs = 1 / (1 + np.exp(-logits))
    probs = np.maximum(probs, base_rate)
    probs_full = np.broadcast_to(probs, X.shape).copy()
    probs_full[~eligible] = 0  # Zero out non-eligible before sampling
    
    # Sample missingness
    mask_samples = rng.random(X.shape)
    mask = mask_samples > probs_full
    mask[existing_nans] = False  # Mark existing NaNs as missing
    
    return mask


def apply_mnar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    mnar_mode: str = "extreme",
    target: str | list[int] = "all",
    strength: float = 2.0,
    rng: np.random.Generator | None = None,
    **kwargs
) -> np.ndarray:
    """Apply MNAR (Missing Not At Random) mechanism.
    
    Missingness depends on the value itself.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate (applied to eligible entries)
        Will be clipped to [0.0, 1.0]
    existing_nans : np.ndarray
        Boolean mask of existing NaNs (should be np.isnan(X) from original data)
    mnar_mode : str
        "high" (high values missing), "low" (low values missing),
        or "extreme" (extreme values missing)
    target : str or list[int]
        "all" (default) or list of dimension indices to mask
    strength : float
        Dependency strength (must be >= 0)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Validate strength
    if strength < 0:
        raise ValueError("strength must be >= 0")
    
    # Clip missing_rate to valid range
    missing_rate = float(np.clip(missing_rate, 0.0, 1.0))
    
    # Early return for edge cases
    if missing_rate <= 0:
        mask = np.ones_like(X, dtype=bool)
        mask[existing_nans] = False
        return mask
    
    if missing_rate >= 1:
        # Mask all eligible positions
        mask = np.zeros_like(X, dtype=bool)
        eligible = _get_eligible_mask(X, existing_nans, target)
        mask[~eligible] = True  # Keep non-eligible as observed
        mask[existing_nans] = False  # Mark existing NaNs as missing
        return mask
    
    mask = np.ones_like(X, dtype=bool)
    
    # Get eligible positions
    eligible = _get_eligible_mask(X, existing_nans, target)
    n_eligible = eligible.sum()
    
    if n_eligible == 0:
        mask[existing_nans] = False
        return mask
    
    # Normalize per dimension (vectorized for speed)
    X_norm = np.zeros_like(X, dtype=float)
    
    if X.ndim == 2:
        # Vectorized normalization for 2D
        means = np.nanmean(X, axis=0, keepdims=True)
        stds = np.nanstd(X, axis=0, keepdims=True)
        stds = np.where(stds > 1e-10, stds, 1.0)  # Avoid division by zero
        X_norm = (X - means) / stds
    else:  # 3D (N, T, D)
        # Vectorized per-sample, per-dimension normalization for 3D
        means = np.nanmean(X, axis=1, keepdims=True)
        stds = np.nanstd(X, axis=1, keepdims=True)
        stds = np.where(stds > 1e-10, stds, 1.0)
        X_norm = (X - means) / stds
    
    # Compute score based on mode
    if mnar_mode == "high":
        score = X_norm
    elif mnar_mode == "low":
        score = -X_norm
    elif mnar_mode == "extreme":
        score = np.abs(X_norm)
    else:
        raise ValueError(f"Unknown mnar_mode: {mnar_mode}")
    
    # Calibrate to achieve target missing rate over eligible positions
    def compute_rate(offset):
        # Clip logits for numerical stability
        logits = np.clip(strength * score + offset, -50, 50)
        probs = 1 / (1 + np.exp(-logits))
        probs[~eligible] = 0  # Only consider eligible positions
        return probs[eligible].mean()  # Rate over eligible only
    
    offset = _calibrate_offset(compute_rate, missing_rate)
    
    # Compute final probabilities with numerical stability
    logits = np.clip(strength * score + offset, -50, 50)
    probs = 1 / (1 + np.exp(-logits))
    probs[~eligible] = 0  # Zero out non-eligible before sampling
    
    # Sample missingness
    mask_samples = rng.random(X.shape)
    mask = mask_samples > probs
    mask[existing_nans] = False  # Mark existing NaNs as missing
    
    return mask


# Mechanism registry for easy dispatch
MECHANISMS = {
    "mcar": apply_mcar,
    "mar": apply_mar,
    "mnar": apply_mnar,
}
