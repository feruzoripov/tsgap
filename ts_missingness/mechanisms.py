"""Missingness mechanism implementations."""

import numpy as np
from typing import Optional, Union, List


def _get_eligible_mask(
    X: np.ndarray,
    existing_nans: np.ndarray,
    target: Union[str, List[int]] = "all"
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
    
    Returns
    -------
    eligible : np.ndarray
        Boolean mask (True=eligible for masking)
    """
    if target == "all":
        return ~existing_nans
    
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
    
    Automatically expands bounds if needed.
    
    Parameters
    ----------
    compute_rate_fn : callable
        Function that takes offset and returns achieved rate
    target_rate : float
        Target missing rate
    initial_low : float
        Initial lower bound
    initial_high : float
        Initial upper bound
    max_iterations : int
        Maximum iterations for binary search
    
    Returns
    -------
    offset : float
        Calibrated offset value
    """
    offset_low, offset_high = initial_low, initial_high
    
    # Expand bounds if needed (bracketing)
    rate_low = compute_rate_fn(offset_low)
    rate_high = compute_rate_fn(offset_high)
    
    # Expand lower bound if needed
    while rate_low < target_rate and offset_low > -100:
        offset_low -= 10
        rate_low = compute_rate_fn(offset_low)
    
    # Expand upper bound if needed
    while rate_high > target_rate and offset_high < 100:
        offset_high += 10
        rate_high = compute_rate_fn(offset_high)
    
    # Binary search
    for _ in range(max_iterations):
        offset_mid = (offset_low + offset_high) / 2
        rate = compute_rate_fn(offset_mid)
        if rate < target_rate:
            offset_low = offset_mid
        else:
            offset_high = offset_mid
    
    return (offset_low + offset_high) / 2


def apply_mcar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    target: Union[str, List[int]] = "all",
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> np.ndarray:
    """Apply MCAR (Missing Completely At Random) mechanism.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate (applied to eligible non-NaN entries)
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
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
    
    mask = np.ones_like(X, dtype=bool)
    
    # Get eligible positions
    eligible = _get_eligible_mask(X, existing_nans, target)
    
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
    driver_dims: Optional[List[int]] = None,
    target: Union[str, List[int]] = "all",
    strength: float = 2.0,
    base_rate: float = 0.01,
    direction: str = "positive",
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> np.ndarray:
    """Apply MAR (Missing At Random) mechanism.
    
    Missingness depends on driver dimensions (other observed variables).
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate (applied to eligible entries)
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
    driver_dims : list[int], optional
        Dimensions that drive missingness (default: first dimension)
    target : str or list[int]
        "all" (default) or list of dimension indices to mask
    strength : float
        Dependency strength (higher = stronger dependency)
    base_rate : float
        Minimum probability to avoid all-zeros
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
    
    mask = np.ones_like(X, dtype=bool)
    
    # Get eligible positions
    eligible = _get_eligible_mask(X, existing_nans, target)
    n_eligible = eligible.sum()
    
    if n_eligible == 0:
        mask[existing_nans] = False
        return mask
    
    # Compute driver signal
    if X.ndim == 2:
        driver = X[:, driver_dims].mean(axis=1, keepdims=True)
    else:  # 3D (N, T, D)
        driver = X[:, :, driver_dims].mean(axis=2, keepdims=True)
    
    # Normalize driver (handle constant signals)
    driver_std = np.nanstd(driver)
    if driver_std > 1e-10:
        driver_norm = (driver - np.nanmean(driver)) / driver_std
    else:
        driver_norm = np.zeros_like(driver)
    
    # Compute probabilities using sigmoid
    if direction == "negative":
        driver_norm = -driver_norm
    
    # Calibrate to achieve target missing rate over eligible positions
    def compute_rate(offset):
        probs = 1 / (1 + np.exp(-(strength * driver_norm + offset)))
        probs = np.maximum(probs, base_rate)
        probs_full = np.broadcast_to(probs, X.shape).copy()
        probs_full[~eligible] = 0  # Only consider eligible positions
        return probs_full[eligible].mean()  # Rate over eligible only
    
    offset = _calibrate_offset(compute_rate, missing_rate)
    
    # Compute final probabilities
    probs = 1 / (1 + np.exp(-(strength * driver_norm + offset)))
    probs = np.maximum(probs, base_rate)
    probs_full = np.broadcast_to(probs, X.shape).copy()
    
    # Sample missingness only at eligible positions
    mask_samples = rng.random(X.shape)
    mask = mask_samples > probs_full
    mask[~eligible] = True  # Keep non-eligible as observed
    mask[existing_nans] = False  # Mark existing NaNs as missing
    
    return mask


def apply_mnar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    mnar_mode: str = "extreme",
    target: Union[str, List[int]] = "all",
    strength: float = 2.0,
    rng: Optional[np.random.Generator] = None,
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
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
    mnar_mode : str
        "high" (high values missing), "low" (low values missing),
        or "extreme" (extreme values missing)
    target : str or list[int]
        "all" (default) or list of dimension indices to mask
    strength : float
        Dependency strength
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    if rng is None:
        rng = np.random.default_rng()
    
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
        # Vectorized normalization for 3D (per sample, per dimension)
        for n in range(X.shape[0]):
            means = np.nanmean(X[n], axis=0, keepdims=True)
            stds = np.nanstd(X[n], axis=0, keepdims=True)
            stds = np.where(stds > 1e-10, stds, 1.0)
            X_norm[n] = (X[n] - means) / stds
    
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
        probs = 1 / (1 + np.exp(-(strength * score + offset)))
        probs[~eligible] = 0  # Only consider eligible positions
        return probs[eligible].mean()  # Rate over eligible only
    
    offset = _calibrate_offset(compute_rate, missing_rate)
    
    # Compute final probabilities
    probs = 1 / (1 + np.exp(-(strength * score + offset)))
    
    # Sample missingness only at eligible positions
    mask_samples = rng.random(X.shape)
    mask = mask_samples > probs
    mask[~eligible] = True  # Keep non-eligible as observed
    mask[existing_nans] = False  # Mark existing NaNs as missing
    
    return mask
