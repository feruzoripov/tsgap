"""Missingness mechanism implementations."""

import numpy as np
from typing import Optional, Union, List


def apply_mcar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    target: Union[str, List[int]] = "all",
    **kwargs
) -> np.ndarray:
    """Apply MCAR (Missing Completely At Random) mechanism.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
    target : str or list[int]
        "all" or list of dimension indices to mask
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    mask = np.ones_like(X, dtype=bool)
    
    # Determine which dimensions to mask
    if target == "all":
        eligible = ~existing_nans
    else:
        # Mask only specified dimensions
        eligible = np.zeros_like(X, dtype=bool)
        if X.ndim == 2:
            eligible[:, target] = ~existing_nans[:, target]
        else:  # 3D
            eligible[:, :, target] = ~existing_nans[:, :, target]
    
    # Count eligible elements
    n_eligible = eligible.sum()
    n_to_mask = int(np.round(n_eligible * missing_rate))
    
    if n_to_mask == 0:
        return mask
    
    # Sample without replacement for exact rate
    eligible_indices = np.where(eligible.ravel())[0]
    if n_to_mask > len(eligible_indices):
        n_to_mask = len(eligible_indices)
    
    masked_indices = np.random.choice(
        eligible_indices, size=n_to_mask, replace=False
    )
    
    # Apply mask
    mask_flat = mask.ravel()
    mask_flat[masked_indices] = False
    mask = mask_flat.reshape(X.shape)
    
    return mask


def apply_mar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    driver_dims: Optional[List[int]] = None,
    strength: float = 2.0,
    base_rate: float = 0.01,
    direction: str = "positive",
    **kwargs
) -> np.ndarray:
    """Apply MAR (Missing At Random) mechanism.
    
    Missingness depends on driver dimensions (other observed variables).
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
    driver_dims : list[int], optional
        Dimensions that drive missingness (default: first dimension)
    strength : float
        Dependency strength (higher = stronger dependency)
    base_rate : float
        Minimum probability to avoid all-zeros
    direction : str
        "positive" (high driver -> high missing) or "negative"
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    if driver_dims is None:
        driver_dims = [0]
    
    mask = np.ones_like(X, dtype=bool)
    
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
    
    # Calibrate to achieve target missing rate
    # Use binary search to find the right offset
    def compute_rate(offset):
        probs = 1 / (1 + np.exp(-(strength * driver_norm + offset)))
        probs = np.maximum(probs, base_rate)
        probs = np.broadcast_to(probs, X.shape).copy()
        probs[existing_nans] = 0
        return probs.mean()
    
    # Binary search for offset
    offset_low, offset_high = -10.0, 10.0
    for _ in range(20):
        offset_mid = (offset_low + offset_high) / 2
        rate = compute_rate(offset_mid)
        if rate < missing_rate:
            offset_low = offset_mid  # Need higher offset for more missing
        else:
            offset_high = offset_mid
    
    offset = (offset_low + offset_high) / 2
    
    # Compute final probabilities
    probs = 1 / (1 + np.exp(-(strength * driver_norm + offset)))
    probs = np.maximum(probs, base_rate)
    probs = np.broadcast_to(probs, X.shape).copy()
    probs[existing_nans] = 0
    
    # Sample missingness
    mask = np.random.rand(*X.shape) > probs
    mask[existing_nans] = False  # Keep existing NaNs as missing
    
    return mask


def apply_mnar(
    X: np.ndarray,
    missing_rate: float,
    existing_nans: np.ndarray,
    mnar_mode: str = "extreme",
    strength: float = 2.0,
    **kwargs
) -> np.ndarray:
    """Apply MNAR (Missing Not At Random) mechanism.
    
    Missingness depends on the value itself.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    missing_rate : float
        Target missing rate
    existing_nans : np.ndarray
        Boolean mask of existing NaNs
    mnar_mode : str
        "high" (high values missing), "low" (low values missing),
        or "extreme" (extreme values missing)
    strength : float
        Dependency strength
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    mask = np.ones_like(X, dtype=bool)
    
    # Normalize per dimension (handle constant signals)
    if X.ndim == 2:
        X_norm = np.zeros_like(X)
        for d in range(X.shape[1]):
            col = X[:, d]
            col_std = np.nanstd(col)
            if col_std > 1e-10:
                X_norm[:, d] = (col - np.nanmean(col)) / col_std
    else:  # 3D (N, T, D)
        X_norm = np.zeros_like(X)
        for n in range(X.shape[0]):
            for d in range(X.shape[2]):
                col = X[n, :, d]
                col_std = np.nanstd(col)
                if col_std > 1e-10:
                    X_norm[n, :, d] = (col - np.nanmean(col)) / col_std
    
    # Compute score based on mode
    if mnar_mode == "high":
        score = X_norm
    elif mnar_mode == "low":
        score = -X_norm
    elif mnar_mode == "extreme":
        score = np.abs(X_norm)
    else:
        raise ValueError(f"Unknown mnar_mode: {mnar_mode}")
    
    # Calibrate to achieve target missing rate
    def compute_rate(offset):
        probs = 1 / (1 + np.exp(-(strength * score + offset)))
        probs_copy = probs.copy()
        probs_copy[existing_nans] = 0
        return probs_copy.mean()
    
    # Binary search for offset
    offset_low, offset_high = -10.0, 10.0
    for _ in range(20):
        offset_mid = (offset_low + offset_high) / 2
        rate = compute_rate(offset_mid)
        if rate < missing_rate:
            offset_low = offset_mid  # Need higher offset for more missing
        else:
            offset_high = offset_mid
    
    offset = (offset_low + offset_high) / 2
    
    # Compute final probabilities
    probs = 1 / (1 + np.exp(-(strength * score + offset)))
    probs[existing_nans] = 0
    
    # Sample missingness
    mask = np.random.rand(*X.shape) > probs
    mask[existing_nans] = False  # Keep existing NaNs as missing
    
    return mask
