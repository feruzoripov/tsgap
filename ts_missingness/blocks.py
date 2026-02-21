"""Block missingness patterns for time-series data."""

import numpy as np
from typing import Tuple


def apply_block_missingness(
    mask: np.ndarray,
    shape: Tuple[int, ...],
    block_len: int = 10,
    block_density: float = 0.7,
    **kwargs
) -> np.ndarray:
    """Apply block (contiguous) missingness pattern.
    
    Converts some point-wise missingness into contiguous blocks.
    
    Parameters
    ----------
    mask : np.ndarray
        Initial boolean mask (True=observed, False=missing)
    shape : tuple
        Shape of the data
    block_len : int
        Length of each missing block (in time steps)
    block_density : float
        Fraction of total missingness allocated to blocks (0.0 to 1.0)
    
    Returns
    -------
    mask : np.ndarray
        Modified mask with block patterns
    """
    if not 0.0 <= block_density <= 1.0:
        raise ValueError("block_density must be between 0.0 and 1.0")
    
    # Count current missing points
    n_missing = (~mask).sum()
    if n_missing == 0:
        return mask
    
    # Determine how many missing points should be in blocks
    n_block_missing = int(n_missing * block_density)
    n_point_missing = n_missing - n_block_missing
    
    # Start fresh: restore some points, then add blocks
    new_mask = mask.copy()
    
    # Get currently missing indices
    missing_indices = np.where(~mask)
    
    # Restore some points (keep only n_point_missing as point-wise)
    if n_point_missing < n_missing:
        # Randomly select which missing points to restore
        n_to_restore = n_missing - n_point_missing
        restore_idx = np.random.choice(
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
        new_mask = _add_blocks(new_mask, shape, block_len, n_block_missing)
    
    return new_mask


def _add_blocks(
    mask: np.ndarray,
    shape: Tuple[int, ...],
    block_len: int,
    n_target: int
) -> np.ndarray:
    """Add contiguous missing blocks to mask."""
    
    if len(shape) == 2:
        T, D = shape
        N = 1
    else:  # 3D
        N, T, D = shape
    
    n_added = 0
    max_attempts = 1000
    attempts = 0
    
    while n_added < n_target and attempts < max_attempts:
        attempts += 1
        
        # Randomly select sample (if 3D), dimension, and start time
        if N > 1:
            n_idx = np.random.randint(0, N)
        else:
            n_idx = 0
        
        d_idx = np.random.randint(0, D)
        t_start = np.random.randint(0, max(1, T - block_len + 1))
        t_end = min(t_start + block_len, T)
        
        # Apply block
        if len(shape) == 2:
            block_mask = mask[t_start:t_end, d_idx]
            n_can_add = block_mask.sum()  # Count currently observed
            mask[t_start:t_end, d_idx] = False
        else:  # 3D
            block_mask = mask[n_idx, t_start:t_end, d_idx]
            n_can_add = block_mask.sum()
            mask[n_idx, t_start:t_end, d_idx] = False
        
        n_added += n_can_add
    
    return mask
