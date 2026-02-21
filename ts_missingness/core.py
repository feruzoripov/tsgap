"""Core API for missingness simulation."""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any
from .mechanisms import apply_mcar, apply_mar, apply_mnar
from .blocks import apply_block_missingness


def simulate_missingness(
    X: np.ndarray,
    mechanism: str,
    missing_rate: float,
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate missingness in time-series data.
    
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, D) or (N, T, D)
    mechanism : str
        One of "mcar", "mar", "mnar"
    missing_rate : float
        Target fraction of missing values (0.0 to 1.0)
    seed : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Mechanism-specific parameters:
        
        MCAR:
            target : str or list[int]
                "all" (default) or list of dimension indices
        
        MAR:
            driver_dims : list[int], required
                Dimensions that drive missingness
            strength : float, default=2.0
                Dependency strength
            base_rate : float, default=0.01
                Minimum probability
            direction : str, default="positive"
                "positive" or "negative"
        
        MNAR:
            mnar_mode : str, default="extreme"
                "high", "low", or "extreme"
            strength : float, default=2.0
                Dependency strength
        
        Block missingness (all mechanisms):
            block : bool, default=False
                Enable block missingness
            block_len : int, default=10
                Length of each missing block
            block_density : float, default=0.7
                Fraction of missingness in blocks
    
    Returns
    -------
    X_missing : np.ndarray
        Data with NaNs inserted (same shape as X)
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim not in [2, 3]:
        raise ValueError("X must be 2D (T, D) or 3D (N, T, D)")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be between 0.0 and 1.0")
    
    mechanism = mechanism.lower()
    if mechanism not in ["mcar", "mar", "mnar"]:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    # Copy input
    X_missing = X.copy()
    
    # Identify existing NaNs
    existing_nans = np.isnan(X)
    
    # Generate mechanism-specific mask
    if mechanism == "mcar":
        mask = apply_mcar(X, missing_rate, existing_nans, **kwargs)
    elif mechanism == "mar":
        mask = apply_mar(X, missing_rate, existing_nans, **kwargs)
    elif mechanism == "mnar":
        mask = apply_mnar(X, missing_rate, existing_nans, **kwargs)
    
    # Apply block missingness if requested
    if kwargs.get("block", False):
        mask = apply_block_missingness(mask, X.shape, **kwargs)
    
    # Apply mask
    X_missing[~mask] = np.nan
    
    return X_missing, mask


def simulate_many_rates(
    X: np.ndarray,
    mechanism: str,
    rates: List[float],
    seed: Optional[int] = None,
    **kwargs
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Simulate missingness at multiple rates.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    mechanism : str
        Missingness mechanism
    rates : list[float]
        List of missing rates to simulate
    seed : int, optional
        Base random seed
    **kwargs : dict
        Mechanism-specific parameters
    
    Returns
    -------
    dict
        Dictionary mapping rate -> (X_missing, mask)
    """
    results = {}
    for i, rate in enumerate(rates):
        # Use different seed for each rate if seed provided
        rate_seed = None if seed is None else seed + i
        X_missing, mask = simulate_missingness(
            X, mechanism, rate, seed=rate_seed, **kwargs
        )
        results[rate] = (X_missing, mask)
    return results


class MissingnessSimulator:
    """Object-oriented interface for missingness simulation.
    
    Parameters
    ----------
    mechanism : str
        Missingness mechanism ("mcar", "mar", "mnar")
    missing_rate : float
        Target missing rate
    seed : int, optional
        Random seed
    **config : dict
        Mechanism-specific configuration
    
    Examples
    --------
    >>> sim = MissingnessSimulator("mcar", missing_rate=0.15, seed=42)
    >>> X_missing, mask = sim.generate(X)
    """
    
    def __init__(
        self,
        mechanism: str,
        missing_rate: float,
        seed: Optional[int] = None,
        **config
    ):
        self.mechanism = mechanism
        self.missing_rate = missing_rate
        self.seed = seed
        self.config = config
    
    def generate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate missingness for input data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        
        Returns
        -------
        X_missing : np.ndarray
            Data with missingness
        mask : np.ndarray
            Boolean mask
        """
        return simulate_missingness(
            X,
            self.mechanism,
            self.missing_rate,
            self.seed,
            **self.config
        )
