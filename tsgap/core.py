"""Core API for missingness simulation."""

from __future__ import annotations

import numpy as np

from .mechanisms import MECHANISMS, _get_eligible_mask
from .patterns import PATTERNS


def simulate_missingness(
    X: np.ndarray,
    mechanism: str,
    missing_rate: float,
    seed: int | None = None,
    pattern: str = "pointwise",
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate missingness in time-series data.
    
    This function separates two concepts: the mechanism, which describes why
    data is missing, and the pattern, which describes how missing values are
    arranged over time.
    
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, D) or (N, T, D)
    mechanism : str
        Missingness mechanism. Supported values are "mcar" (Missing Completely
        At Random), "mar" (Missing At Random, dependent on other variables),
        and "mnar" (Missing Not At Random, dependent on the value itself).
    missing_rate : float
        Target fraction of missing values (0.0 to 1.0)
        Applied to eligible (non-NaN) entries
    seed : int, optional
        Random seed for reproducibility
    pattern : str, optional
        Missingness pattern. Supported values are "pointwise" (scattered
        individual points), "block" (contiguous segments), "monotone" (once
        missing, stays missing), "decay" (missingness increases over time),
        "markov" (temporally dependent flickering), and "gilbert_elliott"
        (bursty two-state loss with leaky good/bad periods; MCAR only).
        Aliases include "point"/"scattered" for pointwise, "contiguous" for
        block, "dropout" for monotone, "degradation" for decay, "flickering"
        for markov, and "gilbert-elliott"/"gilbert"/"burst" for
        gilbert_elliott.
    **kwargs : dict
        Mechanism- and pattern-specific parameters. Common options include
        target, driver_dims, driver_weights, strength, base_rate, direction,
        mnar_mode, block_len, block_frac, block_density, decay_rate,
        decay_center, persist, bad_loss, and good_loss. See the user
        documentation for pattern-specific details and constraints.
    
    Returns
    -------
    X_missing : np.ndarray
        Data with NaNs inserted (same shape as X)
    mask : np.ndarray
        Boolean mask (True=observed, False=missing)
    
    Examples
    --------
    >>> # MCAR with point-wise pattern (default)
    >>> X_missing, mask = simulate_missingness(X, "mcar", 0.15, seed=42)
    
    >>> # MAR with block pattern (sensor dropout depends on activity)
    >>> X_missing, mask = simulate_missingness(
    ...     X, "mar", 0.25, seed=42,
    ...     driver_dims=[0], pattern="block", block_len=10
    ... )

    >>> # Block length can also scale with the time axis
    >>> X_missing, mask = simulate_missingness(
    ...     X, "mcar", 0.20, seed=42,
    ...     pattern="block", block_frac=0.02
    ... )
    
    >>> # MNAR with block pattern (extreme values cause sensor failure)
    >>> X_missing, mask = simulate_missingness(
    ...     X, "mnar", 0.20, seed=42,
    ...     mnar_mode="extreme", pattern="block"
    ... )
    """
    # Create RNG for reproducibility
    rng = np.random.default_rng(seed)
    
    # Validate inputs
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim not in [2, 3]:
        raise ValueError("X must be 2D (T, D) or 3D (N, T, D)")
    
    # Clip missing_rate to valid range (allow out-of-range for convenience)
    missing_rate = float(np.clip(missing_rate, 0.0, 1.0))
    
    # Validate mechanism
    mechanism = mechanism.lower()
    if mechanism not in MECHANISMS:
        raise ValueError(
            f"Unknown mechanism: {mechanism}. "
            f"Must be one of: {list(MECHANISMS.keys())}"
        )
    
    # Validate pattern
    pattern = pattern.lower()
    
    # Backward compatibility: handle old block=True API
    if kwargs.get("block", False):
        pattern = "block"
    
    if pattern not in PATTERNS:
        raise ValueError(
            f"Unknown pattern: {pattern}. "
            f"Must be one of: {list(PATTERNS.keys())}"
        )

    if (
        pattern in {"gilbert_elliott", "gilbert-elliott", "gilbert", "burst"}
        and mechanism != "mcar"
    ):
        raise ValueError(
            "gilbert_elliott pattern currently supports only mechanism='mcar'. "
            "Use pattern='markov' for a simpler burst pattern that composes with "
            "MAR or MNAR."
        )
    
    # Copy input
    X_missing = X.copy()
    
    # Identify existing NaNs
    existing_nans = np.isnan(X)
    eligible_mask = _get_eligible_mask(
        X, existing_nans, target=kwargs.get("target", "all")
    )
    
    # Step 1: Generate mechanism-specific mask (WHY missing)
    mask = MECHANISMS[mechanism](X, missing_rate, existing_nans, rng=rng, **kwargs)
    
    # Step 2: Apply pattern (HOW missing)
    mask = PATTERNS[pattern](
        mask,
        X.shape,
        rng=rng,
        eligible_mask=eligible_mask,
        forced_missing=existing_nans,
        **kwargs
    )
    
    # Apply mask
    X_missing[~mask] = np.nan
    
    return X_missing, mask


def simulate_many_rates(
    X: np.ndarray,
    mechanism: str,
    rates: list[float],
    seed: int | None = None,
    **kwargs
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
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
        seed: int | None = None,
        **config
    ):
        self.mechanism = mechanism
        self.missing_rate = missing_rate
        self.seed = seed
        self.config = config
    
    def generate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
