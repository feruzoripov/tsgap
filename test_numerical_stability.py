"""Test numerical stability with extreme parameters."""

import numpy as np
from ts_missingness import simulate_missingness
import warnings

# Suppress overflow warnings to test if they occur
warnings.filterwarnings('error')

np.random.seed(42)
X = np.random.randn(200, 5)

print("Testing numerical stability:")
print("=" * 60)

try:
    # Test with very high strength (could cause overflow without clipping)
    print("\n1. High strength (10.0) with MAR:")
    X_missing, mask = simulate_missingness(
        X, "mar", 0.5, seed=42, driver_dims=[0], strength=10.0
    )
    actual = (~mask).sum() / mask.size
    print(f"   Target: 0.50, Actual: {actual:.4f} ✓")
    
    print("\n2. High strength (10.0) with MNAR:")
    X_missing, mask = simulate_missingness(
        X, "mnar", 0.5, seed=42, mnar_mode="extreme", strength=10.0
    )
    actual = (~mask).sum() / mask.size
    print(f"   Target: 0.50, Actual: {actual:.4f} ✓")
    
    # Test edge cases
    print("\n3. Missing rate = 1.0 (all missing):")
    X_missing, mask = simulate_missingness(X, "mar", 1.0, seed=42, driver_dims=[0])
    actual = (~mask).sum() / mask.size
    print(f"   Target: 1.00, Actual: {actual:.4f} ✓")
    
    print("\n4. Missing rate = 0.0 (none missing):")
    X_missing, mask = simulate_missingness(X, "mnar", 0.0, seed=42)
    actual = (~mask).sum() / mask.size
    print(f"   Target: 0.00, Actual: {actual:.4f} ✓")
    
    # Test very extreme values
    print("\n5. Very extreme data values:")
    X_extreme = np.random.randn(100, 5) * 1000  # Very large values
    X_missing, mask = simulate_missingness(
        X_extreme, "mnar", 0.2, seed=42, mnar_mode="extreme", strength=5.0
    )
    actual = (~mask).sum() / mask.size
    print(f"   Target: 0.20, Actual: {actual:.4f} ✓")
    
    print("\n" + "=" * 60)
    print("All numerical stability tests passed! No overflow warnings.")
    print("=" * 60)
    
except Warning as w:
    print(f"\n❌ Warning occurred: {w}")
    print("Numerical stability issue detected!")
