"""Example usage of ts_missingness library."""

import numpy as np
from ts_missingness import simulate_missingness, simulate_many_rates, MissingnessSimulator

# Generate sample time-series data
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 timesteps, 5 dimensions

print("=" * 60)
print("Time-Series Missingness Simulation Examples")
print("=" * 60)

# Example 1: MCAR - 15% missing completely at random
print("\n1. MCAR: 15% missing completely at random")
X_missing, mask = simulate_missingness(X, "mcar", missing_rate=0.15, seed=42)
actual_rate = (~mask).sum() / mask.size
print(f"   Target rate: 0.15, Actual rate: {actual_rate:.4f}")
print(f"   Missing values: {(~mask).sum()}/{mask.size}")

# Example 2: MAR - 25% missing depending on dimension 0
print("\n2. MAR: 25% missing depending on dimension 0")
X_missing, mask = simulate_missingness(
    X, "mar", missing_rate=0.25, seed=42,
    driver_dims=[0], strength=2.0
)
actual_rate = (~mask).sum() / mask.size
print(f"   Target rate: 0.25, Actual rate: {actual_rate:.4f}")
print(f"   Missingness depends on values in dimension 0")

# Example 3: MNAR - 10% extreme values missing
print("\n3. MNAR: 10% extreme values missing")
X_missing, mask = simulate_missingness(
    X, "mnar", missing_rate=0.10, seed=42,
    mnar_mode="extreme", strength=2.0
)
actual_rate = (~mask).sum() / mask.size
print(f"   Target rate: 0.10, Actual rate: {actual_rate:.4f}")
print(f"   Extreme values more likely to be missing")

# Example 4: Block missingness - 20% with 60-timestep blocks
print("\n4. Block missingness: 20% in contiguous segments")
X_missing, mask = simulate_missingness(
    X, "mcar", missing_rate=0.20, seed=42,
    block=True, block_len=10, block_density=0.7
)
actual_rate = (~mask).sum() / mask.size
print(f"   Target rate: 0.20, Actual rate: {actual_rate:.4f}")
print(f"   70% of missingness in blocks of length 10")

# Example 5: Multiple rates
print("\n5. Multiple missing rates")
rates = [0.05, 0.15, 0.25]
results = simulate_many_rates(X, "mcar", rates, seed=42)
for rate, (X_miss, m) in results.items():
    actual = (~m).sum() / m.size
    print(f"   Rate {rate}: actual {actual:.4f}")

# Example 6: Object-oriented interface
print("\n6. Object-oriented interface")
sim = MissingnessSimulator("mcar", missing_rate=0.15, seed=42)
X_missing, mask = sim.generate(X)
actual_rate = (~mask).sum() / mask.size
print(f"   Simulator configured with rate 0.15")
print(f"   Actual rate: {actual_rate:.4f}")

# Example 7: Evaluation workflow
print("\n7. Evaluation workflow for imputation")
X_ground_truth = X.copy()
X_missing, mask = simulate_missingness(X, "mcar", 0.20, seed=42)

# Simulate a simple imputation (mean imputation)
X_imputed = X_missing.copy()
for d in range(X.shape[1]):
    col_mean = np.nanmean(X_missing[:, d])
    X_imputed[np.isnan(X_imputed[:, d]), d] = col_mean

# Compute metrics only on masked entries
missing_indices = ~mask
rmse = np.sqrt(np.mean((X_ground_truth[missing_indices] - X_imputed[missing_indices])**2))
mae = np.mean(np.abs(X_ground_truth[missing_indices] - X_imputed[missing_indices]))

print(f"   RMSE on missing values: {rmse:.4f}")
print(f"   MAE on missing values: {mae:.4f}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
