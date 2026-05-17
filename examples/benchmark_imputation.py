"""Benchmark simple imputation baselines under TSGap missingness scenarios.

This script is intentionally dependency-light: it uses only NumPy and TSGap so
it can run anywhere the package itself runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tsgap import simulate_missingness


def make_synthetic_series(
    n_timesteps: int = 600,
    n_features: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Create a smooth multivariate time series with trend and seasonality."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_timesteps)
    X = np.zeros((n_timesteps, n_features), dtype=float)

    for d in range(n_features):
        frequency = d + 1
        phase = d * np.pi / 8
        seasonal = np.sin(2 * np.pi * frequency * t + phase)
        slow_cycle = 0.5 * np.cos(2 * np.pi * (frequency / 2) * t)
        trend = (d + 1) * 0.25 * t
        noise = rng.normal(0, 0.08 + d * 0.01, size=n_timesteps)
        X[:, d] = seasonal + slow_cycle + trend + noise

    return X


def impute_mean(X_missing: np.ndarray) -> np.ndarray:
    """Fill each feature with its observed mean."""
    X_imputed = X_missing.copy()
    for d in range(X_imputed.shape[1]):
        fill = np.nanmean(X_imputed[:, d])
        X_imputed[np.isnan(X_imputed[:, d]), d] = fill
    return X_imputed


def impute_forward_fill(X_missing: np.ndarray) -> np.ndarray:
    """Forward-fill each feature, using the feature mean for leading NaNs."""
    X_imputed = X_missing.copy()
    for d in range(X_imputed.shape[1]):
        col = X_imputed[:, d]
        fill = np.nanmean(col)
        last = fill
        for i, value in enumerate(col):
            if np.isnan(value):
                col[i] = last
            else:
                last = value
        X_imputed[:, d] = col
    return X_imputed


def impute_linear(X_missing: np.ndarray) -> np.ndarray:
    """Linearly interpolate each feature, with edge values held constant."""
    X_imputed = X_missing.copy()
    x_axis = np.arange(X_imputed.shape[0])

    for d in range(X_imputed.shape[1]):
        col = X_imputed[:, d]
        observed = ~np.isnan(col)
        if not observed.any():
            col[:] = 0.0
        elif observed.sum() == 1:
            col[:] = col[observed][0]
        else:
            col[~observed] = np.interp(
                x_axis[~observed], x_axis[observed], col[observed]
            )
        X_imputed[:, d] = col

    return X_imputed


def score_imputation(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute metrics only on artificially missing entries."""
    missing_idx = ~mask
    errors = X_true[missing_idx] - X_imputed[missing_idx]
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def run_benchmark(seed: int = 42, missing_rate: float = 0.2) -> list[dict[str, object]]:
    """Run imputation baselines across representative missingness scenarios."""
    X = make_synthetic_series(seed=seed)

    scenarios = [
        ("mcar_pointwise", "mcar", "pointwise", {}),
        ("mcar_block", "mcar", "block", {"block_len": 16, "block_density": 0.8}),
        ("mar_block", "mar", "block", {"driver_dims": [0], "block_len": 16}),
        ("mnar_monotone", "mnar", "monotone", {"mnar_mode": "extreme"}),
        ("mcar_decay", "mcar", "decay", {"decay_rate": 5.0, "decay_center": 0.6}),
        ("mcar_markov", "mcar", "markov", {"persist": 0.8}),
    ]

    imputers = [
        ("mean", impute_mean),
        ("forward_fill", impute_forward_fill),
        ("linear", impute_linear),
    ]

    rows: list[dict[str, object]] = []
    for scenario_name, mechanism, pattern, kwargs in scenarios:
        X_missing, mask = simulate_missingness(
            X,
            mechanism,
            missing_rate,
            seed=seed,
            pattern=pattern,
            **kwargs,
        )

        for imputer_name, imputer in imputers:
            X_imputed = imputer(X_missing)
            metrics = score_imputation(X, X_imputed, mask)
            rows.append(
                {
                    "scenario": scenario_name,
                    "mechanism": mechanism,
                    "pattern": pattern,
                    "imputer": imputer_name,
                    "target_rate": missing_rate,
                    "actual_rate": float((~mask).mean()),
                    **metrics,
                }
            )

    return rows


def print_results(rows: list[dict[str, object]]) -> None:
    """Print a compact benchmark table."""
    headers = ["scenario", "imputer", "actual_rate", "rmse", "mae"]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "scenario": str(row["scenario"]),
                "imputer": str(row["imputer"]),
                "actual_rate": f"{row['actual_rate']:.3f}",
                "rmse": f"{row['rmse']:.4f}",
                "mae": f"{row['mae']:.4f}",
            }
        )

    widths = {header: len(header) for header in headers}
    for row in formatted_rows:
        for header, value in row.items():
            widths[header] = max(widths[header], len(value))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple imputation baselines under TSGap scenarios."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_benchmark(seed=args.seed, missing_rate=args.missing_rate)
    print_results(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nWrote results to {args.csv}")


if __name__ == "__main__":
    main()
