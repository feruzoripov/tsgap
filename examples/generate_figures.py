"""Generate mechanism × pattern visualization grid for README and paper."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from tsgap import simulate_missingness


def generate_grid():
    """Generate the 3×5 mechanism × pattern heatmap grid."""
    rng = np.random.default_rng(42)
    T, D = 200, 8
    X = rng.standard_normal((T, D))

    # Add structure to make MAR/MNAR visually interesting
    # Dim 0: ramp up over time (good MAR driver)
    X[:, 0] = np.linspace(-2, 2, T) + rng.standard_normal(T) * 0.3
    # Dim 1: some extreme values
    X[50:60, 1] = 4.0
    X[140:150, 1] = -4.0

    mechanisms = [
        ("MCAR", {"mechanism": "mcar"}),
        ("MAR", {"mechanism": "mar", "driver_dims": [0], "strength": 2.5}),
        ("MNAR", {"mechanism": "mnar", "mnar_mode": "extreme", "strength": 2.5}),
    ]

    patterns = [
        ("Pointwise", {"pattern": "pointwise"}),
        ("Block", {"pattern": "block", "block_len": 15}),
        ("Monotone", {"pattern": "monotone"}),
        ("Decay", {"pattern": "decay", "decay_rate": 6.0, "decay_center": 0.6}),
        ("Markov", {"pattern": "markov", "persist": 0.85}),
        ("Gilbert-Elliott", {
            "pattern": "gilbert_elliott", "persist": 0.9,
            "bad_loss": 0.8, "good_loss": 0.02,
        }),
    ]

    n_mechs = len(mechanisms)
    n_patts = len(patterns)

    # Custom colormap: white = observed, colored = missing
    cmap = mcolors.ListedColormap(["#ffffff", "#2563eb"])

    fig, axes = plt.subplots(
        n_mechs, n_patts,
        figsize=(n_patts * 2.6, n_mechs * 2.2),
        constrained_layout=True,
    )

    for i, (mech_name, mech_kw) in enumerate(mechanisms):
        for j, (patt_name, patt_kw) in enumerate(patterns):
            ax = axes[i, j]

            kw = {**mech_kw, **patt_kw, "seed": 42}
            _, mask = simulate_missingness(X, missing_rate=0.20, **kw)

            # Show mask: 1 = missing (colored), 0 = observed (white)
            missing_img = (~mask).astype(int)

            ax.imshow(
                missing_img,
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

            # Rate annotation
            rate = (~mask).sum() / mask.size
            ax.text(
                0.98, 0.02, f"{rate:.0%}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7,
                color="#64748b",
                fontfamily="monospace",
            )

            # Column titles (top row only)
            if i == 0:
                ax.set_title(patt_name, fontsize=11, fontweight="600", pad=6)

            # Row labels (left column only)
            if j == 0:
                ax.set_ylabel(mech_name, fontsize=11, fontweight="600", labelpad=8)

            # Axis labels on edges only
            if i == n_mechs - 1:
                ax.set_xlabel("Features", fontsize=8, color="#64748b")

            ax.tick_params(
                axis="both", which="both",
                labelsize=6, colors="#94a3b8",
                length=2,
            )

    # Save
    out_dir = Path(__file__).parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)

    fig.savefig(
        out_dir / "mechanism_pattern_grid.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved: {out_dir / 'mechanism_pattern_grid.png'}")

    # Also generate a single "before/after" comparison
    generate_before_after(X, out_dir)


def generate_before_after(X, out_dir):
    """Generate before/after heatmaps showing complete data vs. masked data.

    Shows the actual data values as a heatmap, with missing positions
    clearly marked in red. This makes the missingness pattern immediately
    visible against the data.
    """
    examples = [
        ("Complete Data", None),
        ("MCAR + Pointwise", {"mechanism": "mcar", "pattern": "pointwise"}),
        ("MAR + Block", {
            "mechanism": "mar", "pattern": "block",
            "driver_dims": [0], "block_len": 15,
        }),
        ("MNAR + Monotone", {
            "mechanism": "mnar", "pattern": "monotone",
            "mnar_mode": "extreme", "strength": 2.5,
        }),
        ("MCAR + Decay", {
            "mechanism": "mcar", "pattern": "decay",
            "decay_rate": 6.0, "decay_center": 0.6,
        }),
        ("MAR + Markov", {
            "mechanism": "mar", "pattern": "markov",
            "driver_dims": [0], "persist": 0.85,
        }),
        ("MCAR + Gilbert-Elliott", {
            "mechanism": "mcar", "pattern": "gilbert_elliott",
            "persist": 0.9, "bad_loss": 0.8, "good_loss": 0.02,
        }),
    ]

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 3.5), constrained_layout=True)

    # Data colormap: cool blues/purples so red missing overlay is unambiguous
    data_cmap = "YlGnBu"

    for idx, (title, kw) in enumerate(examples):
        ax = axes[idx]

        if kw is None:
            # Complete data — show as heatmap, no missing
            ax.imshow(
                X, aspect="auto", cmap=data_cmap, interpolation="nearest",
                vmin=-3, vmax=3,
            )
            ax.set_title(title, fontsize=9, fontweight="600", pad=6)
        else:
            _, mask = simulate_missingness(X, missing_rate=0.20, seed=42, **kw)

            # Start with data heatmap
            ax.imshow(
                X, aspect="auto", cmap=data_cmap, interpolation="nearest",
                vmin=-3, vmax=3,
            )

            # Overlay missing positions in solid red
            missing_overlay = np.full((*X.shape, 4), 0.0)  # RGBA
            missing_positions = ~mask
            missing_overlay[missing_positions] = [0.91, 0.20, 0.20, 0.90]  # red

            ax.imshow(
                missing_overlay, aspect="auto", interpolation="nearest",
            )

            rate = (~mask).sum() / mask.size
            ax.set_title(f"{title}\n({rate:.0%} missing)", fontsize=9, fontweight="600", pad=6)

        if idx == 0:
            ax.set_ylabel("Time →", fontsize=8, color="#64748b")
        ax.set_xlabel("Features", fontsize=8, color="#64748b")
        ax.tick_params(axis="both", labelsize=6, colors="#94a3b8", length=2)

    fig.savefig(
        out_dir / "before_after.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved: {out_dir / 'before_after.png'}")


def generate_sigmoid(out_dir):
    """Generate a 4-panel figure showing how the sigmoid is used in each context."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    x = np.linspace(-6, 6, 300)

    # --- Panel 1: Basic sigmoid ---
    ax = axes[0]
    sigmoid = 1 / (1 + np.exp(-x))
    ax.plot(x, sigmoid, color="#2563eb", linewidth=2)
    ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.plot(0, 0.5, "o", color="#2563eb", markersize=6, zorder=5)
    ax.annotate("(0, 0.5)", xy=(0, 0.5), xytext=(1.2, 0.38),
                fontsize=8, color="#475569")
    ax.set_title("Sigmoid Function σ(x)", fontsize=10, fontweight="600")
    ax.set_xlabel("x (input)", fontsize=8, color="#64748b")
    ax.set_ylabel("σ(x) (probability)", fontsize=8, color="#64748b")
    ax.set_ylim(-0.05, 1.05)
    ax.text(-5.5, 0.92, "→ 1 as x → +∞", fontsize=7, color="#64748b")
    ax.text(-5.5, 0.06, "→ 0 as x → −∞", fontsize=7, color="#64748b")

    # --- Panel 2: MAR — strength effect ---
    ax = axes[1]
    z = np.linspace(-3, 3, 300)
    beta = 0
    for alpha, color, label in [
        (0.5, "#93c5fd", "α = 0.5 (weak)"),
        (2.0, "#2563eb", "α = 2.0 (default)"),
        (5.0, "#1e3a5f", "α = 5.0 (strong)"),
    ]:
        prob = 1 / (1 + np.exp(-(alpha * z + beta)))
        ax.plot(z, prob, color=color, linewidth=2, label=label)
    ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_title("MAR: Strength Controls Slope", fontsize=10, fontweight="600")
    ax.set_xlabel("z (normalized driver signal)", fontsize=8, color="#64748b")
    ax.set_ylabel("P(missing)", fontsize=8, color="#64748b")
    ax.set_ylim(-0.05, 1.05)

    # --- Panel 3: MNAR — mode effect ---
    ax = axes[2]
    z = np.linspace(-3, 3, 300)
    alpha = 2.0
    beta = 0
    modes = [
        ("high: f(z) = z", z, "#ef4444"),
        ("low: f(z) = −z", -z, "#3b82f6"),
        ("extreme: f(z) = |z|", np.abs(z), "#8b5cf6"),
    ]
    for label, score, color in modes:
        prob = 1 / (1 + np.exp(-(alpha * score + beta)))
        ax.plot(z, prob, color=color, linewidth=2, label=label)
    ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=7, loc="center right", framealpha=0.9)
    ax.set_title("MNAR: Mode Targets Different Values", fontsize=10, fontweight="600")
    ax.set_xlabel("z (normalized data value)", fontsize=8, color="#64748b")
    ax.set_ylabel("P(missing)", fontsize=8, color="#64748b")
    ax.set_ylim(-0.05, 1.05)

    # --- Panel 4: Decay pattern — ramp over time ---
    ax = axes[3]
    t_norm = np.linspace(0, 1, 300)
    configs = [
        ("γ=3, c=0.7 (default)", 3.0, 0.7, "#2563eb"),
        ("γ=6, c=0.5 (steep, early)", 6.0, 0.5, "#ef4444"),
        ("γ=2, c=0.8 (gentle, late)", 2.0, 0.8, "#10b981"),
    ]
    for label, gamma, c, color in configs:
        w = 1 / (1 + np.exp(-gamma * (t_norm - c)))
        ax.plot(t_norm, w, color=color, linewidth=2, label=label)
    ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("Decay: Sigmoid Ramp Over Time", fontsize=10, fontweight="600")
    ax.set_xlabel("t_norm (normalized time, 0→1)", fontsize=8, color="#64748b")
    ax.set_ylabel("w(t) (sampling weight)", fontsize=8, color="#64748b")
    ax.set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=7, colors="#94a3b8", length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15)

    fig.savefig(
        out_dir / "sigmoid_explained.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved: {out_dir / 'sigmoid_explained.png'}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)
    generate_grid()
    generate_sigmoid(out_dir)
