"""Visualizations for Derivatives & Pricing Models (Projects 16-20).

Styled with a dark aesthetic, amber/gold typography, and crisp statistical typography.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

DARK_THEME_STYLE = {
    "figure.facecolor": "#0a0b0e",
    "axes.facecolor": "#12141c",
    "axes.edgecolor": "#262a38",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#1e2230",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
}

COLOR_PALETTE = {
    "bg_main": "#0a0b0e",
    "bg_card": "#12141c",
    "border": "#262a38",
    "gold": "#f59e0b",
    "gold_bright": "#fbbf24",
    "gold_dark": "#b45309",
    "text_light": "#f8fafc",
    "text_dim": "#94a3b8",
    "text_muted": "#64748b",
    "green": "#10b981",
    "red": "#ef4444",
    "blue": "#38bdf8",
    "purple": "#a855f7",
}


def apply_dark_theme():
    """Applies global dark theme."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_black_scholes_card(
    chain_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Black-Scholes Formula card with Calls/Puts Option Chain (Card 16)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.92, "16 | BLACK-SCHOLES OPTION PRICING ENGINE", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold")

    # Formula box
    formula_text = (
        r"$C = S_0 N(d_1) - K e^{-rT} N(d_2)$" "\n"
        r"$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$"
    )
    rect = patches.FancyBboxPatch(
        (0.04, 0.52), 0.92, 0.34,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#181b26",
        edgecolor=COLOR_PALETTE["border"],
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(0.50, 0.69, formula_text, color=COLOR_PALETTE["gold_bright"], fontsize=11, ha="center", va="center", transform=ax.transAxes)

    # Mini Option Chain Table
    headers = ["Strike", "Call Price", "Put Price", "Call IV", "Delta (C)"]
    x_coords = [0.10, 0.30, 0.50, 0.70, 0.90]
    y_pos = 0.42

    for col_name, x in zip(headers, x_coords):
        ax.text(x, y_pos, col_name, color=COLOR_PALETTE["text_dim"], fontsize=8, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.plot([0.04, 0.96], [y_pos - 0.03, y_pos - 0.03], color=COLOR_PALETTE["border"], lw=0.8, transform=ax.transAxes)

    sample_rows = [
        ("100", "5.80", "0.40", "22.1%", "0.62"),
        ("105", "3.45", "0.75", "21.3%", "0.48"),
        ("110", "1.90", "1.25", "20.4%", "0.34"),
        ("115", "0.95", "2.10", "19.6%", "0.21"),
        ("120", "0.40", "3.40", "18.7%", "0.11"),
    ]

    for i, row in enumerate(sample_rows):
        y = y_pos - 0.08 - i * 0.065
        for val, x in zip(row, x_coords):
            c = COLOR_PALETTE["gold_bright"] if x == 0.10 else COLOR_PALETTE["text_light"]
            ax.text(x, y, val, color=c, fontsize=8, ha="center", transform=ax.transAxes)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_volatility_smile(
    moneyness: np.ndarray,
    implied_vols: np.ndarray,
    fitted_svi: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Implied Volatility Smile / Skew curve (Card 17)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    iv_pct = implied_vols * 100

    # Market points
    ax.scatter(moneyness, iv_pct, color=COLOR_PALETTE["gold_bright"], s=45, zorder=5, label="Market Quotes")

    # Curve interpolation
    if fitted_svi is not None:
        ax.plot(moneyness, fitted_svi * 100, color=COLOR_PALETTE["gold"], lw=2.0, label="Parametric SVI Fit")
    else:
        ax.plot(moneyness, iv_pct, color=COLOR_PALETTE["gold"], lw=2.0, linestyle="-")

    ax.set_title("17 | IMPLIED VOLATILITY SMILE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Strike / Spot (Moneyness K/S)", fontsize=8.5, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Implied Volatility (%)", fontsize=8.5, color=COLOR_PALETTE["text_dim"])
    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(min(10, min(iv_pct) - 3), max(45, max(iv_pct) + 5))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_option_greeks_card(
    greeks_dict: Dict[str, float],
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Option Greeks Card (Card 18)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.92, "18 | OPTION GREEKS CALCULATOR", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold")

    greeks = [
        ("Δ", "Delta", "Price sensitivity\nto spot (∂V/∂S)", f"{greeks_dict.get('delta', 0.5234):+.3f}", 0.12),
        ("Γ", "Gamma", "Change in delta\ncurvature (∂²V/∂S²)", f"{greeks_dict.get('gamma', 0.0187):.4f}", 0.31),
        ("Θ", "Theta", "Time decay\nper day (∂V/∂t)", f"{greeks_dict.get('theta_daily', -0.0412):.3f}", 0.50),
        ("ν", "Vega", "Sensitivity to\n1% vol (∂V/∂σ)", f"{greeks_dict.get('vega_pct', 0.1985):.3f}", 0.69),
        ("ρ", "Rho", "Sensitivity to\n1% rate (∂V/∂r)", f"{greeks_dict.get('rho_pct', 0.0481):.3f}", 0.88),
    ]

    for sym, name, desc, val_str, x in greeks:
        # Box
        rect = patches.FancyBboxPatch(
            (x - 0.08, 0.12), 0.16, 0.70,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#181b26",
            edgecolor=COLOR_PALETTE["border"],
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        ax.text(x, 0.74, sym, color=COLOR_PALETTE["gold_bright"], fontsize=18, fontweight="bold", ha="center", transform=ax.transAxes)
        ax.text(x, 0.60, name, color=COLOR_PALETTE["text_light"], fontsize=9, fontweight="bold", ha="center", transform=ax.transAxes)
        ax.text(x, 0.44, desc, color=COLOR_PALETTE["text_dim"], fontsize=6.5, ha="center", linespacing=1.2, transform=ax.transAxes)
        ax.text(x, 0.22, val_str, color=COLOR_PALETTE["gold_bright"], fontsize=9.5, fontweight="bold", ha="center", transform=ax.transAxes)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_binomial_tree_diagram(
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots 3-step Binomial Tree Lattice diagram (Card 19)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.92, "19 | BINOMIAL TREE (3-STEPS EXAMPLE)", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold")

    # Node positions (x, y)
    nodes = {
        (0, 0): (0.10, 0.50, "$S_0$"),
        (1, 1): (0.35, 0.68, "$S_u$"),
        (1, 0): (0.35, 0.32, "$S_d$"),
        (2, 2): (0.62, 0.82, "$Su^2$"),
        (2, 1): (0.62, 0.50, "$S$"),
        (2, 0): (0.62, 0.18, "$Sd^2$"),
        (3, 3): (0.88, 0.90, "$Su^3$"),
        (3, 2): (0.88, 0.64, "$Su$"),
        (3, 1): (0.88, 0.36, "$Sd$"),
        (3, 0): (0.88, 0.10, "$Sd^3$"),
    }

    # Edges
    edges = [
        ((0, 0), (1, 1)), ((0, 0), (1, 0)),
        ((1, 1), (2, 2)), ((1, 1), (2, 1)),
        ((1, 0), (2, 1)), ((1, 0), (2, 0)),
        ((2, 2), (3, 3)), ((2, 2), (3, 2)),
        ((2, 1), (3, 2)), ((2, 1), (3, 1)),
        ((2, 0), (3, 1)), ((2, 0), (3, 0)),
    ]

    for (p1, p2) in edges:
        x1, y1, _ = nodes[p1]
        x2, y2, _ = nodes[p2]
        ax.plot([x1, x2], [y1, y2], color=COLOR_PALETTE["gold_dark"], lw=1.2, alpha=0.8, transform=ax.transAxes)

    for (step, j), (x, y, label) in nodes.items():
        circle = patches.Circle((x, y), 0.038, facecolor="#1a1d29", edgecolor=COLOR_PALETTE["gold_bright"], lw=1.2, transform=ax.transAxes, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, label, color=COLOR_PALETTE["text_light"], fontsize=8, fontweight="bold", ha="center", va="center", transform=ax.transAxes, zorder=5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_monte_carlo_option_paths(
    paths: np.ndarray,
    strike: float = 100.0,
    time_grid: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Monte Carlo Option Pricing Path Simulation (Card 20)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    n_paths, n_steps = paths.shape
    if time_grid is None:
        time_grid = np.linspace(0, 1.0, n_steps)

    # Plot sample trajectories (e.g. 50 paths in varying dark-neon colors)
    colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c", "#4ade80"]
    sample_idx = np.random.choice(n_paths, size=min(50, n_paths), replace=False)

    for i, idx in enumerate(sample_idx):
        c = colors[i % len(colors)]
        ax.plot(time_grid, paths[idx], color=c, alpha=0.35, lw=0.9)

    ax.axhline(strike, color=COLOR_PALETTE["gold_bright"], linestyle="--", lw=1.2, label=f"Strike ($K={strike:.0f}$)")

    ax.set_title("20 | MONTE CARLO SIMULATION", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Time (Years)", fontsize=8.5, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Spot Price ($)", fontsize=8.5, color=COLOR_PALETTE["text_dim"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(40, 170)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_pricing_workflow_pipeline(ax: plt.Axes):
    """Renders model comparison banner matching bottom footer."""
    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.85, "THEN COMPARE YOUR MODELS AGAINST:", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", transform=ax.transAxes)

    steps = ["MARKET\nPRICES", "PRICING\nERROR", "OPTION\nGREEKS", "VOLATILITY\nASSUMPTIONS"]
    x_positions = [0.12, 0.36, 0.60, 0.84]

    for i, (name, x) in enumerate(zip(steps, x_positions)):
        rect = patches.FancyBboxPatch(
            (x - 0.08, 0.20), 0.16, 0.48,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#181b26",
            edgecolor=COLOR_PALETTE["gold"],
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x, 0.44, name, color=COLOR_PALETTE["gold_bright"], fontsize=7.5, fontweight="bold", ha="center", va="center", linespacing=1.2, transform=ax.transAxes)

        if i < len(steps) - 1:
            next_x = x_positions[i+1]
            ax.annotate("", xy=(next_x - 0.09, 0.44), xytext=(x + 0.09, 0.44),
                        arrowprops=dict(arrowstyle="->", color=COLOR_PALETTE["gold"], lw=1.5),
                        xycoords="axes fraction")


def plot_master_pricing_infographic(
    chain_df: pd.DataFrame,
    moneyness: np.ndarray,
    implied_vols: np.ndarray,
    greeks_dict: Dict[str, float],
    mc_paths: np.ndarray,
    output_path: str = "/working_dir/pricing_models/output/pricing_models_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module pricing infographic matching Screenshot 5/7."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(7, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "DERIVATIVES & PRICING", color=COLOR_PALETTE["gold_bright"], fontsize=24, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "16-20 — BUILD PRICING MODELS", color=COLOR_PALETTE["gold"], fontsize=14, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "Black-Scholes, Implied Volatility Solvers, Greeks, Binomial Trees & Monte Carlo Engines", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Card 16 (Black-Scholes)
    ax_16 = fig.add_subplot(gs[1, 0])
    plot_black_scholes_card(chain_df=chain_df, ax=ax_16)
    ax_16_txt = fig.add_subplot(gs[1, 1])
    ax_16_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_16_txt.axis("off")
    ax_16_txt.text(0.05, 0.75, "16 | Black-Scholes Option Pricing Engine", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_16_txt.text(0.05, 0.45, "• Analytical closed-form solution for European Call & Put.\n• Merton continuous dividend yield extension ($q$).\n• Exact Put-Call Parity validation: $C - P = S_0 e^{-qT} - K e^{-rT}$.\n• High-throughput option chain evaluation.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Card 17 (Implied Volatility Solver)
    ax_17 = fig.add_subplot(gs[2, 0])
    plot_volatility_smile(moneyness=moneyness, implied_vols=implied_vols, ax=ax_17)
    ax_17_txt = fig.add_subplot(gs[2, 1])
    ax_17_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_17_txt.axis("off")
    ax_17_txt.text(0.05, 0.75, "17 | Implied Volatility Solver & Smile", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_17_txt.text(0.05, 0.45, "• Fast Newton-Raphson root-finder with Vega derivative.\n• Robust Brent / Bisection fallback for deep OTM strikes.\n• Parametric Gatheral SVI calibration & Cubic Spline fitting.\n• 'Find the volatility that makes your model match market price.'", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Card 18 (Option Greeks Calculator)
    ax_18 = fig.add_subplot(gs[3, 0])
    plot_option_greeks_card(greeks_dict=greeks_dict, ax=ax_18)
    ax_18_txt = fig.add_subplot(gs[3, 1])
    ax_18_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_18_txt.axis("off")
    ax_18_txt.text(0.05, 0.75, "18 | Option Greeks Calculator (Δ, Γ, Θ, ν, ρ)", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_18_txt.text(0.05, 0.45, "• First-Order: Delta (spot), Vega (vol), Theta (time), Rho (rate).\n• Second-Order: Gamma (curvature), Vanna, Volga/Vomma.\n• Third-Order: Speed, Color, Zomma, Charm.\n• Verified against central finite-difference numerical engine.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Card 19 (Binomial Option Pricing Model)
    ax_19 = fig.add_subplot(gs[4, 0])
    plot_binomial_tree_diagram(ax=ax_19)
    ax_19_txt = fig.add_subplot(gs[4, 1])
    ax_19_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_19_txt.axis("off")
    ax_19_txt.text(0.05, 0.75, "19 | Binomial Option Pricing Model", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_19_txt.text(0.05, 0.45, "• Cox-Ross-Rubinstein (CRR), Jarrow-Rudd & Leisen-Reimer lattices.\n• Backward induction dynamic programming algorithm.\n• American Early Exercise Premium: $V_{\\text{American}} - V_{\\text{European}}$.\n• Discrete lattice Greeks extracted directly from tree nodes.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Card 20 (Monte Carlo Option Pricing Engine)
    ax_20 = fig.add_subplot(gs[5, 0])
    plot_monte_carlo_option_paths(paths=mc_paths, strike=100.0, ax=ax_20)
    ax_20_txt = fig.add_subplot(gs[5, 1])
    ax_20_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_20_txt.axis("off")
    ax_20_txt.text(0.05, 0.75, "20 | Monte Carlo Option Pricing Engine", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_20_txt.text(0.05, 0.45, "• Risk-neutral path simulation ($N=100,000$ paths).\n• Variance reduction: Antithetic & Control Variates (>95% error cut).\n• Path-dependent exotics: Asian, Barrier, Lookback derivatives.\n• American option pricing via Longstaff-Schwartz (LSM).", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 7. Comparison Workflow Banner
    ax_pipe = fig.add_subplot(gs[6, :])
    plot_pricing_workflow_pipeline(ax_pipe)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_volatility_surface_3d(
    moneyness_grid: np.ndarray,
    expiry_grid: np.ndarray,
    iv_grid: np.ndarray,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Renders 3D Implied Volatility Surface matching the laptop graphic."""
    apply_dark_theme()
    fig = plt.figure(figsize=(9, 6), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(COLOR_PALETTE["bg_main"])

    M, T = np.meshgrid(moneyness_grid, expiry_grid) if moneyness_grid.ndim == 1 else (moneyness_grid, expiry_grid)

    surf = ax.plot_surface(
        M, T, iv_grid * 100,
        cmap="plasma",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )

    ax.set_title("3D IMPLIED VOLATILITY SURFACE", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Moneyness (K/S)", fontsize=9, color=COLOR_PALETTE["text_dim"], labelpad=8)
    ax.set_ylabel("Expiry T (Years)", fontsize=9, color=COLOR_PALETTE["text_dim"], labelpad=8)
    ax.set_zlabel("Implied Volatility (%)", fontsize=9, color=COLOR_PALETTE["text_dim"], labelpad=8)
    ax.tick_params(axis="both", labelsize=8)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label("Implied Vol (%)", color=COLOR_PALETTE["text_dim"], fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
