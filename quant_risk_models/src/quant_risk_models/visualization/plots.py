"""Visualizations for Quantitative Risk & Portfolio Projects (06-10).

Styled with a dark aesthetic, amber/gold accents, and clean statistical typography.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats

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
    "bull_green": "#10b981",
    "bear_red": "#ef4444",
    "var_95": "#f59e0b",
    "var_99": "#ef4444",
    "cvar_95": "#f97316",
    "hist_bar": "#334155",
    "hist_edge": "#475569",
    "dot_cloud": "#3b82f6",
}


def apply_dark_theme():
    """Applies global dark theme."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_distribution_and_var(
    returns: pd.Series,
    var_95: float = -0.0245,
    var_99: float = -0.0367,
    es_95: float = -0.0315,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots return distribution histogram with 95% / 99% VaR cutoffs and ES matching Card 06/07."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ret_pct = returns * 100
    v95_pct = var_95 * 100 if var_95 < 0 else -var_95 * 100
    v99_pct = var_99 * 100 if var_99 < 0 else -var_99 * 100
    es95_pct = es_95 * 100 if es_95 < 0 else -es_95 * 100

    # Histogram
    n, bins, patches_list = ax.hist(
        ret_pct,
        bins=60,
        density=False,
        color=COLOR_PALETTE["hist_bar"],
        edgecolor=COLOR_PALETTE["hist_edge"],
        alpha=0.85,
        zorder=2,
    )

    # Fitted Density curve
    bin_width = bins[1] - bins[0]
    total_samples = len(ret_pct)
    mu, std = ret_pct.mean(), ret_pct.std()
    x_grid = np.linspace(ret_pct.min(), ret_pct.max(), 300)
    pdf_curve = stats.norm.pdf(x_grid, mu, std) * total_samples * bin_width
    ax.plot(x_grid, pdf_curve, color=COLOR_PALETTE["gold_bright"], lw=2.0, zorder=3, label="Fitted Normal")

    # VaR & ES vertical dashed lines
    ax.axvline(v95_pct, color=COLOR_PALETTE["var_95"], linestyle="--", lw=1.8, zorder=4, label="95% VaR")
    ax.axvline(v99_pct, color=COLOR_PALETTE["var_99"], linestyle="--", lw=1.8, zorder=4, label="99% VaR")

    # Annotate on graph
    max_count = max(n)
    ax.text(v95_pct, max_count * 0.95, "95% VaR", color=COLOR_PALETTE["var_95"], fontsize=8, ha="center", fontweight="bold")
    ax.text(v99_pct, max_count * 0.95, "99% VaR", color=COLOR_PALETTE["var_99"], fontsize=8, ha="center", fontweight="bold")

    # Right side stat box inside plot
    stat_text = (
        f"95% VaR\n"
        f"{v95_pct:.2f}%\n\n"
        f"99% VaR\n"
        f"{v99_pct:.2f}%\n\n"
        f"Expected Shortfall\n(95%)\n"
        f"{es95_pct:.2f}%"
    )
    props = dict(boxstyle="round,pad=0.6", facecolor="#161924", edgecolor=COLOR_PALETTE["border"], alpha=0.9)
    ax.text(0.82, 0.50, stat_text, transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=COLOR_PALETTE["gold_bright"], linespacing=1.4, bbox=props, va="center")

    ax.set_title("DISTRIBUTION OF PORTFOLIO RETURNS", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=12, loc="center")
    ax.set_xlabel("Portfolio Return (%)", color=COLOR_PALETTE["text_dim"], fontsize=9)
    ax.set_ylabel("Frequency", color=COLOR_PALETTE["text_dim"], fontsize=9)
    ax.tick_params(axis="both", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_PALETTE["border"])

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_risk_metrics_card(
    metrics: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Renders structured risk metric table matching Middle Left card."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    rows = [
        ("Annualized Return", f"{metrics.get('annualized_return', 0.1234):.2%}"),
        ("Annualized Volatility", f"{metrics.get('annualized_volatility', 0.1862):.2%}"),
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.66):.2f}"),
        ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0.96):.2f}"),
        ("VaR (95%)", f"{metrics.get('var_95', -0.0245):.2%}"),
        ("VaR (99%)", f"{metrics.get('var_99', -0.0367):.2%}"),
        ("Expected Shortfall (95%)", f"{metrics.get('cvar_95', -0.0315):.2%}"),
        ("Max Drawdown", f"{metrics.get('max_drawdown', -0.1738):.2%}"),
    ]

    ax.text(0.04, 0.94, "RISK METRICS (EXAMPLE PORTFOLIO)", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", transform=ax.transAxes)

    # Header
    ax.text(0.06, 0.84, "Metric", color=COLOR_PALETTE["text_dim"], fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.78, 0.84, "Value", color=COLOR_PALETTE["text_dim"], fontsize=9, fontweight="bold", transform=ax.transAxes, ha="right")
    ax.plot([0.04, 0.94], [0.81, 0.81], color=COLOR_PALETTE["border"], lw=1.2, transform=ax.transAxes)

    # Rows
    y_start = 0.74
    y_step = 0.08
    for i, (metric_name, metric_val) in enumerate(rows):
        y = y_start - i * y_step
        color_val = COLOR_PALETTE["bear_red"] if "-" in metric_val else COLOR_PALETTE["text_light"]
        ax.text(0.06, y, metric_name, color=COLOR_PALETTE["text_dim"], fontsize=8.5, transform=ax.transAxes)
        ax.text(0.78, y, metric_val, color=color_val, fontsize=8.5, fontweight="bold", transform=ax.transAxes, ha="right")
        ax.plot([0.04, 0.94], [y - 0.02, y - 0.02], color=COLOR_PALETTE["border"], lw=0.6, alpha=0.5, transform=ax.transAxes)

    # Box outline
    rect = patches.FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.94,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="none",
        edgecolor=COLOR_PALETTE["border"],
        linewidth=1.2,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_monte_carlo_simulation(
    paths: np.ndarray,
    confidence_level: float = 0.95,
    n_simulations: int = 100000,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Monte Carlo portfolio value trajectories matching Middle Right card."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    n_steps = paths.shape[1]
    time_days = np.arange(n_steps)

    # Plot sample paths (e.g. 60 representative trajectories)
    sample_indices = np.random.choice(paths.shape[0], size=min(60, paths.shape[0]), replace=False)
    for idx in sample_indices:
        ax.plot(time_days, paths[idx], color="#64748b", alpha=0.25, lw=0.8)

    # Compute 95% quantile path across time
    var_quantile_path = np.percentile(paths, (1.0 - confidence_level) * 100, axis=0)
    terminal_var_value = var_quantile_path[-1]

    # Highlight 95% VaR path
    ax.plot(time_days, var_quantile_path, color=COLOR_PALETTE["bear_red"], lw=2.2, label=f"VaR ({int(confidence_level*100)}%) Path")

    ax.set_title(f"MONTE CARLO VaR ({int(confidence_level*100)}%)\nSimulations: {n_simulations:,}",
                 color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", pad=8, loc="center")
    ax.set_xlabel("Time (Days)", color=COLOR_PALETTE["text_dim"], fontsize=8.5)
    ax.set_ylabel("Portfolio Value", color=COLOR_PALETTE["text_dim"], fontsize=8.5)

    # Annotate terminal value
    ax.text(time_days[-1], terminal_var_value - 0.08, f"VaR ({int(confidence_level*100)}%)\n{terminal_var_value:.3f}",
            color=COLOR_PALETTE["bear_red"], fontsize=8, fontweight="bold", ha="right")

    ax.set_ylim(0.0, max(1.5, np.percentile(paths[:, -1], 95) + 0.2))
    ax.tick_params(axis="both", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_PALETTE["border"])

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_efficient_frontier(
    sim_vols: np.ndarray,
    sim_returns: np.ndarray,
    frontier_vols: np.ndarray,
    frontier_returns: np.ndarray,
    optimal_vol: float,
    optimal_return: float,
    min_vol: Optional[float] = None,
    min_vol_return: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Markowitz Efficient Frontier matching Bottom Panel."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Convert to percent
    sv_pct = sim_vols * 100
    sr_pct = sim_returns * 100
    fv_pct = frontier_vols * 100
    fr_pct = frontier_returns * 100
    opt_v_pct = optimal_vol * 100
    opt_r_pct = optimal_return * 100

    # Scatter random portfolios
    ax.scatter(sv_pct, sr_pct, c="#3b82f6", alpha=0.35, s=6, edgecolors="none", zorder=2, label="Random Portfolios")

    # Efficient Frontier curve
    ax.plot(fv_pct, fr_pct, color=COLOR_PALETTE["gold_bright"], lw=2.5, zorder=4, label="Efficient Frontier")

    # Optimal / Tangency Portfolio point
    ax.scatter([opt_v_pct], [opt_r_pct], color=COLOR_PALETTE["gold_bright"], s=100, zorder=5, marker="o", edgecolors="#ffffff", linewidths=1.5)
    ax.text(opt_v_pct + 0.8, opt_r_pct, "Optimal Portfolio", color=COLOR_PALETTE["gold_bright"], fontsize=9, fontweight="bold", va="center", zorder=6)

    # Min Volatility point if provided
    if min_vol is not None and min_vol_return is not None:
        mv_pct = min_vol * 100
        mr_pct = min_vol_return * 100
        ax.scatter([mv_pct], [mr_pct], color=COLOR_PALETTE["bull_green"], s=70, zorder=5, marker="s", edgecolors="#ffffff")
        ax.text(mv_pct + 0.6, mr_pct - 1.0, "Min Vol", color=COLOR_PALETTE["bull_green"], fontsize=8, fontweight="bold", va="center", zorder=6)

    ax.set_title("EFFICIENT FRONTIER (MEAN-VARIANCE OPTIMIZATION)", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel("Annualized Volatility (%)", color=COLOR_PALETTE["text_dim"], fontsize=9)
    ax.set_ylabel("Annualized Return (%)", color=COLOR_PALETTE["text_dim"], fontsize=9)
    ax.tick_params(axis="both", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_PALETTE["border"])

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_master_risk_infographic(
    returns: pd.Series,
    metrics_dict: Dict[str, Any],
    mc_paths: np.ndarray,
    sim_vols: np.ndarray,
    sim_returns: np.ndarray,
    frontier_vols: np.ndarray,
    frontier_returns: np.ndarray,
    optimal_vol: float,
    optimal_return: float,
    output_path: str = "/working_dir/quant_risk_models/output/quant_risk_models_infographic.png",
) -> plt.Figure:
    """Composites the full 5-project risk infographic matching Screenshot 3/7."""
    apply_dark_theme()
    fig = plt.figure(figsize=(14, 18), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(4, 2, height_ratios=[0.45, 1.2, 1.2, 1.3], hspace=0.35, wspace=0.25)

    # Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")

    ax_hdr.text(0.0, 0.75, "RISK & PORTFOLIO PROJECTS", color=COLOR_PALETTE["gold_bright"], fontsize=24, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "06-10 — BUILD RISK MODELS", color=COLOR_PALETTE["gold"], fontsize=14, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "Value at Risk, Expected Shortfall, Monte Carlo Engines & Mean-Variance Optimization", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # Panel 01: Top Wide (Distribution of Portfolio Returns + VaR / ES)
    ax_dist = fig.add_subplot(gs[1, :])
    plot_distribution_and_var(
        returns=returns,
        var_95=metrics_dict.get("var_95", -0.0245),
        var_99=metrics_dict.get("var_99", -0.0367),
        es_95=metrics_dict.get("cvar_95", -0.0315),
        ax=ax_dist,
    )

    # Panel 02: Middle Left (Risk Metrics Table Card)
    ax_card = fig.add_subplot(gs[2, 0])
    plot_risk_metrics_card(metrics_dict, ax=ax_card)

    # Panel 03: Middle Right (Monte Carlo VaR Simulation Fan Chart)
    ax_mc = fig.add_subplot(gs[2, 1])
    plot_monte_carlo_simulation(mc_paths, confidence_level=0.95, n_simulations=100000, ax=ax_mc)

    # Panel 04: Bottom Panel (Efficient Frontier + Explanatory Notes)
    ax_opt = fig.add_subplot(gs[3, :])
    plot_efficient_frontier(
        sim_vols=sim_vols,
        sim_returns=sim_returns,
        frontier_vols=frontier_vols,
        frontier_returns=frontier_returns,
        optimal_vol=optimal_vol,
        optimal_return=optimal_return,
        ax=ax_opt,
    )

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
