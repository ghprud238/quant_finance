"""Visualizations for Quantitative Finance Foundations.

Styled with a dark aesthetic and gold/amber accents matching professional quant infographics.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import seaborn as sns

DARK_THEME_STYLE = {
    "figure.facecolor": "#0c0d12",
    "axes.facecolor": "#14161f",
    "axes.edgecolor": "#2a2e3d",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#1e2230",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
}

COLOR_PALETTE = {
    "bg_main": "#0c0d12",
    "bg_card": "#14161f",
    "border": "#2a2e3d",
    "gold": "#f59e0b",
    "gold_bright": "#fbbf24",
    "gold_dark": "#b45309",
    "text_light": "#f8fafc",
    "text_dim": "#94a3b8",
    "text_muted": "#64748b",
    "bull_green": "#10b981",
    "bear_red": "#ef4444",
    "neutral_gray": "#334155",
    "return_gray": "#64748b",
}


def apply_dark_theme():
    """Applies global dark theme for quant plots."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_equity_price_and_returns(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    returns: pd.Series,
    ticker: str = "AAPL",
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots dual-axis Equity Price & Returns matching Card 01."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Right axis: Returns (thin bars or line)
    ax2.plot(dates, returns * 100, color=COLOR_PALETTE["return_gray"], alpha=0.6, lw=0.8, label="Returns (%)")
    ax2.set_ylabel("Returns (%)", color=COLOR_PALETTE["text_dim"], fontsize=9)
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax2.grid(False)

    # Left axis: Price (gold line)
    ax.plot(dates, prices, color=COLOR_PALETTE["gold_bright"], lw=1.8, label="Price")
    ax.set_ylabel("Price ($)", color=COLOR_PALETTE["gold_bright"], fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=8)
    ax.tick_params(axis="x", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_PALETTE["border"])

    ax.set_title(f"01 | {ticker} EQUITY PRICE & RETURNS", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=10, loc="left")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=8)

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_risk_dashboard_summary(
    metrics: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Renders visual risk metric cards matching Card 02."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    cards = [
        ("Annualized Volatility", f"{metrics.get('annualized_volatility', 0.186):.1%}", COLOR_PALETTE["gold_bright"], 0.18, 0.65),
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 1.24):.2f}", COLOR_PALETTE["gold_bright"], 0.68, 0.65),
        ("Max Drawdown", f"{metrics.get('max_drawdown', -0.243):.1%}", COLOR_PALETTE["bear_red"], 0.18, 0.35),
        ("VaR (95%)", f"{metrics.get('var_95', -0.0235):.2%}", COLOR_PALETTE["bear_red"], 0.68, 0.35),
        ("Realized Beta", f"{metrics.get('realized_beta', 1.05):.2f}", COLOR_PALETTE["text_light"], 0.43, 0.08),
    ]

    ax.text(0.02, 0.94, "02 | PORTFOLIO RISK DASHBOARD", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", transform=ax.transAxes)

    for label, val_str, val_color, x_pos, y_pos in cards:
        # Card box
        box_width = 0.40 if x_pos != 0.43 else 0.50
        box_height = 0.22 if x_pos != 0.43 else 0.20
        rect = patches.FancyBboxPatch(
            (x_pos - box_width / 2, y_pos - 0.06),
            box_width,
            box_height,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor="#1a1d29",
            edgecolor=COLOR_PALETTE["border"],
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        ax.text(x_pos, y_pos + 0.08, label, color=COLOR_PALETTE["text_dim"], fontsize=8, ha="center", va="center", transform=ax.transAxes)
        ax.text(x_pos, y_pos, val_str, color=val_color, fontsize=15, fontweight="bold", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots cross-asset correlation heatmap matching Card 03."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Custom colormap: Dark blue -> dark gray -> gold/amber
    cmap = sns.diverging_palette(220, 38, s=90, l=45, n=100, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "weight": "bold", "color": "#f8fafc"},
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        linewidths=1.0,
        linecolor=COLOR_PALETTE["bg_main"],
    )

    ax.set_title("03 | CORRELATION HEATMAP", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=10, loc="left")
    ax.tick_params(axis="x", labelcolor=COLOR_PALETTE["text_light"], labelsize=8, rotation=0)
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["text_light"], labelsize=8, rotation=0)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8, labelcolor=COLOR_PALETTE["text_dim"])
    cbar.outline.set_edgecolor(COLOR_PALETTE["border"])

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_factor_exposures(
    exposures: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots horizontal factor exposure bar chart matching Card 04."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    factors = list(exposures.index)
    values = list(exposures.values)

    y_pos = np.arange(len(factors))
    colors = [COLOR_PALETTE["gold_bright"] if v >= 0 else COLOR_PALETTE["gold_dark"] for v in values]

    bars = ax.barh(y_pos, values, color=colors, edgecolor=COLOR_PALETTE["border"], height=0.55, align="center")

    ax.axvline(0, color=COLOR_PALETTE["text_dim"], linestyle="-", lw=1.0, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, color=COLOR_PALETTE["text_light"], fontsize=9, fontweight="bold")
    ax.invert_yaxis()  # top-down

    ax.set_xlim(-1.2, 1.2)
    ax.tick_params(axis="x", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3, color=COLOR_PALETTE["border"])

    for bar, val in zip(bars, values):
        x_offset = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        ax.text(val + x_offset, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha=ha, color=COLOR_PALETTE["text_light"], fontsize=8, fontweight="bold")

    ax.set_title("04 | FACTOR EXPOSURE", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=10, loc="left")

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_market_regime_timeline(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    regimes: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots price curve with Bull/Bear/Neutral regime shading matching Card 05."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Normalize price or plot cumulative return for visual clarity
    norm_price = (prices / prices.iloc[0] - 1.0)

    # Shading regions
    regime_colors = {
        "Bull": (COLOR_PALETTE["bull_green"], 0.25),
        "Bear": (COLOR_PALETTE["bear_red"], 0.28),
        "Neutral": (COLOR_PALETTE["neutral_gray"], 0.15),
        2: (COLOR_PALETTE["bull_green"], 0.25),
        0: (COLOR_PALETTE["bear_red"], 0.28),
        1: (COLOR_PALETTE["neutral_gray"], 0.15),
    }

    # Vectorized / span drawing
    current_regime = regimes.iloc[0]
    start_idx = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
            col, alpha = regime_colors.get(current_regime, (COLOR_PALETTE["neutral_gray"], 0.15))
            ax.axvspan(dates[start_idx], dates[i], color=col, alpha=alpha, lw=0)
            current_regime = regimes.iloc[i]
            start_idx = i

    ax.plot(dates, norm_price, color=COLOR_PALETTE["text_light"], lw=1.5, label="Cumulative Return")
    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle=":", lw=0.8)

    # Custom legend patches
    p_bull = patches.Patch(color=COLOR_PALETTE["bull_green"], alpha=0.5, label="Bull")
    p_bear = patches.Patch(color=COLOR_PALETTE["bear_red"], alpha=0.5, label="Bear")
    p_neut = patches.Patch(color=COLOR_PALETTE["neutral_gray"], alpha=0.4, label="Neutral")

    ax.legend(handles=[p_bull, p_bear, p_neut], loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=8, ncol=3)

    ax.set_title("05 | MARKET REGIME DETECTION", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=10, loc="left")
    ax.tick_params(axis="both", labelcolor=COLOR_PALETTE["text_dim"], labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3, color=COLOR_PALETTE["border"])

    plt.tight_layout()
    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_master_infographic(
    dates: pd.DatetimeIndex,
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    risk_metrics: Dict[str, Any],
    corr_matrix: pd.DataFrame,
    factor_exposures: pd.Series,
    regimes: pd.Series,
    output_path: str = "/working_dir/quant_foundations/output/quant_foundations_infographic.png",
) -> plt.Figure:
    """Generates the full composite 5-panel infographic dashboard reproducing the screenshot."""
    apply_dark_theme()
    fig = plt.figure(figsize=(14, 18), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(4, 2, height_ratios=[0.5, 1.2, 1.2, 1.2], hspace=0.35, wspace=0.25)

    # Top Header
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_header.axis("off")

    ax_header.text(0.0, 0.75, "FOUNDATIONS & MARKET DATA", color=COLOR_PALETTE["gold_bright"], fontsize=24, fontweight="bold", fontfamily="sans-serif")
    ax_header.text(0.0, 0.40, "01-05 — BUILD YOUR QUANT FOUNDATION", color=COLOR_PALETTE["gold"], fontsize=14, fontweight="bold")
    ax_header.text(0.0, 0.12, "Core mathematical, statistical & econometric infrastructure for quantitative finance", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # Panel 01: Stock Returns & Volatility (Top Wide or Row 1 Left)
    ax_01 = fig.add_subplot(gs[1, 0])
    target_ticker = "AAPL" if "AAPL" in prices_df.columns else prices_df.columns[0]
    plot_equity_price_and_returns(dates, prices_df[target_ticker], returns_df[target_ticker], ticker=target_ticker, ax=ax_01)

    # Panel 02: Portfolio Risk Dashboard
    ax_02 = fig.add_subplot(gs[1, 1])
    plot_risk_dashboard_summary(risk_metrics, ax=ax_02)

    # Panel 03: Correlation Heatmap
    ax_03 = fig.add_subplot(gs[2, 0])
    plot_correlation_heatmap(corr_matrix, ax=ax_03)

    # Panel 04: Factor Exposure
    ax_04 = fig.add_subplot(gs[2, 1])
    plot_factor_exposures(factor_exposures, ax=ax_04)

    # Panel 05: Market Regime Detection (Spans bottom width)
    ax_05 = fig.add_subplot(gs[3, :])
    bench_ticker = "SPY" if "SPY" in prices_df.columns else prices_df.columns[0]
    plot_market_regime_timeline(dates, prices_df[bench_ticker], regimes, ax=ax_05)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
