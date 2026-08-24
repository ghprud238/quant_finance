"""Visualizations for Systematic Trading Strategies (Projects 11-15).

Reproduces the professional dark-theme infographic styling with gold/amber accents.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

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
    "green_buy": "#10b981",
    "red_sell": "#ef4444",
    "blue_asset": "#38bdf8",
    "yellow_eq": "#facc15",
    "purple_fx": "#a855f7",
    "orange_comm": "#f97316",
}


def apply_dark_theme():
    """Applies global dark theme for matplotlib."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_mean_reversion(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    ma: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    buy_signals: pd.Series,
    sell_signals: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Moving Average Mean-Reversion strategy (Card 11)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, prices, color="#e2e8f0", lw=1.5, label="Price")
    ax.plot(dates, ma, color=COLOR_PALETTE["gold_bright"], lw=1.2, linestyle="-", label="Moving Average")
    ax.plot(dates, upper_band, color=COLOR_PALETTE["border"], lw=0.9, linestyle="--")
    ax.plot(dates, lower_band, color=COLOR_PALETTE["border"], lw=0.9, linestyle="--")
    ax.fill_between(dates, lower_band, upper_band, color="#1e2230", alpha=0.3)

    # Buy / Sell markers
    buy_idx = dates[buy_signals.astype(bool)]
    sell_idx = dates[sell_signals.astype(bool)]

    if len(buy_idx) > 0:
        ax.scatter(buy_idx, prices.loc[buy_idx], color=COLOR_PALETTE["green_buy"], s=35, zorder=5, label="Buy (Oversold)")
    if len(sell_idx) > 0:
        ax.scatter(sell_idx, prices.loc[sell_idx], color=COLOR_PALETTE["red_sell"], s=35, zorder=5, label="Sell (Overbought)")

    ax.set_title("11 | MEAN REVERSION EXAMPLE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5, ncol=2)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_momentum(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    trend_ma: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Momentum Trading strategy (Card 12)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, prices, color="#e2e8f0", lw=1.5, label="Price")
    ax.plot(dates, trend_ma, color=COLOR_PALETTE["gold_bright"], lw=1.2, linestyle="-", label="Trend Filter")

    entry_idx = dates[entries.astype(bool)]
    exit_idx = dates[exits.astype(bool)]

    if len(entry_idx) > 0:
        ax.scatter(entry_idx, prices.loc[entry_idx], color=COLOR_PALETTE["green_buy"], s=35, zorder=5, label="Enter (Momentum)")
    if len(exit_idx) > 0:
        ax.scatter(exit_idx, prices.loc[exit_idx], color=COLOR_PALETTE["red_sell"], s=35, zorder=5, label="Exit (Weakening)")

    ax.set_title("12 | MOMENTUM EXAMPLE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5, ncol=2)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_pairs_spread(
    dates: pd.DatetimeIndex,
    spread: pd.Series,
    z_score: pd.Series,
    upper_thresh: float = 2.0,
    lower_thresh: float = -2.0,
    long_spread_signals: Optional[pd.Series] = None,
    short_spread_signals: Optional[pd.Series] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Pairs Trading / Statistical Arbitrage spread & z-score (Card 13)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, z_score, color="#e2e8f0", lw=1.2, label="Normalized Spread (Z)")
    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle="-", lw=0.8)
    ax.axhline(upper_thresh, color=COLOR_PALETTE["red_sell"], linestyle="--", lw=1.0, label=f"Short Threshold (+{upper_thresh}σ)")
    ax.axhline(lower_thresh, color=COLOR_PALETTE["green_buy"], linestyle="--", lw=1.0, label=f"Long Threshold ({lower_thresh}σ)")
    ax.fill_between(dates, lower_thresh, upper_thresh, color="#1e2230", alpha=0.2)

    if long_spread_signals is not None:
        ls_idx = dates[long_spread_signals.astype(bool)]
        if len(ls_idx) > 0:
            ax.scatter(ls_idx, z_score.loc[ls_idx], color=COLOR_PALETTE["green_buy"], s=35, zorder=5, label="Long Spread")

    if short_spread_signals is not None:
        ss_idx = dates[short_spread_signals.astype(bool)]
        if len(ss_idx) > 0:
            ax.scatter(ss_idx, z_score.loc[ss_idx], color=COLOR_PALETTE["red_sell"], s=35, zorder=5, label="Short Spread")

    ax.set_title("13 | PAIR SPREAD OVER TIME", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5, ncol=3)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_factor_exposure_heatmap(
    factor_matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Factor-Based Long/Short cross-sectional matrix heatmap (Card 14)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Colormap: Dark Red -> Dark Yellow -> Bright Green (Low to High factor exposure)
    cmap = sns.diverging_palette(10, 130, s=85, l=45, n=100, as_cmap=True)

    sns.heatmap(
        factor_matrix,
        ax=ax,
        cmap=cmap,
        vmin=-2.5,
        vmax=2.5,
        cbar_kws={"shrink": 0.8, "label": "Factor Z-Score (Low → High)"},
        linewidths=0.5,
        linecolor=COLOR_PALETTE["bg_main"],
    )

    ax.set_title("14 | FACTOR EXPOSURE EXAMPLE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelcolor=COLOR_PALETTE["text_light"], labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["text_light"], labelsize=8, rotation=0)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_multi_asset_trend(
    dates: pd.DatetimeIndex,
    equity_curve_eq: pd.Series,
    equity_curve_bonds: pd.Series,
    equity_curve_fx: pd.Series,
    equity_curve_comm: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Multi-Asset Trend-Following performance across asset classes (Card 15)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, equity_curve_eq, color=COLOR_PALETTE["yellow_eq"], lw=1.5, label="Equities")
    ax.plot(dates, equity_curve_bonds, color=COLOR_PALETTE["blue_asset"], lw=1.5, label="Bonds")
    ax.plot(dates, equity_curve_fx, color=COLOR_PALETTE["green_buy"], lw=1.5, label="FX")
    ax.plot(dates, equity_curve_comm, color=COLOR_PALETTE["orange_comm"], lw=1.5, label="Commodities")

    ax.set_title("15 | MULTI-ASSET TREND FOLLOWING", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=8, ncol=4)
    ax.set_ylabel("Normalized Growth", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_workflow_pipeline(
    ax: plt.Axes,
):
    """Renders the systematic workflow pipeline card matching the bottom banner."""
    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.5, 0.82, "EVERY STRATEGY SHOULD INCLUDE:", color=COLOR_PALETTE["text_light"], fontsize=11, fontweight="bold", ha="center")

    steps = ["SIGNAL", "POSITION\nSIZING", "TRANSACTION\nCOSTS", "BACKTEST", "RISK\nMETRICS", "OUT-OF-SAMPLE\nTEST"]
    n_steps = len(steps)
    x_positions = np.linspace(0.08, 0.92, n_steps)

    for i, (step_name, x_pos) in enumerate(zip(steps, x_positions)):
        # Step box
        rect = patches.FancyBboxPatch(
            (x_pos - 0.055, 0.25), 0.11, 0.42,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#181b26",
            edgecolor=COLOR_PALETTE["gold"],
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x_pos, 0.46, step_name, color=COLOR_PALETTE["gold_bright"], fontsize=7.5, fontweight="bold", ha="center", va="center", linespacing=1.2, transform=ax.transAxes)

        if i < n_steps - 1:
            next_x = x_positions[i+1]
            ax.annotate("", xy=(next_x - 0.065, 0.46), xytext=(x_pos + 0.065, 0.46),
                        arrowprops=dict(arrowstyle="->", color=COLOR_PALETTE["gold"], lw=1.5),
                        xycoords="axes fraction")

    # Warning text at bottom
    warn_text = "⚠️ A backtest with no transaction costs or out-of-sample validation isn't impressive. It's usually just curve-fitting."
    ax.text(0.5, 0.08, warn_text, color=COLOR_PALETTE["gold_bright"], fontsize=8.5, fontweight="bold", ha="center", transform=ax.transAxes)


def plot_master_systematic_infographic(
    mr_data: Dict[str, Any],
    mom_data: Dict[str, Any],
    pairs_data: Dict[str, Any],
    factor_matrix: pd.DataFrame,
    macro_curves: Dict[str, pd.Series],
    output_path: str = "/working_dir/systematic_strategies/output/systematic_strategies_infographic.png",
) -> plt.Figure:
    """Composites the full 5-strategy infographic matching Screenshot 4/7."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(7, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "QUANT TRADING PROJECTS", color=COLOR_PALETTE["gold_bright"], fontsize=24, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "11-15 — BUILD SYSTEMATIC STRATEGIES", color=COLOR_PALETTE["gold"], fontsize=14, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "Mean Reversion, Momentum, Statistical Arbitrage, Factor Long/Short & Multi-Asset Trend Systems", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Strategy 11 (Mean Reversion)
    ax_11 = fig.add_subplot(gs[1, 0])
    plot_mean_reversion(
        dates=mr_data["dates"],
        prices=mr_data["prices"],
        ma=mr_data["ma"],
        upper_band=mr_data["upper_band"],
        lower_band=mr_data["lower_band"],
        buy_signals=mr_data["buy_signals"],
        sell_signals=mr_data["sell_signals"],
        ax=ax_11,
    )
    ax_11_txt = fig.add_subplot(gs[1, 1])
    ax_11_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_11_txt.axis("off")
    ax_11_txt.text(0.05, 0.75, "11 | Moving Average Mean-Reversion", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_11_txt.text(0.05, 0.45, "• Price deviates from its moving average and reverts back.\n• Statistical Z-score & Bollinger Band boundaries.\n• Buy low (Oversold), Sell high (Overbought).\n• Strictly market-disciplined exit on mean reversion.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Strategy 12 (Momentum)
    ax_12 = fig.add_subplot(gs[2, 0])
    plot_momentum(
        dates=mom_data["dates"],
        prices=mom_data["prices"],
        trend_ma=mom_data["trend_ma"],
        entries=mom_data["entries"],
        exits=mom_data["exits"],
        ax=ax_12,
    )
    ax_12_txt = fig.add_subplot(gs[2, 1])
    ax_12_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_12_txt.axis("off")
    ax_12_txt.text(0.05, 0.75, "12 | Momentum Trading Strategy", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_12_txt.text(0.05, 0.45, "• Ride strong persistent trends across multiple horizons.\n• Dual Moving Average cross + Time-Series Momentum (TSMOM).\n• Enter on trend ignition, exit when momentum weakens.\n• 'Trend is your friend' with volatility risk control.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Strategy 13 (Pairs Trading)
    ax_13 = fig.add_subplot(gs[3, 0])
    plot_pairs_spread(
        dates=pairs_data["dates"],
        spread=pairs_data["spread"],
        z_score=pairs_data["z_score"],
        long_spread_signals=pairs_data.get("long_signals"),
        short_spread_signals=pairs_data.get("short_signals"),
        ax=ax_13,
    )
    ax_13_txt = fig.add_subplot(gs[3, 1])
    ax_13_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_13_txt.axis("off")
    ax_13_txt.text(0.05, 0.75, "13 | Pairs Trading / Statistical Arbitrage", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_13_txt.text(0.05, 0.45, "• Trade cointegrated pairs with mean-reverting stationary spread.\n• Engle-Granger ADF cointegration & Ornstein-Uhlenbeck half-life.\n• Dynamic hedge ratio estimation via Kalman Filter / rolling OLS.\n• Dollar-neutral and market-neutral statistical arbitrage.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Strategy 14 (Factor Long/Short)
    ax_14 = fig.add_subplot(gs[4, 0])
    plot_factor_exposure_heatmap(factor_matrix=factor_matrix, ax=ax_14)
    ax_14_txt = fig.add_subplot(gs[4, 1])
    ax_14_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_14_txt.axis("off")
    ax_14_txt.text(0.05, 0.75, "14 | Factor-Based Long/Short Strategy", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_14_txt.text(0.05, 0.45, "• Cross-sectional multi-factor scoring (Value, Momentum, Quality, Low-Vol, Size).\n• Long Top Quintile (Q5 - high exposure), Short Bottom Quintile (Q1).\n• Dollar-neutral and beta-neutral factor risk harvest.\n• Systematic rebalancing with transaction cost smoothing.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Strategy 15 (Multi-Asset Trend)
    ax_15 = fig.add_subplot(gs[5, 0])
    plot_multi_asset_trend(
        dates=macro_curves["dates"],
        equity_curve_eq=macro_curves["equities"],
        equity_curve_bonds=macro_curves["bonds"],
        equity_curve_fx=macro_curves["fx"],
        equity_curve_comm=macro_curves["commodities"],
        ax=ax_15,
    )
    ax_15_txt = fig.add_subplot(gs[5, 1])
    ax_15_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_15_txt.axis("off")
    ax_15_txt.text(0.05, 0.75, "15 | Multi-Asset Trend-Following System", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_15_txt.text(0.05, 0.45, "• Capture macro trends across Equities, Bonds, FX, and Commodities.\n• Volatility targeting per asset (e.g. 10% annual vol).\n• Equal Risk Contribution (ERC) Risk Parity portfolio aggregation.\n• True multi-asset diversification with crisis alpha.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 7. Workflow Pipeline Bottom Banner
    ax_pipe = fig.add_subplot(gs[6, :])
    plot_workflow_pipeline(ax_pipe)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
