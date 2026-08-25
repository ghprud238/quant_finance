"""Visualizations for Interview-Worthy Quant Projects (26-30).

Reproduces the dark-theme infographic styling with gold/amber accents.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
}


def apply_dark_theme():
    """Applies global dark theme."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_order_book_snapshot(
    snapshot_df: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Level 2 Order Book Depth & Microstructure Snapshot (Card 26)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.92, "26 | ORDER BOOK SNAPSHOT", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold")

    # Headers
    headers = ["BID", "PRICE", "ASK"]
    x_positions = [0.20, 0.50, 0.80]
    y_start = 0.80

    for h, x in zip(headers, x_positions):
        ax.text(x, y_start, h, color=COLOR_PALETTE["text_dim"], fontsize=8.5, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.plot([0.08, 0.92], [y_start - 0.03, y_start - 0.03], color=COLOR_PALETTE["border"], lw=1.0, transform=ax.transAxes)

    # Sample Level 2 ladder matching screenshot
    ladder = [
        ("500", "100.05", "600", False),
        ("800", "100.04", "700", False),
        ("1200", "100.03", "900", False),
        ("—", "100.02", "—", False),
        ("1000", "100.01", "1100", False),
        ("700", "100.00", "1000", False),
        ("700", "99.99", "800", False),
    ]

    for i, (bid_vol, price_str, ask_vol, _) in enumerate(ladder):
        y = y_start - 0.08 - i * 0.085
        # Bid volume bar
        if bid_vol != "—":
            ax.text(0.20, y, bid_vol, color=COLOR_PALETTE["green"], fontsize=8.5, fontweight="bold", ha="center", transform=ax.transAxes)
        else:
            ax.text(0.20, y, "—", color=COLOR_PALETTE["text_muted"], fontsize=8.5, ha="center", transform=ax.transAxes)

        # Price
        ax.text(0.50, y, price_str, color=COLOR_PALETTE["text_light"], fontsize=8.5, fontweight="bold", ha="center", transform=ax.transAxes)

        # Ask volume bar
        if ask_vol != "—":
            ax.text(0.80, y, ask_vol, color=COLOR_PALETTE["red"], fontsize=8.5, fontweight="bold", ha="center", transform=ax.transAxes)
        else:
            ax.text(0.80, y, "—", color=COLOR_PALETTE["text_muted"], fontsize=8.5, ha="center", transform=ax.transAxes)

        ax.plot([0.08, 0.92], [y - 0.025, y - 0.025], color=COLOR_PALETTE["border"], lw=0.5, alpha=0.4, transform=ax.transAxes)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_execution_trajectory(
    time_grid: np.ndarray,
    market_price: np.ndarray,
    execution_price: np.ndarray,
    shortfall_line: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Optimal Execution Trajectory & Implementation Shortfall (Card 27)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(time_grid, market_price, color="#e2e8f0", lw=1.5, label="Market Price")
    ax.scatter(time_grid, execution_price, color=COLOR_PALETTE["gold_bright"], s=28, zorder=5, label="Execution Price")

    if shortfall_line is not None:
        ax.plot(time_grid, shortfall_line, color=COLOR_PALETTE["gold_dark"], linestyle="--", lw=1.2, label="Implementation Shortfall")

    ax.set_title("27 | EXECUTION TRAJECTORY", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Time (Trading Intervals)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Price ($)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_stress_test_bars(
    scenario_names: Optional[List[str]] = None,
    pnl_impacts: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Historical & Hypothetical Stress Test Scenarios (Card 28)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    if scenario_names is None or pnl_impacts is None:
        scenario_names = [
            "2008 Financial Crisis",
            "2020 COVID Crash",
            "Rate Shock (+200bps)",
            "Market Crash (-30%)",
            "Custom Scenario",
        ]
        pnl_impacts = [-12.4, -8.7, -6.1, -15.3, -9.2]

    y_pos = np.arange(len(scenario_names))
    bars = ax.barh(y_pos, pnl_impacts, color=COLOR_PALETTE["red"], edgecolor=COLOR_PALETTE["border"], height=0.55, align="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenario_names, color=COLOR_PALETTE["text_light"], fontsize=8, fontweight="bold")
    ax.invert_yaxis()  # Top-down

    ax.set_xlim(-22, 2)
    ax.set_xlabel("P&L Impact (in %)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.axvline(0, color=COLOR_PALETTE["text_muted"], linestyle="-", lw=1.0)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, pnl_impacts):
        ax.text(val - 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="right", color=COLOR_PALETTE["text_light"], fontsize=7.5, fontweight="bold")

    ax.set_title("28 | STRESS TEST EXAMPLE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_pipeline_flowchart(
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots End-to-End Quant Research Pipeline Flowchart (Card 29)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.92, "29 | PIPELINE OVERVIEW", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold")

    stages = ["DATA", "FEATURES", "BACKTEST", "EVALUATE", "DEPLOY"]
    x_coords = [0.10, 0.30, 0.50, 0.70, 0.90]

    for i, (stage_name, x) in enumerate(zip(stages, x_coords)):
        rect = patches.FancyBboxPatch(
            (x - 0.08, 0.50), 0.16, 0.32,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#181b26",
            edgecolor=COLOR_PALETTE["gold"],
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x, 0.66, stage_name, color=COLOR_PALETTE["gold_bright"], fontsize=8, fontweight="bold", ha="center", va="center", transform=ax.transAxes)

        if i < len(stages) - 1:
            next_x = x_coords[i+1]
            ax.annotate("", xy=(next_x - 0.09, 0.66), xytext=(x + 0.09, 0.66),
                        arrowprops=dict(arrowstyle="->", color=COLOR_PALETTE["gold"], lw=1.5),
                        xycoords="axes fraction")

    # Feedback Loop arrow
    ax.annotate("", xy=(0.10, 0.45), xytext=(0.90, 0.45),
                arrowprops=dict(arrowstyle="->", color=COLOR_PALETTE["blue"], lw=1.5, linestyle="--"),
                xycoords="axes fraction")
    ax.text(0.50, 0.38, "FEEDBACK LOOP", color=COLOR_PALETTE["blue"], fontsize=8, fontweight="bold", ha="center", transform=ax.transAxes)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_strategy_equity_curve_oos(
    dates: pd.DatetimeIndex,
    equity_curve: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Out-of-Sample (OOS) Strategy Equity Curve (Card 30)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Normalized curve (starting at 0.0 or 1.0)
    norm_curve = equity_curve / equity_curve.iloc[0] - 1.0

    ax.plot(dates, norm_curve, color=COLOR_PALETTE["gold_bright"], lw=1.8, label="OOS Strategy Return")
    ax.fill_between(dates, 0, norm_curve, color=COLOR_PALETTE["gold_dark"], alpha=0.25)
    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle="--", lw=0.8)

    ax.set_title("30 | STRATEGY EQUITY CURVE (OOS)", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Cumulative Return", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_interview_badges_footer(ax: plt.Axes):
    """Renders bottom banner with differentiation badges."""
    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.85, "TECHNICAL SKILLS CAN GET YOU AN INTERVIEW. THESE PROJECTS CAN GET YOU THE OFFER.", color=COLOR_PALETTE["gold_bright"], fontsize=9.5, fontweight="bold", transform=ax.transAxes)

    badges = [
        ("Solve real\nproblems", "Market impact & liquidity constraints"),
        ("Show depth\n& curiosity", "Depth beyond textbook theory"),
        ("Think like\na quant", "Risk before returns mindset"),
        ("Stand out in\ninterviews", "Reproducible research systems"),
        ("Build your\nedge", "End-to-end tradable systems"),
    ]
    x_positions = [0.10, 0.30, 0.50, 0.70, 0.90]

    for (b_title, b_desc), x in zip(badges, x_positions):
        rect = patches.FancyBboxPatch(
            (x - 0.08, 0.15), 0.16, 0.55,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#181b26",
            edgecolor=COLOR_PALETTE["gold"],
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x, 0.52, b_title, color=COLOR_PALETTE["gold_bright"], fontsize=7.5, fontweight="bold", ha="center", va="center", linespacing=1.2, transform=ax.transAxes)
        ax.text(x, 0.28, b_desc, color=COLOR_PALETTE["text_dim"], fontsize=6.0, ha="center", va="center", linespacing=1.2, transform=ax.transAxes)


def plot_master_interview_infographic(
    execution_data: Dict[str, Any],
    stress_data: Dict[str, Any],
    strategy_data: Dict[str, Any],
    output_path: str = "/working_dir/interview_quant_projects/output/interview_quant_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module capstone infographic matching Screenshot 7/7."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(7, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "THE PROJECTS THAT DIFFERENTIATE YOU", color=COLOR_PALETTE["gold_bright"], fontsize=24, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "26-30 — BUILD SOMETHING INTERVIEW-WORTHY", color=COLOR_PALETTE["gold"], fontsize=14, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "Limit Order Books, Almgren-Chriss Optimal Execution, Stress Testing, Pipelines & Production Trading", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Card 26 (LOB)
    ax_26 = fig.add_subplot(gs[1, 0])
    plot_order_book_snapshot(ax=ax_26)
    ax_26_txt = fig.add_subplot(gs[1, 1])
    ax_26_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_26_txt.axis("off")
    ax_26_txt.text(0.05, 0.75, "26 | Limit Order Book / Market Microstructure", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_26_txt.text(0.05, 0.45, "• Continuous double auction matching with FIFO Price-Time priority.\n• Real-time Order Book Imbalance (OBI) & Micro-Price tracking.\n• Stochastic Poisson limit order arrivals, cancels & market executions.\n• 'Understand how markets really work beneath the surface.'", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Card 27 (Optimal Execution)
    ax_27 = fig.add_subplot(gs[2, 0])
    plot_execution_trajectory(
        time_grid=execution_data["time_grid"],
        market_price=execution_data["market_price"],
        execution_price=execution_data["execution_price"],
        shortfall_line=execution_data.get("shortfall_line"),
        ax=ax_27,
    )
    ax_27_txt = fig.add_subplot(gs[2, 1])
    ax_27_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_27_txt.axis("off")
    ax_27_txt.text(0.05, 0.75, "27 | Optimal Execution Models (Almgren-Chriss)", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_27_txt.text(0.05, 0.45, "• Calculus of variations closed-form trajectory minimizing market impact.\n• Dynamic trade-off between temporary impact (eta) & holding risk (vol).\n• Perold Implementation Shortfall (IS) component attribution.\n• 'Combine math, market impact models and real-world trading constraints.'", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Card 28 (Stress Testing)
    ax_28 = fig.add_subplot(gs[3, 0])
    plot_stress_test_bars(
        scenario_names=stress_data.get("scenario_names"),
        pnl_impacts=stress_data.get("pnl_impacts"),
        ax=ax_28,
    )
    ax_28_txt = fig.add_subplot(gs[3, 1])
    ax_28_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_28_txt.axis("off")
    ax_28_txt.text(0.05, 0.75, "28 | Portfolio Risk & Stress Testing Engine", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_28_txt.text(0.05, 0.45, "• Historical crisis simulation (2008 GFC, 2020 COVID, 2022 Rates).\n• Hypothetical 2D factor shocks (Equity meltdown x Yield curve shift).\n• Systemic correlation breakdown & Stressed Expected Shortfall.\n• 'Every firm wants quants who understand risk before returns.'", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Card 29 (Research Pipeline)
    ax_29 = fig.add_subplot(gs[4, 0])
    plot_pipeline_flowchart(ax=ax_29)
    ax_29_txt = fig.add_subplot(gs[4, 1])
    ax_29_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_29_txt.axis("off")
    ax_29_txt.text(0.05, 0.75, "29 | End-to-End Quant Research Pipeline", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_29_txt.text(0.05, 0.45, "• 5-Stage automated workflow: DATA -> FEATURES -> BACKTEST -> EVAL -> DEPLOY.\n• Stationarity with Fractional Differencing (FFD) & Purged K-Fold CV.\n• Live model health monitoring with automated drift detection & feedback.\n• 'Shows you can build scalable, reproducible research systems.'", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Card 30 (Production Trading System)
    ax_30 = fig.add_subplot(gs[5, 0])
    plot_strategy_equity_curve_oos(
        dates=strategy_data["dates"],
        equity_curve=strategy_data["equity_curve"],
        ax=ax_30,
    )
    ax_30_txt = fig.add_subplot(gs[5, 1])
    ax_30_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_30_txt.axis("off")
    ax_30_txt.text(0.05, 0.75, "30 | Full Production Systematic Trading System", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_30_txt.text(0.05, 0.45, "• Integrated alpha ensemble + volatility targeting + drawdown circuit breaker.\n• Pre-trade stress testing gate preventing catastrophic drawdown exposure.\n• Almgren-Chriss order execution router with friction deductions.\n• Out-of-sample (2020-2024) multi-year alpha compounding.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 7. Interview Badges Footer
    ax_badges = fig.add_subplot(gs[6, :])
    plot_interview_badges_footer(ax_badges)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
