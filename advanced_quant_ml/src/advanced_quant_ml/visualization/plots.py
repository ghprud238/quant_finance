"""Visualizations for Advanced Quant & Machine Learning (Projects 21-25).

Styled with a dark aesthetic, amber/gold accents, and crisp statistical typography.
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
    "blue": "#38bdf8",
    "green": "#10b981",
    "red": "#ef4444",
}


def apply_dark_theme():
    """Applies global dark theme."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_garch_forecast(
    dates: pd.DatetimeIndex,
    actual_vol: pd.Series,
    garch_vol: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots GARCH(1,1) Volatility Forecast vs Actual Volatility (Card 21)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, actual_vol, color="#e2e8f0", lw=1.2, label="Actual Volatility")
    ax.plot(dates, garch_vol, color=COLOR_PALETTE["gold_bright"], lw=1.5, linestyle="--", label="GARCH Forecast")

    ax.set_title("21 | GARCH(1,1) VOLATILITY FORECAST", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Volatility (Daily σ)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_yield_curve(
    maturities: np.ndarray,
    par_yields: np.ndarray,
    fitted_ns: Optional[np.ndarray] = None,
    tenor_labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots US Treasury Yield Curve with Nelson-Siegel model fit (Card 22)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    if tenor_labels is None:
        tenor_labels = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        maturities_sub = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 5.0, 10.0, 30.0])
    else:
        maturities_sub = maturities

    # Smooth curve
    if fitted_ns is not None:
        t_dense = np.linspace(0.08, 30.0, 200)
        ax.plot(np.arange(len(maturities)), par_yields, color=COLOR_PALETTE["gold"], lw=2.0, linestyle="-")
    else:
        ax.plot(np.arange(len(tenor_labels)), par_yields[:len(tenor_labels)], color=COLOR_PALETTE["gold"], lw=2.0, linestyle="-")

    # Par yield scatter markers
    ax.scatter(np.arange(len(tenor_labels)), par_yields[:len(tenor_labels)], color=COLOR_PALETTE["gold_bright"], s=45, zorder=5)

    ax.set_title("22 | EXAMPLE YIELD CURVE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Yield (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_xticks(np.arange(len(tenor_labels)))
    ax.set_xticklabels(tenor_labels, fontsize=7.5)
    ax.set_xlabel("Maturity", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylim(0.5, max(5.5, max(par_yields) + 0.8))
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_kalman_hedge_ratio(
    time_steps: np.ndarray,
    hedge_ratios: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Dynamic Hedge Ratio estimated via Kalman Filter (Card 23)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(time_steps, hedge_ratios, color=COLOR_PALETTE["gold_bright"], lw=1.5, label="Kalman Hedge Ratio (βt)")
    ax.axhline(np.mean(hedge_ratios), color=COLOR_PALETTE["text_muted"], linestyle=":", lw=1.0, label=f"Mean β ({np.mean(hedge_ratios):.2f})")

    ax.set_title("23 | DYNAMIC HEDGE RATIO (KALMAN FILTER)", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Time (Trading Days)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Hedge Ratio (β)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylim(min(0.0, min(hedge_ratios) - 0.2), max(2.0, max(hedge_ratios) + 0.3))
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_ml_predicted_returns(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    ic: float = 0.084,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Machine Learning Predicted Returns vs Actual Returns (Card 24)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Scatter dots
    ax.scatter(actual_returns, predicted_returns, color=COLOR_PALETTE["gold_bright"], alpha=0.5, s=12, zorder=3, label="Predictions")

    # Fit regression line
    m, b = np.polyfit(actual_returns, predicted_returns, 1)
    x_line = np.linspace(-0.10, 0.10, 100)
    ax.plot(x_line, m * x_line + b, color="#38bdf8", lw=1.8, linestyle="--", zorder=4, label=f"OOS Fit (IC = {ic:+.3f})")

    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle="-", lw=0.6)
    ax.axvline(0, color=COLOR_PALETTE["text_muted"], linestyle="-", lw=0.6)

    ax.set_title("24 | MODEL PREDICTED RETURNS (EXAMPLE)", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Actual Returns", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Predicted Returns", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_xlim(-0.11, 0.11)
    ax.set_ylim(-0.11, 0.11)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_alternative_data_signal(
    dates: pd.DatetimeIndex,
    signal_strength: pd.Series,
    forward_returns: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Alternative Data Signal Strength vs Forward Returns (Card 25)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Left: Signal Strength (Gold bar/line)
    l1 = ax.plot(dates, signal_strength, color=COLOR_PALETTE["gold_bright"], lw=1.6, label="Signal Strength (Z)")
    ax.set_ylabel("Signal Strength", color=COLOR_PALETTE["gold_bright"], fontsize=8)
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=7.5)
    ax.set_ylim(-2.5, 2.5)

    # Right: Forward Returns (Light gray/blue line)
    l2 = ax2.plot(dates, forward_returns * 100, color="#94a3b8", lw=1.1, alpha=0.75, label="Forward Returns (%)")
    ax2.set_ylabel("Forward Returns (%)", color=COLOR_PALETTE["text_dim"], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["text_dim"], labelsize=7.5)
    ax2.grid(False)

    ax.set_title("25 | ALTERNATIVE DATA SIGNAL EXAMPLE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_advanced_quant_philosophy(ax: plt.Axes):
    """Renders the 4 core quant research questions bottom banner."""
    ax.set_facecolor(COLOR_PALETTE["bg_card"])
    ax.axis("off")

    ax.text(0.04, 0.86, "THE GOAL ISN'T TO THROW XGBOOST OR NEURAL NETWORKS AT PRICES. THE GOAL IS TO ANSWER:", color=COLOR_PALETTE["gold_bright"], fontsize=9.5, fontweight="bold", transform=ax.transAxes)

    questions = [
        ("1", "What is the signal?", "Economic hypothesis & feature structure"),
        ("2", "Why should it exist?", "Behavioral bias or institutional friction"),
        ("3", "How stable is it?", "Stationarity & parameter persistence"),
        ("4", "Does it survive costs?", "Slippage, turnover & regime shifts"),
    ]
    x_positions = [0.12, 0.37, 0.62, 0.87]

    for (num, q_title, q_desc), x in zip(questions, x_positions):
        rect = patches.FancyBboxPatch(
            (x - 0.11, 0.12), 0.22, 0.58,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#181b26",
            edgecolor=COLOR_PALETTE["gold"],
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Number badge
        circle = patches.Circle((x - 0.07, 0.52), 0.024, facecolor=COLOR_PALETTE["gold"], transform=ax.transAxes)
        ax.add_patch(circle)
        ax.text(x - 0.07, 0.52, num, color="#0a0b0e", fontsize=7.5, fontweight="bold", ha="center", va="center", transform=ax.transAxes)

        ax.text(x - 0.03, 0.52, q_title, color=COLOR_PALETTE["gold_bright"], fontsize=7.5, fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(x, 0.28, q_desc, color=COLOR_PALETTE["text_dim"], fontsize=6.5, ha="center", va="center", linespacing=1.2, transform=ax.transAxes)


def plot_master_advanced_quant_infographic(
    garch_data: Dict[str, Any],
    yield_data: Dict[str, Any],
    kalman_data: Dict[str, Any],
    ml_data: Dict[str, Any],
    alt_data: Dict[str, Any],
    output_path: str = "/working_dir/advanced_quant_ml/output/advanced_quant_ml_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module Advanced Quant & ML infographic matching Screenshot 6/7."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(7, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "ADVANCED QUANT & ML", color=COLOR_PALETTE["gold_bright"], fontsize=24, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "21-25 — MOVE BEYOND BASIC STRATEGIES", color=COLOR_PALETTE["gold"], fontsize=14, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "GARCH Volatility, Yield Curve Term Structure, Kalman State-Space, ML Return Predictors & Alt Data", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Card 21 (GARCH)
    ax_21 = fig.add_subplot(gs[1, 0])
    plot_garch_forecast(
        dates=garch_data["dates"],
        actual_vol=garch_data["actual_vol"],
        garch_vol=garch_data["garch_vol"],
        ax=ax_21,
    )
    ax_21_txt = fig.add_subplot(gs[1, 1])
    ax_21_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_21_txt.axis("off")
    ax_21_txt.text(0.05, 0.75, "21 | Volatility Forecasting with GARCH", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_21_txt.text(0.05, 0.45, "• GARCH(1,1) & GJR-GARCH asymmetric leverage modeling.\n• Maximum Likelihood Estimation (MLE) of persistence (alpha + beta).\n• Mean-reverting term structure to unconditional variance sigma_L^2.\n• Multi-step forward volatility forecasting for risk & pricing.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Card 22 (Yield Curve)
    ax_22 = fig.add_subplot(gs[2, 0])
    plot_yield_curve(
        maturities=yield_data["maturities"],
        par_yields=yield_data["par_yields"],
        tenor_labels=yield_data.get("tenor_labels"),
        ax=ax_22,
    )
    ax_22_txt = fig.add_subplot(gs[2, 1])
    ax_22_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_22_txt.axis("off")
    ax_22_txt.text(0.05, 0.75, "22 | Yield Curve Term Structure Modeling", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_22_txt.text(0.05, 0.45, "• Nelson-Siegel & Nelson-Siegel-Svensson (NSS) calibration.\n• Decomposition into Level (Beta_0), Slope (Beta_1), and Curvature (Beta_2).\n• Zero-coupon bond bootstrapping & instantaneous forward rates.\n• Key for sovereign rates trading, duration hedging & bond pricing.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Card 23 (Kalman Filter)
    ax_23 = fig.add_subplot(gs[3, 0])
    plot_kalman_hedge_ratio(
        time_steps=kalman_data["time_steps"],
        hedge_ratios=kalman_data["hedge_ratios"],
        ax=ax_23,
    )
    ax_23_txt = fig.add_subplot(gs[3, 1])
    ax_23_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_23_txt.axis("off")
    ax_23_txt.text(0.05, 0.75, "23 | Kalman Filter for Dynamic Pairs Trading", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_23_txt.text(0.05, 0.45, "• Online state-space recursive estimation of dynamic beta_t.\n• Real-time adaptation to structural breaks and regime shifts.\n• Standardized Kalman innovation Z-score (e_t / sqrt(Q_t)).\n• More adaptive and profitable than static OLS hedge ratios.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Card 24 (ML Return Predictor)
    ax_24 = fig.add_subplot(gs[4, 0])
    plot_ml_predicted_returns(
        actual_returns=ml_data["actual_returns"],
        predicted_returns=ml_data["predicted_returns"],
        ic=ml_data.get("ic", 0.084),
        ax=ax_24,
    )
    ax_24_txt = fig.add_subplot(gs[4, 1])
    ax_24_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_24_txt.axis("off")
    ax_24_txt.text(0.05, 0.75, "24 | Machine Learning Return Predictor", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_24_txt.text(0.05, 0.45, "• Feature engineering with Fractional Differencing (memory retention).\n• Regularized linear models & tree ensembles with Purged TimeSeries CV.\n• Out-of-sample Information Coefficient (IC) & Rank IC tracking.\n• Strict leakage prevention and overfitting controls.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Card 25 (Alternative Data)
    ax_25 = fig.add_subplot(gs[5, 0])
    plot_alternative_data_signal(
        dates=alt_data["dates"],
        signal_strength=alt_data["signal_strength"],
        forward_returns=alt_data["forward_returns"],
        ax=ax_25,
    )
    ax_25_txt = fig.add_subplot(gs[5, 1])
    ax_25_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_25_txt.axis("off")
    ax_25_txt.text(0.05, 0.75, "25 | Alternative Data Alpha Model", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_25_txt.text(0.05, 0.45, "• Ingestion of Sentiment, App/Web Traffic, and Satellite activity.\n• Factor neutralization (orthogonalization against Market/Momentum/Size).\n• Multi-horizon Information Coefficient (IC) decay curves.\n• Dollar-neutral long/short strategy execution with transaction costs.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 7. Philosophy Banner
    ax_phil = fig.add_subplot(gs[6, :])
    plot_advanced_quant_philosophy(ax_phil)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
