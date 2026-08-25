"""Visualizations for Frontier Quantitative AI, Advanced Math & Alternative Data (Projects 31-35).

Styled with a dark aesthetic, amber/gold accents, and clean statistical typography.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def plot_sec_semantic_drift(
    drift_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots SEC 10-K Semantic Drift Cross-Sectional Ranking (Project 31)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    sorted_df = drift_df.sort_values("Cosine_Drift_Total", ascending=False)
    tickers = sorted_df["Ticker"].tolist()
    drifts = (sorted_df["Cosine_Drift_Total"] * 100).tolist()

    y_pos = np.arange(len(tickers))
    colors = [COLOR_PALETTE["red"] if d > 12.0 else (COLOR_PALETTE["gold"] if d > 4.0 else COLOR_PALETTE["green"]) for d in drifts]

    bars = ax.barh(y_pos, drifts, color=colors, edgecolor=COLOR_PALETTE["border"], height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers, fontsize=8, fontweight="bold", color=COLOR_PALETTE["text_light"])
    ax.invert_yaxis()

    ax.set_title("31 | SEC 10-K SEMANTIC DRIFT (LAZY PRICES ALPHA)", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Year-over-Year Cosine Drift (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    for bar, d in zip(bars, drifts):
        ax.text(d + 0.5, bar.get_y() + bar.get_height() / 2, f"{d:.1f}%", va="center", color=COLOR_PALETTE["text_light"], fontsize=7.5, fontweight="bold")

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_heston_surface_3d(
    moneyness: np.ndarray,
    expiries: np.ndarray,
    iv_grid: np.ndarray,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Renders 3D Calibrated Heston Implied Volatility Surface (Project 32)."""
    apply_dark_theme()
    fig = plt.figure(figsize=(9, 6), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(COLOR_PALETTE["bg_main"])

    M, T = np.meshgrid(moneyness, expiries) if moneyness.ndim == 1 else (moneyness, expiries)

    surf = ax.plot_surface(
        M, T, iv_grid * 100,
        cmap="plasma",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )

    ax.set_title("32 | CALIBRATED HESTON 3D VOLATILITY SURFACE", color=COLOR_PALETTE["gold_bright"], fontsize=11, fontweight="bold", pad=15)
    ax.set_xlabel("Moneyness (K/S)", fontsize=8.5, color=COLOR_PALETTE["text_dim"], labelpad=8)
    ax.set_ylabel("Expiry T (Years)", fontsize=8.5, color=COLOR_PALETTE["text_dim"], labelpad=8)
    ax.set_zlabel("Implied Volatility (%)", fontsize=8.5, color=COLOR_PALETTE["text_dim"], labelpad=8)
    ax.tick_params(axis="both", labelsize=7.5)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label("Heston Model IV (%)", color=COLOR_PALETTE["text_dim"], fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_vpin_toxicity_timeline(
    timestamps: pd.DatetimeIndex,
    prices: np.ndarray,
    vpin_series: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots VPIN (Volume-Synchronized Probability of Toxicity) timeline (Project 33)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Top: Spot Price
    ax.plot(timestamps, prices, color="#e2e8f0", lw=1.3, label="Price")
    ax.set_ylabel("Spot Price ($)", color="#e2e8f0", fontsize=8)
    ax.tick_params(axis="y", labelcolor="#e2e8f0", labelsize=7.5)

    # Bottom: VPIN Series
    ax2.plot(timestamps, vpin_series * 100, color=COLOR_PALETTE["gold_bright"], lw=1.5, label="VPIN (%)")
    ax2.axhline(80.0, color=COLOR_PALETTE["red"], linestyle="--", lw=1.0, label="Flash Crash Alert (80%)")
    ax2.set_ylabel("VPIN Order Flow Toxicity (%)", color=COLOR_PALETTE["gold_bright"], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=7.5)
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    ax.set_title("33 | VPIN TOXICITY & FLASH-CRASH DETECTION", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_supply_chain_gnn_alpha(
    dates: pd.DatetimeIndex,
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Supply-Chain GNN Spillover Momentum alpha (Project 34)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, strategy_equity, color=COLOR_PALETTE["gold_bright"], lw=1.8, label="Supply-Chain GNN Strategy")
    ax.plot(dates, benchmark_equity, color="#64748b", lw=1.2, linestyle="--", label="Standalone Momentum")

    ax.set_title("34 | SUPPLY-CHAIN GNN SPILLOVER ALPHA", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Normalized Wealth Growth", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_wasserstein_dro_frontier(
    nominal_vols: np.ndarray,
    nominal_returns: np.ndarray,
    robust_vols: np.ndarray,
    robust_returns: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Nominal vs Wasserstein Robust Efficient Frontier (Project 35)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(nominal_vols * 100, nominal_returns * 100, color="#38bdf8", lw=1.5, linestyle="--", label="Nominal SAA Frontier")
    ax.plot(robust_vols * 100, robust_returns * 100, color=COLOR_PALETTE["gold_bright"], lw=2.2, label="Wasserstein Robust DRO Frontier (ε=0.015)")

    ax.set_title("35 | WASSERSTEIN DISTRIBUTIONALLY ROBUST FRONTIER", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Annualized Volatility (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Annualized Expected Return (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_master_frontier_infographic(
    drift_df: pd.DataFrame,
    vpin_data: Dict[str, Any],
    gnn_data: Dict[str, Any],
    dro_data: Dict[str, Any],
    output_path: str = "/working_dir/genai_advanced_quant/output/genai_advanced_quant_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module Frontier Quant AI & Math infographic."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(6, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 0.45], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "FRONTIER QUANTITATIVE AI & MATHEMATICAL ALPHA", color=COLOR_PALETTE["gold_bright"], fontsize=22, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "31-35 — GENERATIVE AI, STOCHASTIC CALCULUS & ALTERNATIVE DATA", color=COLOR_PALETTE["gold"], fontsize=13, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "SEC 10-K Semantic Drift, Heston FFT Calibration, VPIN Toxicity, GNN Supply Chain & Wasserstein DRO", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Project 31 (LLM Semantic Drift)
    ax_31 = fig.add_subplot(gs[1, 0])
    plot_sec_semantic_drift(drift_df, ax=ax_31)
    ax_31_txt = fig.add_subplot(gs[1, 1])
    ax_31_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_31_txt.axis("off")
    ax_31_txt.text(0.05, 0.75, "31 | Financial LLM & SEC 10-K Semantic Drift", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_31_txt.text(0.05, 0.42, "• Sublinear TF-IDF & dense embeddings of Risk Factors (Item 1A) and MD&A.\n• 'Lazy Prices' anomaly: high disclosure revisions indicate unpriced risk.\n• Loughran-McDonald domain financial sentiment & uncertainty scoring.\n• Dollar-neutral Long Low-Drift (Lazy) vs Short High-Drift alpha.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Project 32 (Heston FFT Calibration)
    ax_32_txt = fig.add_subplot(gs[2, :])
    ax_32_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_32_txt.axis("off")
    ax_32_txt.text(0.04, 0.82, "32 | Heston Stochastic Volatility & Carr-Madan FFT / COS Option Calibration", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_32_txt.text(0.04, 0.50, "• Heston (1993) SDE: dS_t = (r-q)S_t dt + sqrt(v_t) S_t dW_t^S, dv_t = kappa(theta - v_t)dt + xi sqrt(v_t) dW_t^v with correlation rho.\n• Stable characteristic function formulation (Albrecher et al. 2007) avoiding complex logarithm branch cuts.\n• Carr-Madan (1999) Fast Fourier Transform (FFT) and Fang-Oosterlee (2008) Fourier-Cosine (COS) sub-millisecond option pricing.\n• Full volatility surface inversion calibrating (v0, kappa, theta, xi, rho) subject to the Feller condition 2*kappa*theta > xi^2.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Project 33 (VPIN Flow Toxicity)
    ax_33 = fig.add_subplot(gs[3, 0])
    plot_vpin_toxicity_timeline(
        timestamps=vpin_data["timestamps"],
        prices=vpin_data["prices"],
        vpin_series=vpin_data["vpin_series"],
        ax=ax_33,
    )
    ax_33_txt = fig.add_subplot(gs[3, 1])
    ax_33_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_33_txt.axis("off")
    ax_33_txt.text(0.05, 0.75, "33 | Volume Synchronized Probability of Toxicity (VPIN)", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_33_txt.text(0.05, 0.42, "• Volume Clock discretization: slicing trade flow into constant-volume buckets.\n• Bulk Volume Classification (BVC) separating probabilistic buy/sell volume.\n• Real-time adverse selection index: VPIN = sum(|V_b - V_s|) / (N * V).\n• Early-warning indicator for flash crashes, liquidity dry-ups & toxic order flow.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Project 34 (Supply-Chain GNN)
    ax_34 = fig.add_subplot(gs[4, 0])
    plot_supply_chain_gnn_alpha(
        dates=gnn_data["dates"],
        strategy_equity=gnn_data["strategy_equity"],
        benchmark_equity=gnn_data["benchmark_equity"],
        ax=ax_34,
    )
    ax_34_txt = fig.add_subplot(gs[4, 1])
    ax_34_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_34_txt.axis("off")
    ax_34_txt.text(0.05, 0.75, "34 | Supply-Chain Knowledge Graph & GNN Spillover", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_34_txt.text(0.05, 0.42, "• Directed customer-supplier graph with revenue dependency adjacency A_ij.\n• Graph Convolutional Network (GCN) message-passing of earnings surprises.\n• Lead-lag spillover momentum: customer revisions propagate to suppliers with lag.\n• Network PageRank centrality and customer concentration risk (HHI).", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Project 35 (Wasserstein DRO)
    ax_35 = fig.add_subplot(gs[5, 0])
    plot_wasserstein_dro_frontier(
        nominal_vols=dro_data["nominal_vols"],
        nominal_returns=dro_data["nominal_returns"],
        robust_vols=dro_data["robust_vols"],
        robust_returns=dro_data["robust_returns"],
        ax=ax_35,
    )
    ax_35_txt = fig.add_subplot(gs[5, 1])
    ax_35_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_35_txt.axis("off")
    ax_35_txt.text(0.05, 0.75, "35 | Wasserstein Distributionally Robust Optimization", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_35_txt.text(0.05, 0.42, "• Min-max optimization over Wasserstein ambiguity ball of radius epsilon.\n• Exact convex dual reformulation: min (-w^T mu + gamma/2 w^T Sigma w + eps ||w||_p).\n• Robust Out-of-Sample stability mitigating Markowitz 'error maximization'.\n• Superior CAGR and drawdown control under severe regime shifts.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
