"""Visualizations for Global Macro AI, Crypto & Cross-Economy Sentiment (Projects 46-50).

Styled with a dark aesthetic, amber/gold accents, and clean statistical typography.
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
    "purple": "#a855f7",
}


def apply_dark_theme():
    """Applies global dark theme."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_central_bank_hawk_dove(
    dates: pd.DatetimeIndex,
    fed_score: np.ndarray,
    ecb_score: np.ndarray,
    rbi_score: np.ndarray,
    bcb_score: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Central Bank Hawk/Dove Monetary Policy Stance across DM and EM (Project 46)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, fed_score, color=COLOR_PALETTE["blue"], lw=1.8, label="US Federal Reserve (FOMC)")
    ax.plot(dates, ecb_score, color=COLOR_PALETTE["gold_bright"], lw=1.8, label="European Central Bank (ECB)")
    ax.plot(dates, rbi_score, color=COLOR_PALETTE["green"], lw=1.5, linestyle="--", label="Reserve Bank of India (RBI)")
    ax.plot(dates, bcb_score, color=COLOR_PALETTE["purple"], lw=1.5, linestyle=":", label="Banco Central do Brasil (BCB)")

    ax.axhline(0.0, color=COLOR_PALETTE["text_muted"], linestyle="-", lw=0.8)
    ax.axhline(0.3, color=COLOR_PALETTE["red"], linestyle="--", lw=0.8, alpha=0.6, label="Hawkish Threshold (+0.3)")
    ax.axhline(-0.3, color=COLOR_PALETTE["green"], linestyle="--", lw=0.8, alpha=0.6, label="Dovish Threshold (-0.3)")

    ax.set_title("46 | CENTRAL BANK HAWK/DOVE STANCE INDEX", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Hawk/Dove Score [-1.0, +1.0]", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylim(-1.0, 1.0)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.0)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_sovereign_spillover_matrix(
    countries: List[str],
    spillover_matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Diebold-Yilmaz Sovereign Risk Directional Spillover Heatmap (Project 47)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.0), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    im = ax.imshow(spillover_matrix, cmap="magma", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label("Directional Spillover (%)", color=COLOR_PALETTE["text_dim"], fontsize=7.5)
    cbar.ax.tick_params(labelsize=7.0)

    ax.set_xticks(np.arange(len(countries)))
    ax.set_yticks(np.arange(len(countries)))
    ax.set_xticklabels(countries, fontsize=8, color=COLOR_PALETTE["text_light"], fontweight="bold")
    ax.set_yticklabels(countries, fontsize=8, color=COLOR_PALETTE["text_light"], fontweight="bold")

    for i in range(len(countries)):
        for j in range(len(countries)):
            val = spillover_matrix[i, j]
            color = "white" if val > 15.0 else "#94a3b8"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=7.0, fontweight="bold")

    ax.set_title("47 | DIEBOLD-YILMAZ SOVEREIGN RISK SPILLOVER NETWORK", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Receiving Country j", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Transmitting Country i", fontsize=8, color=COLOR_PALETTE["text_dim"])

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_crypto_fear_greed_timeline(
    dates: pd.DatetimeIndex,
    fgi_series: np.ndarray,
    crypto_price: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Multi-Source Crypto Fear & Greed Index vs Asset Price (Project 48)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Left: Spot Price (log scale)
    ax.plot(dates, crypto_price, color="#e2e8f0", lw=1.6, label="BTC/USD Price")
    ax.set_ylabel("Price ($)", color="#e2e8f0", fontsize=8)
    ax.tick_params(axis="y", labelcolor="#e2e8f0", labelsize=7.5)
    ax.set_yscale("log")

    # Right: Fear & Greed Index (0 to 100)
    ax2.plot(dates, fgi_series, color=COLOR_PALETTE["gold_bright"], lw=1.4, label="Fear & Greed Index")
    ax2.axhline(75.0, color=COLOR_PALETTE["red"], linestyle="--", lw=0.8, label="Extreme Greed (>75)")
    ax2.axhline(25.0, color=COLOR_PALETTE["green"], linestyle="--", lw=0.8, label="Extreme Fear (<25)")
    ax2.set_ylabel("Fear & Greed Score", color=COLOR_PALETTE["gold_bright"], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=7.5)
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    ax.set_title("48 | MULTI-SOURCE CRYPTO FEAR & GREED SENTIMENT", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_fx_carry_equity_curve(
    dates: pd.DatetimeIndex,
    em_carry_equity: np.ndarray,
    dm_carry_equity: np.ndarray,
    benchmark_equity: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Cross-Economy Emerging & Developed FX Carry Trade Performance (Project 49)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, (em_carry_equity / em_carry_equity[0] - 1.0) * 100, color=COLOR_PALETTE["gold_bright"], lw=2.0, label="High-Yield EM Carry (BRL/MXN/INR)")
    ax.plot(dates, (dm_carry_equity / dm_carry_equity[0] - 1.0) * 100, color=COLOR_PALETTE["blue"], lw=1.5, label="G10 DM Carry (USD/EUR/JPY)")
    ax.plot(dates, (benchmark_equity / benchmark_equity[0] - 1.0) * 100, color="#64748b", lw=1.2, linestyle="--", label="Unhedged G10 FX Index")

    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle="--", lw=0.8)

    ax.set_title("49 | CROSS-ECONOMY FX CARRY & UIP FORWARD BIAS", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Cumulative Return (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_agentic_swarm_allocations(
    asset_names: List[str],
    macro_agent_weights: np.ndarray,
    crypto_agent_weights: np.ndarray,
    sentiment_agent_weights: np.ndarray,
    pm_final_weights: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Multi-Agent Committee Consensus & Final Portfolio Allocation (Project 50)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    x = np.arange(len(asset_names))
    width = 0.20

    ax.bar(x - 1.5 * width, macro_agent_weights * 100, width, color=COLOR_PALETTE["blue"], label="Macro Economist Agent")
    ax.bar(x - 0.5 * width, crypto_agent_weights * 100, width, color=COLOR_PALETTE["purple"], label="Crypto Microstructure Agent")
    ax.bar(x + 0.5 * width, sentiment_agent_weights * 100, width, color=COLOR_PALETTE["green"], label="Sentiment Alpha Agent")
    ax.bar(x + 1.5 * width, pm_final_weights * 100, width, color=COLOR_PALETTE["gold_bright"], edgecolor="white", lw=1.2, label="PM Committee Final Weight")

    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, fontsize=8, color=COLOR_PALETTE["text_light"], fontweight="bold")
    ax.set_title("50 | MULTI-AGENT COMMITTEE ASSET ALLOCATION", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Target Portfolio Allocation (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.0)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_master_macro_ai_infographic(
    cb_data: Dict[str, Any],
    spillover_data: Dict[str, Any],
    sentiment_data: Dict[str, Any],
    carry_data: Dict[str, Any],
    agent_data: Dict[str, Any],
    output_path: str = "/working_dir/macro_ai_cross_markets/output/macro_ai_cross_markets_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module Global Macro AI & Cross-Market infographic."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(6, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "GLOBAL MACRO AI, CRYPTO & CROSS-ECONOMY SENTIMENT", color=COLOR_PALETTE["gold_bright"], fontsize=22, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "46-50 — MULTILINGUAL CENTRAL BANK NLP, SOVEREIGN CONTAGION & AGENTIC SWARMS", color=COLOR_PALETTE["gold"], fontsize=13, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "Central Bank Hawk/Dove NLP, Diebold-Yilmaz Spillovers, Crypto Sentiment, FX Carry & Multi-Agent Swarms", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Project 46 (Central Bank NLP)
    ax_46 = fig.add_subplot(gs[1, 0])
    plot_central_bank_hawk_dove(
        dates=cb_data["dates"],
        fed_score=cb_data["fed_score"],
        ecb_score=cb_data["ecb_score"],
        rbi_score=cb_data["rbi_score"],
        bcb_score=cb_data["bcb_score"],
        ax=ax_46,
    )
    ax_46_txt = fig.add_subplot(gs[1, 1])
    ax_46_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_46_txt.axis("off")
    ax_46_txt.text(0.05, 0.75, "46 | Multilingual Central Bank Hawk/Dove NLP", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_46_txt.text(0.05, 0.42, "• Sublinear TF-IDF and domain lexicon scoring of DM & EM monetary policy.\n• Hawk-Dove Stance Score: H_t = (N_hawk - N_dove) / (N_hawk + N_dove).\n• Taylor rule gap residualization and predictive 2Y sovereign yield forecasting.\n• Covers Fed, ECB, BOJ, RBI (India), BCB (Brazil), and Banxico (Mexico).", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Project 47 (Sovereign Contagion & Spillovers)
    ax_47 = fig.add_subplot(gs[2, 0])
    plot_sovereign_spillover_matrix(
        countries=spillover_data["countries"],
        spillover_matrix=spillover_data["spillover_matrix"],
        ax=ax_47,
    )
    ax_47_txt = fig.add_subplot(gs[2, 1])
    ax_47_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_47_txt.axis("off")
    ax_47_txt.text(0.05, 0.75, "47 | Sovereign Contagion & Volatility Spillovers", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_47_txt.text(0.05, 0.42, "• Vector Autoregression (VAR) on DM and EM sovereign bond yield spreads.\n• Diebold-Yilmaz (2012) total spillover index & directional transmission matrix.\n• Identifies net risk transmitters vs net risk receivers across global debt.\n• Bivariate Clayton/Gumbel copulas modeling asymmetric crisis tail dependence.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Project 48 (Crypto & Macro Sentiment)
    ax_48 = fig.add_subplot(gs[3, 0])
    plot_crypto_fear_greed_timeline(
        dates=sentiment_data["dates"],
        fgi_series=sentiment_data["fgi_series"],
        crypto_price=sentiment_data["crypto_price"],
        ax=ax_48,
    )
    ax_48_txt = fig.add_subplot(gs[3, 1])
    ax_48_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_48_txt.axis("off")
    ax_48_txt.text(0.05, 0.75, "48 | Multi-Source News & Crypto Sentiment Engine", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_48_txt.text(0.05, 0.42, "• Multi-channel ingestion: Financial news wires, Crypto Twitter & Reddit.\n• Reconstructs Crypto Fear & Greed Index from 6 weighted micro components.\n• Aspect-Based Sentiment Analysis (ABSA) across Macro, Policy & On-Chain.\n• Evaluates lead-lag cross-correlation against asset returns for alpha generation.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Project 49 (FX Carry & UIP Parity)
    ax_49 = fig.add_subplot(gs[4, 0])
    plot_fx_carry_equity_curve(
        dates=carry_data["dates"],
        em_carry_equity=carry_data["em_carry_equity"],
        dm_carry_equity=carry_data["dm_carry_equity"],
        benchmark_equity=carry_data["benchmark_equity"],
        ax=ax_49,
    )
    ax_49_txt = fig.add_subplot(gs[4, 1])
    ax_49_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_49_txt.axis("off")
    ax_49_txt.text(0.05, 0.75, "49 | Cross-Economy FX Carry Trade & Vol Surface", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_49_txt.text(0.05, 0.42, "• Covered (CIP) & Uncovered (UIP) Interest Rate Parity arbitrage analytics.\n• Forward Rate Bias anomaly: Long High-Yield EM (BRL/MXN/INR), Short G10.\n• Malz (1997) FX Volatility Surface: ATM vol, 25-delta Risk Reversal & Butterfly.\n• Volatility-managed carry strategy with stop-loss crash protection.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Project 50 (Autonomous Multi-Agent Swarm)
    ax_50 = fig.add_subplot(gs[5, 0])
    plot_agentic_swarm_allocations(
        asset_names=agent_data["asset_names"],
        macro_agent_weights=agent_data["macro_agent_weights"],
        crypto_agent_weights=agent_data["crypto_agent_weights"],
        sentiment_agent_weights=agent_data["sentiment_agent_weights"],
        pm_final_weights=agent_data["pm_final_weights"],
        ax=ax_50,
    )
    ax_50_txt = fig.add_subplot(gs[5, 1])
    ax_50_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_50_txt.axis("off")
    ax_50_txt.text(0.05, 0.75, "50 | Autonomous Multi-Agent Macro Hedge Fund Swarm", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_50_txt.text(0.05, 0.42, "• Specialized LLM agents: Macro Economist, Crypto Microstructure & Sentiment.\n• Investment Committee PM Chair reconciles proposals via Black-Litterman blending.\n• Constrained portfolio optimization under strict drawdown & VaR risk limits.\n• Generates automated Investment Committee Memos & dynamic rebalancing.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
