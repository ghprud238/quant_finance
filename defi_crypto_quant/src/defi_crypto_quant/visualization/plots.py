"""Visualizations for DeFi, AMM Liquidity & Crypto Quantitative Finance (Projects 41-45).

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


def plot_v3_liquidity_density(
    prices: np.ndarray,
    v2_liquidity: np.ndarray,
    v3_liquidity: np.ndarray,
    current_price: float = 3000.0,
    price_lower: float = 2500.0,
    price_upper: float = 3500.0,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Uniswap v3 Concentrated Liquidity Density vs v2 (Project 41)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(prices, v2_liquidity, color="#64748b", lw=1.5, linestyle="--", label="Uniswap v2 Constant Product (Uniform)")
    ax.fill_between(prices, 0, v3_liquidity, color=COLOR_PALETTE["gold"], alpha=0.35, label="Uniswap v3 Concentrated Range")
    ax.plot(prices, v3_liquidity, color=COLOR_PALETTE["gold_bright"], lw=2.0)

    ax.axvline(current_price, color="#38bdf8", linestyle="-", lw=1.2, label=f"Current Spot Price (${current_price:,.0f})")
    ax.axvline(price_lower, color=COLOR_PALETTE["green"], linestyle=":", lw=1.0, label=f"Range Lower (${price_lower:,.0f})")
    ax.axvline(price_upper, color=COLOR_PALETTE["red"], linestyle=":", lw=1.0, label=f"Range Upper (${price_upper:,.0f})")

    ax.set_title("41 | UNISWAP V3 CONCENTRATED LIQUIDITY DENSITY", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Pool Price ($/ETH)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Effective Depth ($/%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.0)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_lvr_vs_fee_revenue(
    timestamps: pd.DatetimeIndex,
    cumulative_fees: np.ndarray,
    cumulative_lvr: np.ndarray,
    net_lp_pnl: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots LVR (Loss-Versus-Rebalancing) vs Fee Revenue & Net LP P&L (Project 42)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(timestamps, cumulative_fees, color=COLOR_PALETTE["green"], lw=1.6, label="Cumulative Swap Fees")
    ax.plot(timestamps, cumulative_lvr, color=COLOR_PALETTE["red"], lw=1.6, label="Cumulative LVR Adverse Selection")
    ax.plot(timestamps, net_lp_pnl, color=COLOR_PALETTE["gold_bright"], lw=2.0, label="Net LP P&L (Fees - LVR)")

    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle="--", lw=0.8)

    ax.set_title("42 | LOSS-VERSUS-REBALANCING (LVR) & LP PROFITABILITY", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("P&L ($ / $100k Capital)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_mev_sandwich_dynamics(
    stages: List[str],
    pool_prices: List[float],
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots 4-Stage Price Trajectory during Atomic MEV Sandwich Attack (Project 43)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    x_pos = np.arange(len(stages))
    ax.plot(x_pos, pool_prices, color=COLOR_PALETTE["gold_bright"], marker="o", markersize=8, lw=2.0, label="AMM Marginal Price")

    for i, (s, p) in enumerate(zip(stages, pool_prices)):
        ax.text(i, p + 8, f"${p:,.1f}", color=COLOR_PALETTE["text_light"], fontsize=8, fontweight="bold", ha="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, fontsize=7.5, color=COLOR_PALETTE["text_light"], fontweight="bold")
    ax.set_title("43 | ATOMIC MEV SANDWICH ATTACK DYNAMICS", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Pool Price ($/WETH)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_perp_funding_basis_equity(
    dates: pd.DatetimeIndex,
    basis_equity: np.ndarray,
    spot_price: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Cash-and-Carry Delta-Neutral Basis Trading Equity vs Spot (Project 44)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Left: Delta-Neutral Basis Strategy Equity (Gold compounding)
    norm_basis = (basis_equity / basis_equity[0] - 1.0) * 100
    ax.plot(dates, norm_basis, color=COLOR_PALETTE["gold_bright"], lw=2.0, label="Delta-Neutral Basis Return (%)")
    ax.set_ylabel("Basis Trading Return (%)", color=COLOR_PALETTE["gold_bright"], fontsize=8)
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=7.5)

    # Right: Underlying Spot Asset Price (Cyan)
    norm_spot = (spot_price / spot_price[0] - 1.0) * 100
    ax2.plot(dates, norm_spot, color="#38bdf8", lw=1.2, alpha=0.6, linestyle="--", label="Spot Asset Volatility (%)")
    ax2.set_ylabel("Spot Return (%)", color="#38bdf8", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#38bdf8", labelsize=7.5)
    ax2.grid(False)

    ax.set_title("44 | PERPETUAL FUNDING CASH-AND-CARRY BASIS TRADING", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_onchain_mvrv_regimes(
    dates: pd.DatetimeIndex,
    spot_price: np.ndarray,
    mvrv_z_score: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots On-Chain MVRV Z-Score & Market Cycle Regimes (Project 45)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Left: Spot Price (log scale)
    ax.plot(dates, spot_price, color="#e2e8f0", lw=1.5, label="BTC/ETH Spot Price")
    ax.set_ylabel("Spot Price ($)", color="#e2e8f0", fontsize=8)
    ax.tick_params(axis="y", labelcolor="#e2e8f0", labelsize=7.5)
    ax.set_yscale("log")

    # Right: MVRV Z-Score
    ax2.plot(dates, mvrv_z_score, color=COLOR_PALETTE["gold_bright"], lw=1.3, label="MVRV Z-Score")
    ax2.axhline(4.0, color=COLOR_PALETTE["red"], linestyle="--", lw=1.0, label="Overheated Euphoria (Z > 4.0)")
    ax2.axhline(0.0, color=COLOR_PALETTE["green"], linestyle="--", lw=1.0, label="Capitulation Accumulation (Z < 0.0)")
    ax2.set_ylabel("MVRV Z-Score", color=COLOR_PALETTE["gold_bright"], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=7.5)
    ax2.grid(False)

    ax.set_title("45 | ON-CHAIN MVRV Z-SCORE & CYCLE REGIMES", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_master_defi_crypto_infographic(
    amm_data: Dict[str, Any],
    lvr_data: Dict[str, Any],
    mev_data: Dict[str, Any],
    perp_data: Dict[str, Any],
    onchain_data: Dict[str, Any],
    output_path: str = "/working_dir/defi_crypto_quant/output/defi_crypto_quant_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module DeFi & Crypto Quant infographic."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(6, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "DECENTRALIZED FINANCE & CRYPTO QUANTITATIVE FINANCE", color=COLOR_PALETTE["gold_bright"], fontsize=22, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "41-45 — AMM LIQUIDITY, LVR, MEV ARBITRAGE, PERP FUNDING & ON-CHAIN ALPHA", color=COLOR_PALETTE["gold"], fontsize=13, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "Uniswap v3 Virtual Reserves, Loss-Versus-Rebalancing, Flash Loan MEV, Basis Yield & MVRV Regimes", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Project 41 (Uniswap v3 AMM)
    ax_41 = fig.add_subplot(gs[1, 0])
    plot_v3_liquidity_density(
        prices=amm_data["prices"],
        v2_liquidity=amm_data["v2_liquidity"],
        v3_liquidity=amm_data["v3_liquidity"],
        current_price=amm_data.get("current_price", 3000.0),
        ax=ax_41,
    )
    ax_41_txt = fig.add_subplot(gs[1, 1])
    ax_41_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_41_txt.axis("off")
    ax_41_txt.text(0.05, 0.75, "41 | CFMMs & Uniswap v3 Concentrated Liquidity", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_41_txt.text(0.05, 0.42, "• Virtual reserves invariant (x + L/sqrt(P_b))(y + L*sqrt(P_a)) = L^2.\n• Tick pricing P(i) = 1.0001^i with step-wise multi-tick crossing.\n• Capital efficiency multiplier: up to 40x depth concentration vs v2.\n• Exact token input/output fee routing across fee tiers (0.05%, 0.30%, 1.00%).", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Project 42 (LVR & Impermanent Loss)
    ax_42 = fig.add_subplot(gs[2, 0])
    plot_lvr_vs_fee_revenue(
        timestamps=lvr_data["timestamps"],
        cumulative_fees=lvr_data["cumulative_fees"],
        cumulative_lvr=lvr_data["cumulative_lvr"],
        net_lp_pnl=lvr_data["net_lp_pnl"],
        ax=ax_42,
    )
    ax_42_txt = fig.add_subplot(gs[2, 1])
    ax_42_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_42_txt.axis("off")
    ax_42_txt.text(0.05, 0.75, "42 | Impermanent Loss & Loss-Versus-Rebalancing (LVR)", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_42_txt.text(0.05, 0.42, "• Impermanent loss formula IL(k) = 2*sqrt(k)/(1+k) - 1.\n• Continuous-time LVR = integral(sigma^2/8 * S_t * L_t dt) (Milionis et al. 2022).\n• Decomposes passive LP returns into Fee Revenue minus LVR adverse selection.\n• Determines breakeven volatility sigma_break and minimum required fee rates.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Project 43 (MEV & Spatial Arbitrage)
    ax_43 = fig.add_subplot(gs[3, 0])
    plot_mev_sandwich_dynamics(
        stages=mev_data["stages"],
        pool_prices=mev_data["pool_prices"],
        ax=ax_43,
    )
    ax_43_txt = fig.add_subplot(gs[3, 1])
    ax_43_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_43_txt.axis("off")
    ax_43_txt.text(0.05, 0.75, "43 | Cross-DEX Flash Loans & MEV Searcher Engine", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_43_txt.text(0.05, 0.42, "• Closed-form optimal flash loan size Delta_x* for multi-venue spatial arbitrage.\n• Bellman-Ford negative-cycle search on -ln(R_ij) for triangular arbitrage.\n• Mempool atomic sandwich simulation: frontrun -> victim -> backrun execution.\n• Priority gas bidding and Flashbots builder bribe optimization (85% bribe).", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Project 44 (Perp Funding Basis Trading)
    ax_44 = fig.add_subplot(gs[4, 0])
    plot_perp_funding_basis_equity(
        dates=perp_data["dates"],
        basis_equity=perp_data["basis_equity"],
        spot_price=perp_data["spot_price"],
        ax=ax_44,
    )
    ax_44_txt = fig.add_subplot(gs[4, 1])
    ax_44_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_44_txt.axis("off")
    ax_44_txt.text(0.05, 0.75, "44 | Perpetual Funding Arbitrage & Basis Trading", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_44_txt.text(0.05, 0.42, "• 8-Hour funding rate clamping mechanism: Clamp(Premium Index + Interest Rate).\n• Cash-and-carry delta-neutral strategy: Long Spot + Short Perpetual Futures.\n• Captures double-digit annualized funding yields (APY) with low volatility.\n• Liquidation buffers, maintenance margins & flash crash margin call monitoring.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Project 45 (On-Chain Telemetry Alpha)
    ax_45 = fig.add_subplot(gs[5, 0])
    plot_onchain_mvrv_regimes(
        dates=onchain_data["dates"],
        spot_price=onchain_data["spot_price"],
        mvrv_z_score=onchain_data["mvrv_z_score"],
        ax=ax_45,
    )
    ax_45_txt = fig.add_subplot(gs[5, 1])
    ax_45_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_45_txt.axis("off")
    ax_45_txt.text(0.05, 0.75, "45 | On-Chain Blockchain Telemetry & Whale Alpha", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_45_txt.text(0.05, 0.42, "• MVRV Ratio & Z-Score = (Market Cap - Realized Cap) / sigma_Cap.\n• Exchange Flow Imbalance (EFI) & Whale Wallet (>= 1,000 BTC) accumulation.\n• Network Value to Transactions (NVT) and Active Address Velocity.\n• Multi-factor regime classifier: Accumulation Bottom vs Overheated Euphoria.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
