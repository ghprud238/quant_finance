"""Visualizations for Climate Quantitative Finance & Carbon Markets (Projects 36-40).

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
    "cyan": "#06b6d4",
}


def apply_dark_theme():
    """Applies global dark theme."""
    plt.rcParams.update(DARK_THEME_STYLE)


def plot_fuel_switching_parity(
    carbon_prices: np.ndarray,
    clean_spark_spreads: np.ndarray,
    clean_dark_spreads: np.ndarray,
    switching_price: float,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots EU ETS Fuel-Switching Economics (Clean Spark vs Clean Dark Spread) (Project 36)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(carbon_prices, clean_spark_spreads, color=COLOR_PALETTE["cyan"], lw=1.8, label="Clean Spark Spread (Gas)")
    ax.plot(carbon_prices, clean_dark_spreads, color=COLOR_PALETTE["red"], lw=1.8, label="Clean Dark Spread (Coal)")

    ax.axvline(switching_price, color=COLOR_PALETTE["gold_bright"], linestyle="--", lw=1.5, label=f"Switching Parity (€{switching_price:.1f}/t)")
    ax.axhline(0, color=COLOR_PALETTE["text_muted"], linestyle=":", lw=0.8)

    ax.set_title("36 | EU ETS COAL-TO-GAS FUEL SWITCHING PARITY", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("EUA Carbon Allowance Price (€/tCO2)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Clean Generation Spread (€/MWh)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_greenium_term_structure(
    maturities: np.ndarray,
    twin_bond_greeniums: np.ndarray,
    fitted_ns_greenium: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Green Bond Greenium Term Structure & Matched Pairs (Project 37)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.scatter(maturities, twin_bond_greeniums, color=COLOR_PALETTE["green"], s=35, zorder=4, label="Twin Bond Matched Pairs")
    ax.plot(maturities, fitted_ns_greenium, color=COLOR_PALETTE["gold_bright"], lw=2.0, label="Nelson-Siegel Greenium Curve")

    ax.set_title("37 | GREEN BOND GREENIUM TERM STRUCTURE", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Maturity (Years)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_ylabel("Greenium Yield Premium (bps)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_climate_var_stress(
    sectors: List[str],
    orderly_var: List[float],
    disorderly_var: List[float],
    hot_house_var: List[float],
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Cross-Sector NGFS Climate VaR Stress Losses (Project 38)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(sectors))
    height = 0.25

    ax.barh(y_pos - height, orderly_var, height=height, color=COLOR_PALETTE["green"], label="Net Zero 2050 (Orderly)")
    ax.barh(y_pos, disorderly_var, height=height, color=COLOR_PALETTE["gold"], label="Delayed Transition (Disorderly)")
    ax.barh(y_pos + height, hot_house_var, height=height, color=COLOR_PALETTE["red"], label="Hot House World (Physical)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sectors, fontsize=8, color=COLOR_PALETTE["text_light"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_title("38 | NGFS CLIMATE VALUE-AT-RISK (CLIMATE VAR)", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Equity Value Impairment (%)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.axvline(0, color=COLOR_PALETTE["text_muted"], linestyle="-", lw=0.8)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.0)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_satellite_emissions_alpha(
    dates: pd.DatetimeIndex,
    strategy_equity: pd.Series,
    long_leg_equity: pd.Series,
    short_leg_equity: pd.Series,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Satellite Emissions Alternative Data Alpha Strategy (Project 39)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.plot(dates, strategy_equity, color=COLOR_PALETTE["gold_bright"], lw=1.8, label="Dollar-Neutral Alpha (L/S)")
    ax.plot(dates, long_leg_equity, color=COLOR_PALETTE["green"], lw=1.2, linestyle="--", label="Long Leg (Clean Leaders)")
    ax.plot(dates, short_leg_equity, color=COLOR_PALETTE["red"], lw=1.2, linestyle=":", label="Short Leg (Heavy Plumers)")

    ax.set_title("39 | SATELLITE GHG EMISSIONS ALPHA (L/S)", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("Normalized Growth", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", facecolor=COLOR_PALETTE["bg_card"], edgecolor=COLOR_PALETTE["border"], fontsize=7.5)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_ppa_cannibalization_curve(
    hours: np.ndarray,
    solar_generation: np.ndarray,
    market_spot_price: np.ndarray,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots Solar Merit-Order Duck Curve & Cannibalization Discount (Project 40)."""
    apply_dark_theme()
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()

    # Left: Solar Generation (Gold fill)
    l1 = ax.fill_between(hours, 0, solar_generation, color=COLOR_PALETTE["gold"], alpha=0.35, label="Solar Generation (MW)")
    ax.set_ylabel("Generation (MW)", color=COLOR_PALETTE["gold_bright"], fontsize=8)
    ax.tick_params(axis="y", labelcolor=COLOR_PALETTE["gold_bright"], labelsize=7.5)

    # Right: Spot Electricity Price (Duck Curve in Cyan)
    l2 = ax2.plot(hours, market_spot_price, color=COLOR_PALETTE["cyan"], lw=2.0, label="Spot Price ($/MWh Duck Curve)")
    ax2.set_ylabel("Power Price ($/MWh)", color=COLOR_PALETTE["cyan"], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["cyan"], labelsize=7.5)
    ax2.grid(False)

    ax.set_title("40 | RENEWABLE PPA DUCK CURVE & CANNIBALIZATION", color=COLOR_PALETTE["gold_bright"], fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Hour of Day (0 to 24)", fontsize=8, color=COLOR_PALETTE["text_dim"])
    ax.set_xticks(np.arange(0, 25, 4))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path and created_fig:
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig if created_fig else None


def plot_master_climate_infographic(
    fuel_data: Dict[str, Any],
    greenium_data: Dict[str, Any],
    stress_data: Dict[str, Any],
    satellite_data: Dict[str, Any],
    ppa_data: Dict[str, Any],
    output_path: str = "/working_dir/climate_quant_finance/output/climate_quant_infographic.png",
) -> plt.Figure:
    """Composites the full 5-module Climate Quantitative Finance infographic."""
    apply_dark_theme()
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor=COLOR_PALETTE["bg_main"])
    gs = fig.add_gridspec(6, 2, height_ratios=[0.4, 1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.38, wspace=0.25)

    # 1. Header
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor(COLOR_PALETTE["bg_main"])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.75, "CLIMATE QUANTITATIVE FINANCE & CARBON MARKETS", color=COLOR_PALETTE["gold_bright"], fontsize=22, fontweight="bold")
    ax_hdr.text(0.0, 0.40, "36-40 — SUSTAINABILITY, EMISSIONS ALPHA & CLIMATE RISK MODELING", color=COLOR_PALETTE["gold"], fontsize=13, fontweight="bold")
    ax_hdr.text(0.0, 0.12, "EU ETS Carbon Pricing, Green Bond Greenium, NGFS Climate Stress, Satellite Plume Alpha & Renewable PPAs", color=COLOR_PALETTE["text_dim"], fontsize=10)

    # 2. Project 36 (Carbon Allowance Pricing)
    ax_36 = fig.add_subplot(gs[1, 0])
    plot_fuel_switching_parity(
        carbon_prices=fuel_data["carbon_prices"],
        clean_spark_spreads=fuel_data["clean_spark_spreads"],
        clean_dark_spreads=fuel_data["clean_dark_spreads"],
        switching_price=fuel_data["switching_price"],
        ax=ax_36,
    )
    ax_36_txt = fig.add_subplot(gs[1, 1])
    ax_36_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_36_txt.axis("off")
    ax_36_txt.text(0.05, 0.75, "36 | Carbon Allowance Pricing & ETS Fuel-Switching", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_36_txt.text(0.05, 0.42, "• Clean Spark (Gas) vs Clean Dark (Coal) spread economic parity.\n• Theoretical fuel-switching price P_switch = (P_gas/eta_g - P_coal/eta_c)/(EF_c - EF_g).\n• Mean-reverting jump-diffusion carbon price simulation with policy jumps.\n• Cost-of-carry futures curve with compliance convenience yield.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 3. Project 37 (Green Bond Greenium)
    ax_37 = fig.add_subplot(gs[2, 0])
    plot_greenium_term_structure(
        maturities=greenium_data["maturities"],
        twin_bond_greeniums=greenium_data["twin_bond_greeniums"],
        fitted_ns_greenium=greenium_data["fitted_ns_greenium"],
        ax=ax_37,
    )
    ax_37_txt = fig.add_subplot(gs[2, 1])
    ax_37_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_37_txt.axis("off")
    ax_37_txt.text(0.05, 0.75, "37 | Green Bond Valuation & Greenium Decomposition", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_37_txt.text(0.05, 0.42, "• Matched-pair green vs conventional twin bond yield spread analysis.\n• Multi-factor decomposition: Liquidity spread, Credit rating, and ESG disclosure.\n• Nelson-Siegel parametric term structure modeling of the Greenium.\n• Quantifies institutional willingness-to-pay for verifiable green use-of-proceeds.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 4. Project 38 (NGFS Climate Stress)
    ax_38 = fig.add_subplot(gs[3, 0])
    plot_climate_var_stress(
        sectors=stress_data["sectors"],
        orderly_var=stress_data["orderly_var"],
        disorderly_var=stress_data["disorderly_var"],
        hot_house_var=stress_data["hot_house_var"],
        ax=ax_38,
    )
    ax_38_txt = fig.add_subplot(gs[3, 1])
    ax_38_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_38_txt.axis("off")
    ax_38_txt.text(0.05, 0.75, "38 | NGFS Climate Scenario Stress Testing", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_38_txt.text(0.05, 0.42, "• NGFS Phase IV pathways: Net Zero 2050, Delayed Transition, and Hot House World.\n• Transition Risk: Scope 1, 2, 3 carbon tax EBITDA impairment & pass-through drag.\n• Physical Risk: Geospatial hazard damage functions across acute/chronic perils.\n• Merton structural credit rating migration & spread widening under Climate VaR.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 5. Project 39 (Satellite Emissions Alpha)
    ax_39 = fig.add_subplot(gs[4, 0])
    plot_satellite_emissions_alpha(
        dates=satellite_data["dates"],
        strategy_equity=satellite_data["strategy_equity"],
        long_leg_equity=satellite_data["long_leg_equity"],
        short_leg_equity=satellite_data["short_leg_equity"],
        ax=ax_39,
    )
    ax_39_txt = fig.add_subplot(gs[4, 1])
    ax_39_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_39_txt.axis("off")
    ax_39_txt.text(0.05, 0.75, "39 | Geospatial & Satellite Emissions Alpha Model", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_39_txt.text(0.05, 0.42, "• Satellite plume telemetry (Sentinel-5P/GHGSat) vs self-reported disclosures.\n• Cross-sectional Emissions Surprise Z-Score: Z = (Measured - Disclosed) / sigma.\n• Dollar-neutral equity strategy: Long Clean Abaters vs Short High-Plume Disclosers.\n• Generates market-neutral alpha net of transaction costs & turnover friction.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    # 6. Project 40 (Renewable PPA & Weather Derivatives)
    ax_40 = fig.add_subplot(gs[5, 0])
    plot_ppa_cannibalization_curve(
        hours=ppa_data["hours"],
        solar_generation=ppa_data["solar_generation"],
        market_spot_price=ppa_data["market_spot_price"],
        ax=ax_40,
    )
    ax_40_txt = fig.add_subplot(gs[5, 1])
    ax_40_txt.set_facecolor(COLOR_PALETTE["bg_card"])
    ax_40_txt.axis("off")
    ax_40_txt.text(0.05, 0.75, "40 | Renewable Energy PPA Valuation & Weather Derivatives", color=COLOR_PALETTE["gold_bright"], fontsize=12, fontweight="bold")
    ax_40_txt.text(0.05, 0.42, "• Hourly Weibull wind & diurnal solar yield generation simulation (8,760h).\n• Capture price & merit-order cannibalization discount during peak sun (Duck Curve).\n• Valuation of Pay-As-Produced (PAP) vs Baseload PPAs with shaping risk.\n• Heating/Cooling Degree Day (HDD/CDD) Weather Swaps & Options pricing via Burn/MC.", color=COLOR_PALETTE["text_light"], fontsize=9.5, linespacing=1.6)

    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig
