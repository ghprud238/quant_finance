"""
EU ETS Carbon Allowance Pricing, Fuel-Switching Parity & Green Bond Greenium Decomposition.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass
class FuelSwitchingResult:
    """Output container for thermal power generation spreads and carbon parity."""
    power_price_mwh: float
    gas_price_mwh: float
    coal_price_mwh: float
    carbon_price_tco2: float
    clean_spark_spread: float
    clean_dark_spread: float
    fuel_switch_parity_price: float
    merit_order_dominant_fuel: str
    gas_switching_incentive_eur: float


class CarbonAllowanceModel:
    """
    EU ETS Carbon Allowance (EUA) Pricing and Fuel-Switching Parity Model.
    Calculates thermal generation spreads (CSS vs CDS) and theoretical fuel-switching carbon price.
    """

    def __init__(
        self,
        efficiency_gas: float = 0.50,      # CCGT thermal efficiency (~50%)
        efficiency_coal: float = 0.38,     # Coal thermal efficiency (~38%)
        emission_factor_gas: float = 0.37, # tCO2 / MWh_e for natural gas
        emission_factor_coal: float = 0.95,# tCO2 / MWh_e for coal
    ):
        self.efficiency_gas = efficiency_gas
        self.efficiency_coal = efficiency_coal
        self.emission_factor_gas = emission_factor_gas
        self.emission_factor_coal = emission_factor_coal

    def evaluate_spreads(
        self,
        power_price_mwh: float,
        gas_price_mwh: float,
        coal_price_mwh: float,
        carbon_price_tco2: float,
    ) -> FuelSwitchingResult:
        """
        Compute Clean Spark Spread (Gas) and Clean Dark Spread (Coal),
        and find the theoretical carbon parity price P_switch where CSS == CDS.
        """
        # Clean Spark Spread (CCGT Gas): Power - (Gas / eta_gas) - EF_gas * Carbon
        css = power_price_mwh - (gas_price_mwh / self.efficiency_gas) - (self.emission_factor_gas * carbon_price_tco2)

        # Clean Dark Spread (Coal): Power - (Coal / eta_coal) - EF_coal * Carbon
        cds = power_price_mwh - (coal_price_mwh / self.efficiency_coal) - (self.emission_factor_coal * carbon_price_tco2)

        # Economic parity: CSS = CDS => P_switch = (Gas/eta_gas - Coal/eta_coal) / (EF_coal - EF_gas)
        delta_eff_cost = (gas_price_mwh / self.efficiency_gas) - (coal_price_mwh / self.efficiency_coal)
        delta_ef = self.emission_factor_coal - self.emission_factor_gas
        p_switch = delta_eff_cost / delta_ef

        dominant = "GAS_CCGT" if css >= cds else "COAL_THERMAL"
        gas_incentive = css - cds

        return FuelSwitchingResult(
            power_price_mwh=power_price_mwh,
            gas_price_mwh=gas_price_mwh,
            coal_price_mwh=coal_price_mwh,
            carbon_price_tco2=carbon_price_tco2,
            clean_spark_spread=float(css),
            clean_dark_spread=float(cds),
            fuel_switch_parity_price=float(p_switch),
            merit_order_dominant_fuel=dominant,
            gas_switching_incentive_eur=float(gas_incentive),
        )

    def simulate_mean_reverting_jump_diffusion(
        self,
        s0: float = 75.0,
        kappa: float = 0.85,
        theta: float = 80.0,
        sigma: float = 0.35,
        jump_intensity: float = 0.50,
        jump_mean: float = -0.10,
        jump_std: float = 0.15,
        horizon_years: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate carbon allowance price trajectories with regulatory jump shocks."""
        np.random.seed(seed)
        dt = horizon_years / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = s0

        for t in range(1, n_steps + 1):
            s_prev = paths[:, t-1]
            # Drift + Diffusion
            z = np.random.standard_normal(n_paths)
            drift = kappa * (theta - s_prev) * dt
            diffusion = sigma * s_prev * np.sqrt(dt) * z

            # Poisson jumps
            n_jumps = np.random.poisson(jump_intensity * dt, n_paths)
            jump_sizes = np.random.normal(jump_mean, jump_std, n_paths) * n_jumps
            paths[:, t] = np.maximum(5.0, s_prev + drift + diffusion + s_prev * jump_sizes)

        return paths


@dataclass
class GreeniumDecompositionResult:
    """Output container for Green Bond greenium breakdown."""
    issuer: str
    green_bond_yield: float
    vanilla_bond_yield: float
    raw_greenium_bps: float
    liquidity_adjustment_bps: float
    duration_adjustment_bps: float
    pure_fundamental_greenium_bps: float
    is_statistically_significant: bool


class GreenBondValuationEngine:
    """
    Evaluates sustainability premia (Greenium) across matched-pair twin bonds:
    Raw Greenium = y_vanilla - y_green (in bps).
    Decomposes spread into fundamental ESG premium vs liquidity/maturity biases.
    """

    @staticmethod
    def decompose_twin_pair(
        green_yield_pct: float,
        vanilla_yield_pct: float,
        green_bid_ask_bps: float,
        vanilla_bid_ask_bps: float,
        green_duration: float,
        vanilla_duration: float,
        yield_curve_slope_bps_per_year: float = 8.0,
        issuer: str = "Corporate",
    ) -> GreeniumDecompositionResult:
        """Decompose raw green bond spread into pure fundamental greenium."""
        raw_greenium_bps = (vanilla_yield_pct - green_yield_pct) * 100.0

        # Liquidity adjustment (vanilla bonds are typically more liquid)
        liq_adjustment = (vanilla_bid_ask_bps - green_bid_ask_bps) * 0.50

        # Duration mismatch adjustment
        dur_diff = vanilla_duration - green_duration
        dur_adjustment = dur_diff * yield_curve_slope_bps_per_year

        # Pure greenium
        pure_greenium = raw_greenium_bps - dur_adjustment + liq_adjustment
        is_sig = bool(pure_greenium > 1.0)

        return GreeniumDecompositionResult(
            issuer=issuer,
            green_bond_yield=green_yield_pct,
            vanilla_bond_yield=vanilla_yield_pct,
            raw_greenium_bps=float(raw_greenium_bps),
            liquidity_adjustment_bps=float(liq_adjustment),
            duration_adjustment_bps=float(dur_adjustment),
            pure_fundamental_greenium_bps=float(pure_greenium),
            is_statistically_significant=is_sig,
        )

    @classmethod
    def factor_attribution_regression(
        cls,
        pairs_dataframe: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Multivariate regression decomposing greenium across market factors:
        Greenium_i = alpha + beta1 * Liquidity_Diff + beta2 * Rating + beta3 * ESG_Score + eps.
        """
        y = pairs_dataframe["raw_greenium_bps"].values
        x_liq = pairs_dataframe["liquidity_diff_bps"].values
        x_esg = pairs_dataframe["esg_transparency_score"].values

        # Multi-factor regression
        x_mat = np.column_stack([np.ones(len(y)), x_liq, x_esg])
        beta = np.linalg.lstsq(x_mat, y, rcond=None)[0]

        y_pred = x_mat @ beta
        r2 = float(1.0 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))

        return {
            "Alpha_Base_Greenium_bps": float(beta[0]),
            "Beta_Liquidity_Spread": float(beta[1]),
            "Beta_ESG_Transparency": float(beta[2]),
            "R_Squared": r2,
            "Mean_Greenium_bps": float(np.mean(y)),
        }
