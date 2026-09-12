"""
Risk: NGFS Climate Scenario Stress Testing & Merton Structural Credit Migration.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import scipy.stats as stats


class NGFSClimateStressEngine:
    """Evaluates corporate EBITDA impairment, DCF Climate VaR, and stressed Merton default probabilities."""

    def __init__(self):
        # NGFS Scenario carbon prices ($/tCO2e) and physical hazard multipliers
        self.scenarios = {
            "Net Zero 2050": {"carbon_price_2030": 140.0, "carbon_price_2050": 350.0, "physical_mult": 1.10},
            "Delayed Transition": {"carbon_price_2030": 40.0, "carbon_price_2050": 450.0, "physical_mult": 1.35},
            "Current Policies (Hot House)": {"carbon_price_2030": 20.0, "carbon_price_2050": 35.0, "physical_mult": 3.10},
        }

    def evaluate_corporate_climate_var(
        self,
        company_name: str,
        market_cap_m: float,
        ebitda_m: float,
        scope1_t: float,
        scope2_t: float,
        scope3_t: float,
        pass_through_rate: float = 0.40,
        scenario_name: str = "Net Zero 2050",
        wacc: float = 0.08,
    ) -> Dict[str, Any]:
        """Compute DCF Climate Value-at-Risk % equity impairment under shadow carbon prices."""
        scen = self.scenarios.get(scenario_name, self.scenarios["Net Zero 2050"])
        carbon_price = scen["carbon_price_2030"]

        # Scope emissions liability (100% Scope 1, 100% Scope 2, 20% Scope 3)
        effective_emissions = scope1_t + scope2_t + 0.20 * scope3_t
        gross_carbon_cost_m = (effective_emissions * carbon_price) / 1e6
        net_carbon_cost_m = gross_carbon_cost_m * (1.0 - pass_through_rate)

        ebitda_hit_pct = min(net_carbon_cost_m / max(ebitda_m, 1e-6), 1.0)

        # DCF valuation impairment: Capitalized EBITDA compression
        present_value_cost_m = net_carbon_cost_m / wacc
        climate_var_pct = -min(present_value_cost_m / max(market_cap_m, 1e-6), 0.95)

        return {
            "company": company_name,
            "scenario": scenario_name,
            "carbon_price_usd": carbon_price,
            "ebitda_hit_pct": float(ebitda_hit_pct),
            "climate_var_pct": float(climate_var_pct),
            "annual_cost_m": float(net_carbon_cost_m),
        }

    def merton_stressed_default_probability(
        self,
        asset_value: float,
        debt_face_value: float,
        asset_volatility: float = 0.25,
        risk_free_rate: float = 0.04,
        time_years: float = 5.0,
        equity_impairment_pct: float = 0.20,
    ) -> Dict[str, float]:
        """Calculate base vs stressed Merton default probability under climate impairment."""
        # Base Distance to Default
        d2_base = (np.log(asset_value / debt_face_value) + (risk_free_rate - 0.5 * asset_volatility**2) * time_years) / (asset_volatility * np.sqrt(time_years))
        pd_base = stats.norm.cdf(-d2_base)

        # Stressed Asset Value
        stressed_assets = asset_value * (1.0 - abs(equity_impairment_pct) * 0.7)
        d2_stress = (np.log(stressed_assets / debt_face_value) + (risk_free_rate - 0.5 * asset_volatility**2) * time_years) / (asset_volatility * np.sqrt(time_years))
        pd_stress = stats.norm.cdf(-d2_stress)

        spread_widening_bps = max((pd_stress - pd_base) * 0.60 * 10000.0, 0.0)

        return {
            "pd_base_pct": float(pd_base * 100.0),
            "pd_stressed_pct": float(pd_stress * 100.0),
            "credit_spread_widening_bps": float(spread_widening_bps),
        }
