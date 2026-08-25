"""NGFS Climate Scenario Stress Testing & Physical/Transition Risk Engine (Project 38).

Implements Network for Greening the Financial System (NGFS) climate scenario pathways,
micro-level corporate transition cost models (Scope 1/2/3 pass-through elasticity),
geospatial acute/chronic physical damage functions, Climate Value-at-Risk (Climate VaR),
and Merton structural default model credit migration under asset impairment.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm


class NGFSScenarioType(Enum):
    """Core NGFS Phase IV Scenario Framework."""
    NET_ZERO_2050 = "Net Zero 2050 (Orderly)"
    DELAYED_TRANSITION = "Delayed Transition (Disorderly)"
    CURRENT_POLICIES = "Current Policies / Hot House World"


@dataclass
class NGFSScenario:
    """Quantitative macro-financial and climate parameters for an NGFS scenario."""
    scenario_type: NGFSScenarioType
    description: str
    carbon_price_trajectory: Dict[int, float]  # Year -> USD / tCO2e
    physical_hazard_multiplier: Dict[int, float]  # Year -> Physical damage multiplier relative to base
    macro_gdp_shock_pct: Dict[int, float]  # Year -> Macroeconomic GDP contraction %
    interest_rate_shock_bps: Dict[int, float]  # Year -> Interest rate delta in bps


@dataclass
class CompanyClimateProfile:
    """Corporate micro-level financial, operational, and emissions profile."""
    ticker: str
    name: str
    sector: str
    market_cap: float  # USD Millions
    total_debt: float  # USD Millions
    cash: float  # USD Millions
    ebitda: float  # USD Millions (Annual)
    asset_volatility: float  # Annual asset return volatility (sigma_V)
    scope1_emissions_t: float  # Scope 1 Direct GHG (tCO2e / year)
    scope2_emissions_t: float  # Scope 2 Indirect Electricity (tCO2e / year)
    scope3_emissions_t: float  # Scope 3 Supply Chain / Value Chain (tCO2e / year)
    carbon_pass_through_rate: float = 0.50  # Fraction (0 to 1) of carbon cost passed to customers
    scope3_liability_weight: float = 0.20  # Fraction (0 to 1) of Scope 3 subject to carbon pricing
    physical_hazard_exposure: float = 0.30  # Vulnerability score (0 to 1) to floods, heat, wildfires
    asset_replacement_cost: float = 1000.0  # USD Millions (Total fixed physical asset base)
    cost_of_capital: float = 0.08  # WACC for DCF Climate VaR valuation
    risk_free_rate: float = 0.04  # Risk-free rate for structural Merton model
    lgd: float = 0.45  # Loss Given Default for credit spread calculations

    @property
    def enterprise_value(self) -> float:
        return self.market_cap + self.total_debt - self.cash

    @property
    def total_assets_book(self) -> float:
        return self.market_cap + self.total_debt

    @property
    def total_emissions_t(self) -> float:
        return self.scope1_emissions_t + self.scope2_emissions_t + self.scope3_emissions_t

    @property
    def carbon_intensity_revenue(self) -> float:
        return self.total_emissions_t / max(1.0, self.ebitda)


@dataclass
class ClimateStressResult:
    """Granular stress testing output for a single firm under a specific NGFS scenario."""
    ticker: str
    scenario_name: str
    horizon_year: int
    carbon_price_usd: float
    annual_transition_cost_usd: float
    annual_physical_damage_usd: float
    baseline_ebitda: float
    stressed_ebitda: float
    ebitda_impairment_pct: float
    baseline_equity_val: float
    stressed_equity_val: float
    climate_equity_var_pct: float  # Percentage change in equity valuation (Climate VaR)
    baseline_pd: float  # Merton baseline default probability (1-Year)
    stressed_pd: float  # Merton stressed default probability (1-Year)
    credit_spread_widening_bps: float  # Delta in credit spread (bps)


@dataclass
class PortfolioClimateStressReport:
    """Aggregated portfolio-level climate risk and scenario analysis."""
    scenario_name: str
    horizon_year: int
    n_companies: int
    total_portfolio_value: float
    stressed_portfolio_value: float
    portfolio_climate_var_pct: float
    total_annual_transition_cost: float
    total_annual_physical_damage: float
    weighted_baseline_pd: float
    weighted_stressed_pd: float
    weighted_spread_widening_bps: float
    company_results: List[ClimateStressResult]

    def summary_table(self) -> pd.DataFrame:
        """Returns structured DataFrame of company-level stress impacts."""
        records = []
        for r in self.company_results:
            records.append({
                "Ticker": r.ticker,
                "Scenario": r.scenario_name,
                "Horizon": r.horizon_year,
                "Carbon_Price": f"${r.carbon_price_usd:.0f}/t",
                "Baseline_EBITDA": f"${r.baseline_ebitda:,.1f}M",
                "Stressed_EBITDA": f"${r.stressed_ebitda:,.1f}M",
                "EBITDA_Impact_%": f"{r.ebitda_impairment_pct:+.1%}",
                "Climate_VaR_%": f"{r.climate_equity_var_pct:+.1%}",
                "Baseline_PD": f"{r.baseline_pd:.2%}",
                "Stressed_PD": f"{r.stressed_pd:.2%}",
                "Spread_Delta_bps": f"{r.credit_spread_widening_bps:+.0f} bps",
            })
        return pd.DataFrame(records)


class NGFSClimateStressEngine:
    """Industrial NGFS Climate Stress Testing Engine."""

    def __init__(self, scenarios: Optional[Dict[NGFSScenarioType, NGFSScenario]] = None):
        self.scenarios = scenarios or self._default_ngfs_scenarios()

    @staticmethod
    def _default_ngfs_scenarios() -> Dict[NGFSScenarioType, NGFSScenario]:
        """Builds standard NGFS Phase IV representative scenario pathways."""
        net_zero = NGFSScenario(
            scenario_type=NGFSScenarioType.NET_ZERO_2050,
            description="Orderly 1.5C transition with ambitious early policy, high shadow carbon price, minimal physical warming.",
            carbon_price_trajectory={2025: 50.0, 2030: 140.0, 2035: 200.0, 2040: 280.0, 2050: 350.0},
            physical_hazard_multiplier={2025: 1.05, 2030: 1.10, 2035: 1.15, 2040: 1.20, 2050: 1.25},
            macro_gdp_shock_pct={2025: -0.5, 2030: -1.2, 2035: -1.5, 2040: -1.8, 2050: -2.0},
            interest_rate_shock_bps={2025: 25.0, 2030: 50.0, 2035: 50.0, 2040: 25.0, 2050: 0.0},
        )

        delayed = NGFSScenario(
            scenario_type=NGFSScenarioType.DELAYED_TRANSITION,
            description="Disorderly transition: late policy action after 2030 triggers abrupt, punitive carbon price shock and stranded assets.",
            carbon_price_trajectory={2025: 20.0, 2030: 40.0, 2035: 280.0, 2040: 380.0, 2050: 450.0},
            physical_hazard_multiplier={2025: 1.10, 2030: 1.25, 2035: 1.45, 2040: 1.65, 2050: 1.85},
            macro_gdp_shock_pct={2025: -0.2, 2030: -0.8, 2035: -4.5, 2040: -5.8, 2050: -6.5},
            interest_rate_shock_bps={2025: 0.0, 2030: 25.0, 2035: 150.0, 2040: 100.0, 2050: 50.0},
        )

        hot_house = NGFSScenario(
            scenario_type=NGFSScenarioType.CURRENT_POLICIES,
            description="Hot House World: no new climate policies, low carbon pricing, but severe chronic warming and extreme acute weather damages.",
            carbon_price_trajectory={2025: 15.0, 2030: 20.0, 2035: 25.0, 2040: 30.0, 2050: 35.0},
            physical_hazard_multiplier={2025: 1.20, 2030: 1.50, 2035: 1.90, 2040: 2.35, 2050: 3.10},
            macro_gdp_shock_pct={2025: -0.3, 2030: -1.5, 2035: -3.8, 2040: -7.2, 2050: -14.5},
            interest_rate_shock_bps={2025: 0.0, 2030: 0.0, 2035: -25.0, 2040: -50.0, 2050: -100.0},
        )

        return {
            NGFSScenarioType.NET_ZERO_2050: net_zero,
            NGFSScenarioType.DELAYED_TRANSITION: delayed,
            NGFSScenarioType.CURRENT_POLICIES: hot_house,
        }

    def evaluate_transition_risk(
        self,
        company: CompanyClimateProfile,
        scenario: NGFSScenario,
        year: int = 2030,
    ) -> Dict[str, float]:
        """Calculates transition carbon cost liability under shadow carbon price and pass-through elasticity."""
        carbon_price = scenario.carbon_price_trajectory.get(year, 100.0)

        effective_emissions_t = (
            company.scope1_emissions_t +
            company.scope2_emissions_t +
            company.scope3_liability_weight * company.scope3_emissions_t
        )

        gross_carbon_cost_m = (carbon_price * effective_emissions_t) / 1e6
        unpassed_fraction = max(0.0, 1.0 - company.carbon_pass_through_rate)
        net_carbon_cost_m = gross_carbon_cost_m * unpassed_fraction

        return {
            "carbon_price": carbon_price,
            "effective_emissions_t": effective_emissions_t,
            "gross_carbon_cost_m": gross_carbon_cost_m,
            "net_carbon_cost_m": net_carbon_cost_m,
        }

    def evaluate_physical_risk(
        self,
        company: CompanyClimateProfile,
        scenario: NGFSScenario,
        year: int = 2030,
        base_annual_damage_rate: float = 0.015,
    ) -> Dict[str, float]:
        """Calculates chronic and acute physical asset damages under geospatial hazard warming multipliers."""
        hazard_mult = scenario.physical_hazard_multiplier.get(year, 1.20)

        annual_damage_m = (
            company.asset_replacement_cost *
            company.physical_hazard_exposure *
            base_annual_damage_rate *
            hazard_mult
        )

        return {
            "hazard_multiplier": hazard_mult,
            "annual_damage_m": annual_damage_m,
        }

    def compute_merton_credit_migration(
        self,
        company: CompanyClimateProfile,
        asset_impairment_pct: float,
        maturity_years: float = 1.0,
    ) -> Dict[str, float]:
        """Computes structural Merton default probability (PD) shift and credit spread widening."""
        V0 = company.enterprise_value
        D = max(1.0, company.total_debt)
        sigma = company.asset_volatility
        r = company.risk_free_rate
        T = maturity_years

        d1_base = (np.log(V0 / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2_base = d1_base - sigma * np.sqrt(T)
        pd_base = float(norm.cdf(-d2_base))

        V_stressed = max(D * 0.5, V0 * (1.0 - asset_impairment_pct))
        d1_stress = (np.log(V_stressed / D) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2_stress = d1_stress - sigma * np.sqrt(T)
        pd_stress = float(norm.cdf(-d2_stress))

        lgd = company.lgd
        spread_base = - (1.0 / T) * np.log(max(1e-6, 1.0 - lgd * pd_base)) * 10000.0
        spread_stress = - (1.0 / T) * np.log(max(1e-6, 1.0 - lgd * pd_stress)) * 10000.0
        spread_delta_bps = max(0.0, spread_stress - spread_base)

        return {
            "baseline_pd": pd_base,
            "stressed_pd": pd_stress,
            "baseline_spread_bps": spread_base,
            "stressed_spread_bps": spread_stress,
            "spread_delta_bps": spread_delta_bps,
        }

    def stress_test_company(
        self,
        company: CompanyClimateProfile,
        scenario_type: NGFSScenarioType,
        year: int = 2030,
        dcf_horizon_years: int = 10,
    ) -> ClimateStressResult:
        """Executes full integrated transition + physical + Climate VaR + credit stress on a firm."""
        scenario = self.scenarios[scenario_type]

        trans_res = self.evaluate_transition_risk(company, scenario, year)
        phys_res = self.evaluate_physical_risk(company, scenario, year)

        annual_trans_cost = trans_res["net_carbon_cost_m"]
        annual_phys_damage = phys_res["annual_damage_m"]
        total_annual_drag = annual_trans_cost + annual_phys_damage

        base_ebitda = company.ebitda
        stressed_ebitda = max(0.0, base_ebitda - total_annual_drag)
        ebitda_impairment_pct = (stressed_ebitda - base_ebitda) / max(1.0, base_ebitda)

        r_wacc = company.cost_of_capital
        discount_factor_sum = sum(1.0 / ((1.0 + r_wacc)**t) for t in range(1, dcf_horizon_years + 1))
        pv_cash_flow_loss = total_annual_drag * discount_factor_sum

        terminal_impairment = total_annual_drag / max(0.02, r_wacc) * 0.40
        total_equity_loss = min(company.market_cap * 0.95, pv_cash_flow_loss + terminal_impairment)

        stressed_equity_val = max(company.market_cap * 0.05, company.market_cap - total_equity_loss)
        climate_equity_var_pct = (stressed_equity_val - company.market_cap) / company.market_cap

        asset_impairment_pct = total_equity_loss / max(1.0, company.enterprise_value)

        credit_res = self.compute_merton_credit_migration(company, asset_impairment_pct)

        return ClimateStressResult(
            ticker=company.ticker,
            scenario_name=scenario.scenario_type.value,
            horizon_year=year,
            carbon_price_usd=trans_res["carbon_price"],
            annual_transition_cost_usd=annual_trans_cost,
            annual_physical_damage_usd=annual_phys_damage,
            baseline_ebitda=base_ebitda,
            stressed_ebitda=stressed_ebitda,
            ebitda_impairment_pct=ebitda_impairment_pct,
            baseline_equity_val=company.market_cap,
            stressed_equity_val=stressed_equity_val,
            climate_equity_var_pct=climate_equity_var_pct,
            baseline_pd=credit_res["baseline_pd"],
            stressed_pd=credit_res["stressed_pd"],
            credit_spread_widening_bps=credit_res["spread_delta_bps"],
        )

    def run_portfolio_stress_test(
        self,
        portfolio: List[CompanyClimateProfile],
        scenario_type: NGFSScenarioType,
        year: int = 2030,
    ) -> PortfolioClimateStressReport:
        """Evaluates an entire multi-asset portfolio under an NGFS scenario."""
        scenario = self.scenarios[scenario_type]
        company_results = [self.stress_test_company(c, scenario_type, year) for c in portfolio]

        total_val = sum(c.market_cap for c in portfolio)
        stressed_val = sum(r.stressed_equity_val for r in company_results)
        portfolio_var_pct = (stressed_val - total_val) / total_val

        total_trans = sum(r.annual_transition_cost_usd for r in company_results)
        total_phys = sum(r.annual_physical_damage_usd for r in company_results)

        weights = np.array([c.market_cap for c in portfolio]) / total_val
        w_base_pd = float(np.sum(weights * np.array([r.baseline_pd for r in company_results])))
        w_stress_pd = float(np.sum(weights * np.array([r.stressed_pd for r in company_results])))
        w_spread_delta = float(np.sum(weights * np.array([r.credit_spread_widening_bps for r in company_results])))

        return PortfolioClimateStressReport(
            scenario_name=scenario.scenario_type.value,
            horizon_year=year,
            n_companies=len(portfolio),
            total_portfolio_value=total_val,
            stressed_portfolio_value=stressed_val,
            portfolio_climate_var_pct=portfolio_var_pct,
            total_annual_transition_cost=total_trans,
            total_annual_physical_damage=total_phys,
            weighted_baseline_pd=w_base_pd,
            weighted_stressed_pd=w_stress_pd,
            weighted_spread_widening_bps=w_spread_delta,
            company_results=company_results,
        )

    def multi_scenario_comparison(
        self,
        portfolio: List[CompanyClimateProfile],
        year: int = 2030,
    ) -> pd.DataFrame:
        """Compares portfolio-level Climate VaR and credit shifts across all 3 NGFS scenarios."""
        rows = []
        for s_type in [NGFSScenarioType.NET_ZERO_2050, NGFSScenarioType.DELAYED_TRANSITION, NGFSScenarioType.CURRENT_POLICIES]:
            rep = self.run_portfolio_stress_test(portfolio, s_type, year)
            rows.append({
                "NGFS_Scenario": rep.scenario_name,
                "Horizon_Year": year,
                "Portfolio_Climate_VaR_%": f"{rep.portfolio_climate_var_pct:+.2%}",
                "Total_Annual_Transition_Cost": f"${rep.total_annual_transition_cost:,.1f}M",
                "Total_Annual_Physical_Damage": f"${rep.total_annual_physical_damage:,.1f}M",
                "Weighted_Baseline_PD": f"{rep.weighted_baseline_pd:.2%}",
                "Weighted_Stressed_PD": f"{rep.weighted_stressed_pd:.2%}",
                "Spread_Widening_bps": f"{rep.weighted_spread_widening_bps:+.0f} bps",
            })
        return pd.DataFrame(rows)
