"""
Risk: Portfolio Macro Stress Testing & Historical Crisis Replay Engine.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StressScenario:
    name: str
    description: str
    equity_shock: float
    yield_shock_bps: float
    credit_spread_bps: float
    volatility_surge_pct: float
    commodity_shock: float = 0.0


class PortfolioStressTestingEngine:
    """Evaluates multi-asset portfolio drawdowns under macroeconomic and historical crises."""

    def __init__(self, portfolio_value: float = 10_000_000.0):
        self.portfolio_value = portfolio_value

        # Baseline asset allocation weights
        self.weights = {
            "US_Equities": 0.40,
            "Global_Equities": 0.15,
            "Treasury_Bonds_10Y": 0.25,
            "Corporate_Credit": 0.10,
            "Commodities_Gold": 0.10,
        }
        # Sensitivities
        self.betas = {"US_Equities": 1.05, "Global_Equities": 0.95, "Corporate_Credit": 0.35}
        self.durations = {"Treasury_Bonds_10Y": 7.5, "Corporate_Credit": 5.2}

    def evaluate_scenario(self, scenario: StressScenario) -> Dict[str, Any]:
        """Compute portfolio loss and percentage impact under a specific scenario."""
        pnl_pct = 0.0

        # Equity impact
        pnl_pct += self.weights["US_Equities"] * self.betas["US_Equities"] * scenario.equity_shock
        pnl_pct += self.weights["Global_Equities"] * self.betas["Global_Equities"] * scenario.equity_shock

        # Fixed income duration impact: -Duration * Delta_y
        dy = scenario.yield_shock_bps / 10000.0
        pnl_pct += self.weights["Treasury_Bonds_10Y"] * (-self.durations["Treasury_Bonds_10Y"] * dy)

        # Corporate credit spread impact
        ds = scenario.credit_spread_bps / 10000.0
        pnl_pct += self.weights["Corporate_Credit"] * (
            -self.durations["Corporate_Credit"] * dy - self.durations["Corporate_Credit"] * ds + 0.35 * scenario.equity_shock
        )

        # Commodities
        pnl_pct += self.weights["Commodities_Gold"] * scenario.commodity_shock

        pnl_dollar = self.portfolio_value * pnl_pct

        return {
            "scenario": scenario.name,
            "pnl_pct": float(pnl_pct),
            "pnl_dollar": float(pnl_dollar),
            "equity_shock": scenario.equity_shock,
            "yield_shock_bps": scenario.yield_shock_bps,
        }

    def run_standard_crisis_suite(self) -> pd.DataFrame:
        """Run standard crisis benchmarks matching the master portfolio."""
        scenarios = [
            StressScenario("2008 Global Financial Crisis", "Lehman collapse & credit freeze", -0.38, -150.0, 400.0, 1.50, -0.15),
            StressScenario("2020 COVID Crash", "March 2020 liquidity shock", -0.30, -100.0, 250.0, 1.20, -0.25),
            StressScenario("2022 Interest Rate Shock", "Rapid Fed tightening cycle", -0.20, 200.0, 120.0, 0.40, 0.20),
            StressScenario("Global Equity Meltdown (-30%)", "Unconditional equity market crash", -0.30, -80.0, 300.0, 1.00, -0.10),
            StressScenario("Geopolitical Stagflation Shock", "Oil surge and inflation shock", -0.15, 150.0, 150.0, 0.60, 0.40),
        ]
        records = [self.evaluate_scenario(s) for s in scenarios]
        df = pd.DataFrame(records)
        return df

    def correlation_breakdown_stress(self, crisis_alpha: float = 0.70) -> Dict[str, float]:
        """Model systemic volatility surge when cross-asset correlations spike towards 1.0."""
        base_vol = 0.0963
        distressed_vol = base_vol * (1.0 + crisis_alpha)
        return {
            "base_annual_vol": base_vol,
            "stressed_annual_vol": distressed_vol,
            "vol_surge_pct": ((distressed_vol - base_vol) / base_vol) * 100.0,
        }
