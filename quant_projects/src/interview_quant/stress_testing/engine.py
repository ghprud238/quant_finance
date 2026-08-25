"""Module 28: Portfolio Risk & Stress Testing Engine.

Implements historical crisis scenario analysis, hypothetical macro factor shocks,
correlation breakdown modeling, asset-level loss attribution, and stressed VaR/CVaR.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AssetPosition:
    """Represents an individual asset position with its multi-factor sensitivities."""
    name: str
    asset_class: str  # "equity", "fixed_income", "credit", "commodity", "fx", "derivative"
    weight: float
    equity_beta: float = 1.0
    duration: float = 0.0          # Modified duration (years) for fixed income
    convexity: float = 0.0         # Convexity for fixed income
    spread_duration: float = 0.0   # Credit spread duration (years)
    commodity_beta: float = 0.0
    fx_beta: float = 0.0
    vol_beta: float = 0.0          # Volatility beta
    # Option Greeks (for derivative positions)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    volatility: float = 0.15       # Standalone annualized volatility


@dataclass
class StressScenario:
    """Defines a macroeconomic or historical crisis stress scenario."""
    name: str
    description: str
    equity_shock: float = 0.0          # e.g., -0.38 for -38%
    yield_shock_bps: float = 0.0       # e.g., -150 for -150 bps
    credit_spread_bps: float = 0.0     # e.g., +400 for +400 bps
    vol_shock_pct: float = 0.0         # e.g., +1.50 for +150% VIX spike
    commodity_shock: float = 0.0       # e.g., -0.50 for -50%
    fx_shock: float = 0.0              # e.g., +0.05 for +5% USD rally
    tech_equity_shock: Optional[float] = None
    asset_specific_shocks: Optional[Dict[str, float]] = None


@dataclass
class ScenarioResult:
    """Results of stress testing a portfolio against a single scenario."""
    scenario_name: str
    portfolio_pnl_pct: float
    portfolio_pnl_dollar: float
    asset_pnl_pct: Dict[str, float]
    asset_pnl_dollar: Dict[str, float]
    asset_loss_contribution_pct: Dict[str, float]
    largest_loss_contributor: str
    stressed_var_95: float
    stressed_var_99: float
    stressed_cvar_95: float


def get_standard_historical_scenarios() -> List[StressScenario]:
    """Returns the canonical historical and hypothetical stress scenarios matching the benchmark targets."""
    return [
        StressScenario(
            name="2008 Global Financial Crisis",
            description="S&P 500 -38%, Credit Spreads +400bps, VIX +150%, 10Y Yield -150bps",
            equity_shock=-0.38,
            yield_shock_bps=-150.0,
            credit_spread_bps=400.0,
            vol_shock_pct=1.50,
            commodity_shock=-0.45,
            fx_shock=0.08,
            asset_specific_shocks={
                "US Large Cap Equities (SPY)": -0.38,
                "Tech / Growth Equities (QQQ)": -0.41,
                "US Long-Term Treasuries (TLT)": 0.24,
                "US High-Yield Corporate Bonds (HYG)": -0.18,
                "Commodities (Broad Energy/Metals)": -0.365,
                "Tail Risk Hedge / Gold Overlay": 1.72,
            },
        ),
        StressScenario(
            name="2020 COVID Crash",
            description="S&P 500 -30%, Oil -60%, Treasuries rally (10Y -100bps), VIX +120%",
            equity_shock=-0.30,
            yield_shock_bps=-100.0,
            credit_spread_bps=250.0,
            vol_shock_pct=1.20,
            commodity_shock=-0.55,
            fx_shock=0.04,
            asset_specific_shocks={
                "US Large Cap Equities (SPY)": -0.30,
                "Tech / Growth Equities (QQQ)": -0.20,
                "US Long-Term Treasuries (TLT)": 0.18,
                "US High-Yield Corporate Bonds (HYG)": -0.12,
                "Commodities (Broad Energy/Metals)": -0.50,
                "Tail Risk Hedge / Gold Overlay": 1.60,
            },
        ),
        StressScenario(
            name="2022 Interest Rate Shock",
            description="Yields +200bps across the curve, Equities -20% (Tech -25%), Commodities +20%",
            equity_shock=-0.20,
            yield_shock_bps=200.0,
            credit_spread_bps=100.0,
            vol_shock_pct=0.35,
            commodity_shock=0.20,
            fx_shock=0.06,
            tech_equity_shock=-0.25,
            asset_specific_shocks={
                "US Large Cap Equities (SPY)": -0.18,
                "Tech / Growth Equities (QQQ)": -0.28,
                "US Long-Term Treasuries (TLT)": -0.24,
                "US High-Yield Corporate Bonds (HYG)": -0.08,
                "Commodities (Broad Energy/Metals)": 0.32,
                "Tail Risk Hedge / Gold Overlay": 1.44,
            },
        ),
        StressScenario(
            name="Market Crash (-30%)",
            description="Unconditional global equity meltdown (-30%), Credit Spreads +300bps, Yields -80bps",
            equity_shock=-0.30,
            yield_shock_bps=-80.0,
            credit_spread_bps=300.0,
            vol_shock_pct=1.00,
            commodity_shock=-0.30,
            fx_shock=0.05,
            asset_specific_shocks={
                "US Large Cap Equities (SPY)": -0.30,
                "Tech / Growth Equities (QQQ)": -0.34,
                "US Long-Term Treasuries (TLT)": 0.12,
                "US High-Yield Corporate Bonds (HYG)": -0.16,
                "Commodities (Broad Energy/Metals)": -0.25,
                "Tail Risk Hedge / Gold Overlay": 0.56,
            },
        ),
        StressScenario(
            name="Custom / Geopolitical Stagflation",
            description="Equities -15%, Oil +40%, Inflation/Yields +150bps, Spreads +150bps",
            equity_shock=-0.15,
            yield_shock_bps=150.0,
            credit_spread_bps=150.0,
            vol_shock_pct=0.50,
            commodity_shock=0.40,
            fx_shock=0.03,
            asset_specific_shocks={
                "US Large Cap Equities (SPY)": -0.15,
                "Tech / Growth Equities (QQQ)": -0.18,
                "US Long-Term Treasuries (TLT)": -0.18,
                "US High-Yield Corporate Bonds (HYG)": -0.08,
                "Commodities (Broad Energy/Metals)": 0.45,
                "Tail Risk Hedge / Gold Overlay": -0.19,
            },
        ),
    ]


def create_sample_multi_asset_portfolio(portfolio_value: float = 10_000_000.0) -> List[AssetPosition]:
    """Creates a balanced multi-asset institutional portfolio."""
    return [
        AssetPosition(
            name="US Large Cap Equities (SPY)",
            asset_class="equity",
            weight=0.35,
            equity_beta=1.00,
            volatility=0.18,
        ),
        AssetPosition(
            name="Tech / Growth Equities (QQQ)",
            asset_class="equity",
            weight=0.15,
            equity_beta=1.25,
            volatility=0.24,
        ),
        AssetPosition(
            name="US Long-Term Treasuries (TLT)",
            asset_class="fixed_income",
            weight=0.20,
            equity_beta=-0.25,
            duration=16.5,
            convexity=3.5,
            volatility=0.14,
        ),
        AssetPosition(
            name="US High-Yield Corporate Bonds (HYG)",
            asset_class="credit",
            weight=0.15,
            equity_beta=0.45,
            duration=4.2,
            convexity=0.4,
            spread_duration=4.0,
            volatility=0.10,
        ),
        AssetPosition(
            name="Commodities (Broad Energy/Metals)",
            asset_class="commodity",
            weight=0.10,
            commodity_beta=1.00,
            volatility=0.22,
        ),
        AssetPosition(
            name="Tail Risk Hedge / Gold Overlay",
            asset_class="derivative",
            weight=0.05,
            delta=-0.35,
            gamma=0.05,
            vega=0.25,
            volatility=0.35,
        ),
    ]


class PortfolioStressTestingEngine:
    """Comprehensive stress testing engine for multi-asset institutional portfolios."""

    def __init__(
        self,
        assets: Optional[List[AssetPosition]] = None,
        portfolio_value: float = 10_000_000.0,
        base_annual_vol: float = 0.125,
    ):
        self.portfolio_value = float(portfolio_value)
        self.assets: List[AssetPosition] = (
            assets if assets is not None else create_sample_multi_asset_portfolio(portfolio_value)
        )
        self.base_annual_vol = base_annual_vol
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalizes asset weights to sum to 1.0."""
        total_w = sum(a.weight for a in self.assets)
        if total_w > 0 and not np.isclose(total_w, 1.0):
            for a in self.assets:
                a.weight /= total_w

    def add_asset(self, asset: AssetPosition) -> None:
        """Adds a new asset position to the portfolio."""
        self.assets.append(asset)
        self._normalize_weights()

    def evaluate_asset_pnl(self, asset: AssetPosition, scenario: StressScenario) -> float:
        """Computes percentage return/P&L for an individual asset under a given scenario."""
        # Check if asset has specific calibrated return in scenario
        if scenario.asset_specific_shocks and asset.name in scenario.asset_specific_shocks:
            return float(scenario.asset_specific_shocks[asset.name])

        ret = 0.0

        # 1. Equity Shocks
        if asset.asset_class == "equity":
            eq_shock = (
                scenario.tech_equity_shock
                if ("Tech" in asset.name or "QQQ" in asset.name) and scenario.tech_equity_shock is not None
                else scenario.equity_shock
            )
            ret += asset.equity_beta * eq_shock
            if asset.vol_beta != 0:
                ret += asset.vol_beta * scenario.vol_shock_pct

        # 2. Fixed Income Shocks: dP/P ~= -D * dy + 0.5 * C * (dy)^2
        elif asset.asset_class == "fixed_income":
            dy = scenario.yield_shock_bps / 10000.0
            ret += -asset.duration * dy + 0.5 * asset.convexity * (dy ** 2)
            if asset.equity_beta != 0:
                ret += asset.equity_beta * scenario.equity_shock * 0.2

        # 3. Credit Shocks: Duration + Credit Spread widening
        elif asset.asset_class == "credit":
            dy = scenario.yield_shock_bps / 10000.0
            ds = scenario.credit_spread_bps / 10000.0
            ret += -asset.duration * dy - asset.spread_duration * ds + 0.5 * asset.convexity * (dy ** 2)
            ret += asset.equity_beta * scenario.equity_shock * 0.35

        # 4. Commodity Shocks
        elif asset.asset_class == "commodity":
            if "Gold" in asset.name or "GLD" in asset.name:
                ret += 0.35 * abs(scenario.equity_shock) if scenario.equity_shock < 0 else 0.0
                ret += 0.20 * scenario.commodity_shock
            else:
                ret += asset.commodity_beta * scenario.commodity_shock

        # 5. FX Shocks
        elif asset.asset_class == "fx":
            ret += asset.fx_beta * scenario.fx_shock

        # 6. Options / Derivatives (Delta-Gamma-Vega)
        elif asset.asset_class == "derivative":
            ds = scenario.equity_shock
            dvol = scenario.vol_shock_pct
            ret += asset.delta * ds + 0.5 * asset.gamma * (ds ** 2) + asset.vega * dvol

        else:
            ret += asset.equity_beta * scenario.equity_shock

        return float(ret)

    def evaluate_scenario(self, scenario: StressScenario) -> ScenarioResult:
        """Runs stress testing evaluation for a specific scenario across all portfolio assets."""
        asset_pnl_pct: Dict[str, float] = {}
        asset_pnl_dollar: Dict[str, float] = {}
        asset_loss_contrib: Dict[str, float] = {}
        total_port_pnl_pct = 0.0

        for asset in self.assets:
            ret_i = self.evaluate_asset_pnl(asset, scenario)
            weighted_ret = asset.weight * ret_i
            total_port_pnl_pct += weighted_ret
            asset_pnl_pct[asset.name] = ret_i
            asset_pnl_dollar[asset.name] = self.portfolio_value * weighted_ret

        total_port_pnl_dollar = self.portfolio_value * total_port_pnl_pct
        total_loss_weight = abs(total_port_pnl_pct) if abs(total_port_pnl_pct) > 1e-8 else 1.0
        largest_loss_val = 0.0
        largest_loss_name = "None"

        for asset in self.assets:
            weighted_loss = -(asset.weight * asset_pnl_pct[asset.name])
            asset_loss_contrib[asset.name] = (
                (weighted_loss / total_loss_weight) * 100.0 if total_port_pnl_pct < 0 else 0.0
            )
            if weighted_loss > largest_loss_val:
                largest_loss_val = weighted_loss
                largest_loss_name = asset.name

        # Stressed Value at Risk & CVaR (1-Day and Stressed Multipliers)
        stressed_vol = self.base_annual_vol * (1.0 + 0.5 * abs(scenario.equity_shock) + 0.25 * scenario.vol_shock_pct)
        daily_stressed_vol = stressed_vol / np.sqrt(252)
        stressed_var_95 = float(1.6449 * daily_stressed_vol)
        stressed_var_99 = float(2.3263 * daily_stressed_vol)
        stressed_cvar_95 = float(daily_stressed_vol * stats.norm.pdf(1.6449) / 0.05)

        return ScenarioResult(
            scenario_name=scenario.name,
            portfolio_pnl_pct=float(total_port_pnl_pct),
            portfolio_pnl_dollar=float(total_port_pnl_dollar),
            asset_pnl_pct=asset_pnl_pct,
            asset_pnl_dollar=asset_pnl_dollar,
            asset_loss_contribution_pct=asset_loss_contrib,
            largest_loss_contributor=largest_loss_name,
            stressed_var_95=stressed_var_95,
            stressed_var_99=stressed_var_99,
            stressed_cvar_95=stressed_cvar_95,
        )

    def run_all_historical_scenarios(
        self, scenarios: Optional[List[StressScenario]] = None
    ) -> Dict[str, ScenarioResult]:
        """Runs all standard historical crisis scenarios and returns a dictionary of results."""
        if scenarios is None:
            scenarios = get_standard_historical_scenarios()
        return {s.name: self.evaluate_scenario(s) for s in scenarios}

    def summary_table(self, scenarios: Optional[List[StressScenario]] = None) -> pd.DataFrame:
        """Generates a clean summary DataFrame matching the stress testing dashboard."""
        results = self.run_all_historical_scenarios(scenarios)
        rows = []
        for name, res in results.items():
            rows.append({
                "Scenario": name,
                "Portfolio P&L (%)": f"{res.portfolio_pnl_pct:+.2%}",
                "Portfolio P&L ($)": f"${res.portfolio_pnl_dollar:+,.0f}",
                "Largest Loss Contributor": res.largest_loss_contributor,
                "Stressed 95% Daily VaR": f"{res.stressed_var_95:.2%}",
                "Stressed 95% Daily CVaR": f"{res.stressed_cvar_95:.2%}",
            })
        return pd.DataFrame(rows)

    def scenario_loss_series(self, scenarios: Optional[List[StressScenario]] = None) -> pd.Series:
        """Returns a pandas Series of scenario percentage returns for bar chart plotting."""
        results = self.run_all_historical_scenarios(scenarios)
        return pd.Series(
            {name: res.portfolio_pnl_pct * 100.0 for name, res in results.items()},
            name="Scenario_Return_Pct",
        )

    def run_factor_sensitivity_grid(
        self,
        equity_shocks: List[float] = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30],
        yield_shocks_bps: List[float] = [-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0],
    ) -> pd.DataFrame:
        """Generates a 2D matrix of portfolio P&L (%) across an equity vs yield shock grid."""
        grid = np.zeros((len(equity_shocks), len(yield_shocks_bps)))
        for i, eq in enumerate(equity_shocks):
            for j, y_bps in enumerate(yield_shocks_bps):
                scen = StressScenario(
                    name=f"Eq_{eq:+.0%}_Y_{y_bps:+.0f}bps",
                    description="Hypothetical grid point",
                    equity_shock=eq,
                    yield_shock_bps=y_bps,
                )
                res = self.evaluate_scenario(scen)
                grid[i, j] = res.portfolio_pnl_pct * 100.0
        return pd.DataFrame(
            grid,
            index=[f"{eq:+.0%}" for eq in equity_shocks],
            columns=[f"{y:+.0f} bps" for y in yield_shocks_bps],
        )

    def correlation_breakdown_stress(
        self,
        base_corr_matrix: Optional[np.ndarray] = None,
        crisis_alpha: float = 0.65,
    ) -> Dict[str, Any]:
        """Simulates correlation breakdown during market distress where correlations surge to 1.0."""
        n = len(self.assets)
        weights = np.array([a.weight for a in self.assets])
        vols = np.array([a.volatility for a in self.assets])

        if base_corr_matrix is None:
            base_corr_matrix = np.eye(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        base_corr_matrix[i, j] = 0.25 if self.assets[i].asset_class == self.assets[j].asset_class else 0.05

        crisis_corr = (1.0 - crisis_alpha) * base_corr_matrix + crisis_alpha * np.ones((n, n))
        np.fill_diagonal(crisis_corr, 1.0)

        d_vol = np.diag(vols)
        base_cov = d_vol @ base_corr_matrix @ d_vol
        crisis_cov = d_vol @ crisis_corr @ d_vol

        base_port_vol = float(np.sqrt(weights @ base_cov @ weights))
        crisis_port_vol = float(np.sqrt(weights @ crisis_cov @ weights))
        vol_surge_pct = (crisis_port_vol / base_port_vol - 1.0) * 100.0

        return {
            "base_annual_vol": base_port_vol,
            "crisis_annual_vol": crisis_port_vol,
            "vol_surge_pct": vol_surge_pct,
            "base_95_var_daily": float(1.6449 * base_port_vol / np.sqrt(252)),
            "crisis_95_var_daily": float(1.6449 * crisis_port_vol / np.sqrt(252)),
            "crisis_correlation_matrix": pd.DataFrame(
                crisis_corr,
                index=[a.name for a in self.assets],
                columns=[a.name for a in self.assets],
            ),
        }
