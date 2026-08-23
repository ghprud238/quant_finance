"""Portfolio Risk Report and Performance Analytics.

Implements full risk summary table matching institutional risk reporting:
- Annualized Return (CAGR)
- Annualized Volatility
- Sharpe Ratio (configurable Rf)
- Sortino Ratio (downside risk below target MAR)
- Historical & Parametric VaR (95%, 99%)
- Expected Shortfall / CVaR (95%, 99%)
- Max Drawdown & Recovery Duration
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from ..cvar.expected_shortfall import ExpectedShortfallModel


class PortfolioRiskReport:
    """Calculates and formats comprehensive portfolio risk and tail metrics."""

    def __init__(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        portfolio_name: str = "Risk Portfolio",
        risk_free_rate: float = 0.02,
        target_return: float = 0.0,
        periods_per_year: int = 252,
    ):
        if isinstance(returns, pd.Series):
            self.returns = returns.dropna()
        else:
            arr = np.asarray(returns, dtype=float)
            arr = arr[~np.isnan(arr)]
            self.returns = pd.Series(arr)

        if len(self.returns) < 5:
            raise ValueError(f"At least 5 return observations required, got {len(self.returns)}")

        self.portfolio_name = portfolio_name
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.periods_per_year = periods_per_year
        self.es_model = ExpectedShortfallModel()

    def annualized_return(self, geometric: bool = True) -> float:
        """Calculates compound annualized return (CAGR) or arithmetic annual return."""
        n = len(self.returns)
        if n == 0:
            return 0.0
        if geometric:
            comp_factor = np.prod(1.0 + self.returns)
            if comp_factor <= 0:
                return -1.0
            return float(comp_factor ** (self.periods_per_year / n) - 1.0)
        else:
            return float(np.mean(self.returns) * self.periods_per_year)

    def annualized_volatility(self) -> float:
        """Calculates annualized sample volatility."""
        return float(np.std(self.returns, ddof=1) * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self) -> float:
        """Calculates annualized Sharpe Ratio."""
        vol = self.annualized_volatility()
        if vol <= 1e-12:
            return 0.0
        ann_ret = self.annualized_return(geometric=False)
        return float((ann_ret - self.risk_free_rate) / vol)

    def sortino_ratio(self) -> float:
        """Calculates annualized Sortino Ratio against MAR target."""
        daily_target = (1.0 + self.target_return) ** (1.0 / self.periods_per_year) - 1.0
        underperformance = np.minimum(0.0, self.returns - daily_target)
        downside_variance = np.mean(underperformance ** 2)
        downside_vol = float(np.sqrt(downside_variance) * np.sqrt(self.periods_per_year))
        if downside_vol <= 1e-12:
            return 0.0
        ann_ret = self.annualized_return(geometric=False)
        return float((ann_ret - self.risk_free_rate) / downside_vol)

    def drawdown_series(self) -> pd.DataFrame:
        """Computes full historical drawdown curve, high-water-marks, and durations."""
        cum_wealth = (1.0 + self.returns).cumprod()
        hwm = cum_wealth.cummax()
        drawdown = (cum_wealth - hwm) / hwm

        durations = []
        curr_dur = 0
        for dd in drawdown:
            if dd < 0:
                curr_dur += 1
            else:
                curr_dur = 0
            durations.append(curr_dur)

        return pd.DataFrame({
            "cumulative_wealth": cum_wealth,
            "high_water_mark": hwm,
            "drawdown": drawdown,
            "drawdown_duration": durations,
        }, index=self.returns.index)

    def max_drawdown(self) -> float:
        """Calculates peak-to-trough maximum drawdown."""
        dd = self.drawdown_series()["drawdown"]
        return float(np.min(dd))

    def max_drawdown_duration(self) -> int:
        """Calculates maximum drawdown duration in periods."""
        dd = self.drawdown_series()["drawdown_duration"]
        return int(np.max(dd))

    def var_95(self, method: str = "historical") -> float:
        """Value at Risk at 95% confidence level."""
        if method == "historical":
            return self.es_model.historical_var(self.returns, confidence_level=0.95, as_loss=True)
        elif method == "parametric":
            return self.es_model.parametric_gaussian_var(self.returns, confidence_level=0.95, as_loss=True)
        elif method == "cornish_fisher":
            return self.es_model.cornish_fisher_var(self.returns, confidence_level=0.95, as_loss=True)
        return self.es_model.historical_var(self.returns, confidence_level=0.95, as_loss=True)

    def var_99(self, method: str = "historical") -> float:
        """Value at Risk at 99% confidence level."""
        if method == "historical":
            return self.es_model.historical_var(self.returns, confidence_level=0.99, as_loss=True)
        elif method == "parametric":
            return self.es_model.parametric_gaussian_var(self.returns, confidence_level=0.99, as_loss=True)
        elif method == "cornish_fisher":
            return self.es_model.cornish_fisher_var(self.returns, confidence_level=0.99, as_loss=True)
        return self.es_model.historical_var(self.returns, confidence_level=0.99, as_loss=True)

    def cvar_95(self, method: str = "historical") -> float:
        """Expected Shortfall (CVaR) at 95% confidence level."""
        if method == "historical":
            return self.es_model.historical_es(self.returns, confidence_level=0.95, as_loss=True)
        elif method == "parametric":
            return self.es_model.parametric_gaussian_es(self.returns, confidence_level=0.95, as_loss=True)
        return self.es_model.historical_es(self.returns, confidence_level=0.95, as_loss=True)

    def cvar_99(self, method: str = "historical") -> float:
        """Expected Shortfall (CVaR) at 99% confidence level."""
        if method == "historical":
            return self.es_model.historical_es(self.returns, confidence_level=0.99, as_loss=True)
        elif method == "parametric":
            return self.es_model.parametric_gaussian_es(self.returns, confidence_level=0.99, as_loss=True)
        return self.es_model.historical_es(self.returns, confidence_level=0.99, as_loss=True)

    def to_dict(self) -> Dict[str, Any]:
        """Compiles all metrics into a clean dictionary."""
        return {
            "portfolio_name": self.portfolio_name,
            "observations": len(self.returns),
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "var_95_historical": self.var_95("historical"),
            "var_99_historical": self.var_99("historical"),
            "cvar_95_historical": self.cvar_95("historical"),
            "cvar_99_historical": self.cvar_99("historical"),
            "var_95_parametric": self.var_95("parametric"),
            "cvar_95_parametric": self.cvar_95("parametric"),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
        }

    def summary_dataframe(self) -> pd.DataFrame:
        """Generates structured pandas DataFrame matching executive risk dashboards."""
        d = self.to_dict()
        rows = [
            {"Metric": "Annualized Return (CAGR)", "Value": f"{d['annualized_return']:+.2%}", "Category": "Performance"},
            {"Metric": "Annualized Volatility", "Value": f"{d['annualized_volatility']:.2%}", "Category": "Volatility"},
            {"Metric": f"Sharpe Ratio (Rf={self.risk_free_rate:.1%})", "Value": f"{d['sharpe_ratio']:.2f}", "Category": "Risk-Adjusted"},
            {"Metric": "Sortino Ratio", "Value": f"{d['sortino_ratio']:.2f}", "Category": "Risk-Adjusted"},
            {"Metric": "Value-at-Risk (95% Daily)", "Value": f"{d['var_95_historical']:.2%}", "Category": "Tail Risk"},
            {"Metric": "Value-at-Risk (99% Daily)", "Value": f"{d['var_99_historical']:.2%}", "Category": "Tail Risk"},
            {"Metric": "Expected Shortfall (95% Daily CVaR)", "Value": f"{d['cvar_95_historical']:.2%}", "Category": "Tail Risk"},
            {"Metric": "Expected Shortfall (99% Daily CVaR)", "Value": f"{d['cvar_99_historical']:.2%}", "Category": "Tail Risk"},
            {"Metric": "Maximum Drawdown", "Value": f"{d['max_drawdown']:.2%}", "Category": "Drawdown"},
            {"Metric": "Max Drawdown Duration", "Value": f"{d['max_drawdown_duration']} periods", "Category": "Drawdown"},
        ]
        return pd.DataFrame(rows)

    def print_terminal_card(self):
        """Renders formatted terminal dashboard card."""
        df = self.summary_dataframe()
        border = "-" * 76
        print(border)
        print(f"{self.portfolio_name.upper():^76}")
        print(border)
        for _, row in df.iterrows():
            metric = row['Metric']
            val = row['Value']
            cat = row['Category']
            print(f"{metric:<36} {val:>15}   [{cat:<12}]")
        print(border)
