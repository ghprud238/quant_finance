"""Portfolio Risk Dashboard.

This module provides the PortfolioRiskDashboard class for end-to-end
portfolio risk and performance analysis, aggregation, tabular summary,
and visual terminal dashboard rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from quant_foundations.portfolio.risk_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_series,
    excess_kurtosis,
    gain_loss_ratio,
    information_ratio,
    jensens_alpha,
    max_drawdown,
    max_drawdown_duration,
    omega_ratio,
    realized_beta,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    tracking_error,
    value_at_risk,
    win_rate,
)


class PortfolioRiskDashboard:
    """Comprehensive Portfolio Risk and Performance Dashboard.

    Parameters
    ----------
    returns : Series, DataFrame, array, dict, or sequence
        Asset returns (multi-column DataFrame) or pre-aggregated portfolio returns (Series/1D array).
    weights : dict, list, array, or Series, optional
        Asset weights corresponding to columns in `returns`. If None and multiple assets are provided,
        defaults to equal-weighted allocation.
    benchmark_returns : Series, DataFrame, array, or sequence, optional
        Benchmark periodic returns (e.g. S&P 500, MSCI World) for relative risk metrics.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate (e.g. 0.02 for 2.0%).
    periods_per_year : int, default 252
        Trading periods per year (252 for daily, 52 for weekly, 12 for monthly).
    confidence_level : float, default 0.95
        Confidence level for VaR and CVaR calculations.
    target_return : float, default 0.0
        Annualized Minimum Acceptable Return (MAR) for Sortino Ratio.
    name : str, default 'Portfolio'
        Display name for the portfolio.
    benchmark_name : str, default 'Benchmark'
        Display name for the benchmark.
    """

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float], Dict[str, Sequence[float]]],
        weights: Optional[Union[Sequence[float], Dict[str, float], pd.Series, np.ndarray]] = None,
        benchmark_returns: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[float]]] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        confidence_level: float = 0.95,
        target_return: float = 0.0,
        name: str = "Portfolio",
        benchmark_name: str = "Benchmark",
    ) -> None:
        self.risk_free_rate = float(risk_free_rate)
        self.periods_per_year = int(periods_per_year)
        self.confidence_level = float(confidence_level)
        self.target_return = float(target_return)
        self.name = str(name)
        self.benchmark_name = str(benchmark_name)

        # Process returns and weights
        self.asset_returns: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.Series] = None
        self.portfolio_returns: pd.Series = self._process_inputs(returns, weights)

        # Process benchmark returns
        self.benchmark_returns: Optional[pd.Series] = None
        if benchmark_returns is not None:
            self._process_benchmark(benchmark_returns)

    def _process_inputs(
        self,
        returns: Any,
        weights: Optional[Any],
    ) -> pd.Series:
        """Parse and validate returns and asset weights."""
        if isinstance(returns, dict):
            df = pd.DataFrame(returns)
        elif isinstance(returns, pd.DataFrame):
            df = returns.copy()
        elif isinstance(returns, pd.Series):
            df = returns.to_frame()
        elif isinstance(returns, np.ndarray):
            if returns.ndim == 1:
                df = pd.DataFrame({self.name: returns})
            else:
                cols = [f"Asset_{i+1}" for i in range(returns.shape[1])]
                df = pd.DataFrame(returns, columns=cols)
        elif isinstance(returns, (list, tuple)):
            df = pd.DataFrame({self.name: list(returns)})
        else:
            df = pd.DataFrame(returns)

        # Clean NaN rows across assets
        df = df.dropna(how="all")

        if df.shape[1] == 1:
            # Single asset / portfolio return
            series = df.iloc[:, 0].astype(float).dropna()
            self.asset_returns = df
            self.weights = pd.Series([1.0], index=[df.columns[0]])
            return series

        # Multiple assets
        self.asset_returns = df
        num_assets = df.shape[1]

        if weights is None:
            w = np.ones(num_assets) / num_assets
            self.weights = pd.Series(w, index=df.columns)
        elif isinstance(weights, dict):
            w_list = [weights.get(col, 0.0) for col in df.columns]
            w_arr = np.array(w_list, dtype=float)
            total_w = np.sum(w_arr)
            if total_w > 0:
                w_arr = w_arr / total_w
            self.weights = pd.Series(w_arr, index=df.columns)
        elif isinstance(weights, pd.Series):
            aligned_w = weights.reindex(df.columns).fillna(0.0).values.astype(float)
            total_w = np.sum(aligned_w)
            if total_w > 0:
                aligned_w = aligned_w / total_w
            self.weights = pd.Series(aligned_w, index=df.columns)
        else:
            w_arr = np.asarray(weights, dtype=float)
            if len(w_arr) != num_assets:
                raise ValueError(f"Length of weights ({len(w_arr)}) must match number of assets ({num_assets})")
            total_w = np.sum(w_arr)
            if total_w > 0:
                w_arr = w_arr / total_w
            self.weights = pd.Series(w_arr, index=df.columns)

        # Calculate weighted portfolio returns
        weighted_returns = (df * self.weights).sum(axis=1).astype(float)
        weighted_returns.name = self.name
        return weighted_returns

    def _process_benchmark(self, benchmark_returns: Any) -> None:
        """Parse and align benchmark return series."""
        if isinstance(benchmark_returns, pd.DataFrame):
            b_series = benchmark_returns.iloc[:, 0].astype(float)
        elif isinstance(benchmark_returns, pd.Series):
            b_series = benchmark_returns.astype(float)
        elif isinstance(benchmark_returns, np.ndarray):
            b_series = pd.Series(benchmark_returns.squeeze(), dtype=float)
        else:
            b_series = pd.Series(list(benchmark_returns), dtype=float)

        b_series = b_series.dropna()
        b_series.name = self.benchmark_name
        self.benchmark_returns = b_series

    @property
    def has_benchmark(self) -> bool:
        """Whether a benchmark has been configured."""
        return self.benchmark_returns is not None and len(self.benchmark_returns) > 0

    def drawdown_table(self) -> pd.DataFrame:
        """Compute full drawdown series DataFrame for the portfolio."""
        return drawdown_series(self.portfolio_returns)

    def metrics(self) -> Dict[str, Any]:
        """Calculate and return a dictionary of all quantitative risk and performance metrics."""
        r = self.portfolio_returns
        py = self.periods_per_year
        rf = self.risk_free_rate
        cl = self.confidence_level
        mar = self.target_return

        # Basic Return & Volatility
        ann_ret = annualized_return(r, periods_per_year=py, geometric=True)
        ann_vol = annualized_volatility(r, periods_per_year=py)
        cum_ret = float((1.0 + r).prod() - 1.0) if len(r) > 0 else 0.0

        # Risk-Adjusted Ratios
        sharpe = sharpe_ratio(r, risk_free_rate=rf, periods_per_year=py)
        sortino = sortino_ratio(r, risk_free_rate=rf, target_return=mar, periods_per_year=py)
        calmar = calmar_ratio(r, periods_per_year=py)
        omega = omega_ratio(r, threshold=mar / py)
        tail = tail_ratio(r, upper_p=95.0, lower_p=5.0)

        # Drawdown Metrics
        mdd = max_drawdown(r)
        mdd_dur = max_drawdown_duration(r)

        # Value at Risk (VaR)
        var_hist = value_at_risk(r, confidence_level=cl, method="historical")
        var_param = value_at_risk(r, confidence_level=cl, method="parametric")
        var_cf = value_at_risk(r, confidence_level=cl, method="cornish_fisher")
        var_mc = value_at_risk(r, confidence_level=cl, method="monte_carlo", random_state=42)

        # Conditional VaR (CVaR / Expected Shortfall)
        cvar_hist = conditional_value_at_risk(r, confidence_level=cl, method="historical")
        cvar_param = conditional_value_at_risk(r, confidence_level=cl, method="parametric")

        # Distribution Metrics
        skew = skewness(r)
        kurt = excess_kurtosis(r)
        win_r = win_rate(r)
        best_d = float(r.max()) if len(r) > 0 else 0.0
        worst_d = float(r.min()) if len(r) > 0 else 0.0
        gl_ratio = gain_loss_ratio(r)

        results: Dict[str, Any] = {
            "portfolio_name": self.name,
            "periods_count": len(r),
            "periods_per_year": py,
            "risk_free_rate": rf,
            "confidence_level": cl,
            "annualized_return": ann_ret,
            "cumulative_return": cum_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": mdd,
            "max_drawdown_duration": mdd_dur,
            "var_historical": var_hist,
            "var_parametric": var_param,
            "var_cornish_fisher": var_cf,
            "var_monte_carlo": var_mc,
            "cvar_historical": cvar_hist,
            "cvar_parametric": cvar_param,
            "omega_ratio": omega,
            "tail_ratio": tail,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "win_rate": win_r,
            "best_day": best_d,
            "worst_day": worst_d,
            "gain_loss_ratio": gl_ratio,
        }

        # Benchmark Metrics
        if self.has_benchmark and self.benchmark_returns is not None:
            rb = self.benchmark_returns
            results.update(
                {
                    "benchmark_name": self.benchmark_name,
                    "benchmark_annualized_return": annualized_return(rb, periods_per_year=py),
                    "benchmark_annualized_volatility": annualized_volatility(rb, periods_per_year=py),
                    "realized_beta": realized_beta(r, rb),
                    "jensens_alpha": jensens_alpha(r, rb, risk_free_rate=rf, periods_per_year=py),
                    "tracking_error": tracking_error(r, rb, periods_per_year=py),
                    "information_ratio": information_ratio(r, rb, periods_per_year=py),
                }
            )

        return results

    def summary(self, as_dataframe: bool = True) -> Union[pd.DataFrame, Dict[str, Any]]:
        """Generate a structured summary table of all calculated risk and performance metrics.

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, returns a pandas DataFrame. If False, returns a dictionary.

        Returns
        -------
        pd.DataFrame or dict
            Structured summary.
        """
        m = self.metrics()
        if not as_dataframe:
            return m

        cl_pct = int(self.confidence_level * 100)
        rows: List[Dict[str, Any]] = [
            # Performance
            {"Category": "Performance", "Metric": "Annualized Return (CAGR)", "Value": m["annualized_return"], "Formatted": f"{m['annualized_return']:+.2%}", "Description": "Compound Annual Growth Rate"},
            {"Category": "Performance", "Metric": "Cumulative Total Return", "Value": m["cumulative_return"], "Formatted": f"{m['cumulative_return']:+.2%}", "Description": "Total cumulative growth over period"},
            {"Category": "Performance", "Metric": "Win Rate (Hit Ratio)", "Value": m["win_rate"], "Formatted": f"{m['win_rate']:.2%}", "Description": "Fraction of positive return days"},
            {"Category": "Performance", "Metric": "Best Single Period", "Value": m["best_day"], "Formatted": f"{m['best_day']:+.2%}", "Description": "Maximum daily return"},
            {"Category": "Performance", "Metric": "Worst Single Period", "Value": m["worst_day"], "Formatted": f"{m['worst_day']:+.2%}", "Description": "Minimum daily return"},

            # Risk & Volatility
            {"Category": "Risk & Volatility", "Metric": "Annualized Volatility", "Value": m["annualized_volatility"], "Formatted": f"{m['annualized_volatility']:.2%}", "Description": "Annualized sample standard deviation"},
            {"Category": "Risk & Volatility", "Metric": "Sharpe Ratio", "Value": m["sharpe_ratio"], "Formatted": f"{m['sharpe_ratio']:.2f}", "Description": f"Excess return per unit vol (Rf={self.risk_free_rate:.1%})"},
            {"Category": "Risk & Volatility", "Metric": "Sortino Ratio", "Value": m["sortino_ratio"], "Formatted": f"{m['sortino_ratio']:.2f}", "Description": f"Excess return per unit downside deviation (MAR={self.target_return:.1%})"},
            {"Category": "Risk & Volatility", "Metric": "Calmar Ratio", "Value": m["calmar_ratio"], "Formatted": f"{m['calmar_ratio']:.2f}", "Description": "Annualized return / Max Drawdown"},
            {"Category": "Risk & Volatility", "Metric": "Max Drawdown", "Value": m["max_drawdown"], "Formatted": f"{m['max_drawdown']:.2%}", "Description": "Worst peak-to-trough decline"},
            {"Category": "Risk & Volatility", "Metric": "Max Drawdown Duration", "Value": m["max_drawdown_duration"], "Formatted": f"{m['max_drawdown_duration']} periods", "Description": "Longest recovery period in days"},

            # Tail Risk / VaR
            {"Category": "Tail Risk & VaR", "Metric": f"Historical VaR ({cl_pct}%)", "Value": m["var_historical"], "Formatted": f"{m['var_historical']:+.2%}", "Description": f"Empirical {100-cl_pct}% quantile return"},
            {"Category": "Tail Risk & VaR", "Metric": f"Parametric Gaussian VaR ({cl_pct}%)", "Value": m["var_parametric"], "Formatted": f"{m['var_parametric']:+.2%}", "Description": "Normal distribution VaR cutoff"},
            {"Category": "Tail Risk & VaR", "Metric": f"Cornish-Fisher VaR ({cl_pct}%)", "Value": m["var_cornish_fisher"], "Formatted": f"{m['var_cornish_fisher']:+.2%}", "Description": "Modified VaR adjusted for skew/kurtosis"},
            {"Category": "Tail Risk & VaR", "Metric": f"Monte Carlo VaR ({cl_pct}%)", "Value": m["var_monte_carlo"], "Formatted": f"{m['var_monte_carlo']:+.2%}", "Description": "Simulated path empirical quantile"},
            {"Category": "Tail Risk & VaR", "Metric": f"Historical CVaR / ES ({cl_pct}%)", "Value": m["cvar_historical"], "Formatted": f"{m['cvar_historical']:+.2%}", "Description": "Expected Shortfall below Historical VaR"},
            {"Category": "Tail Risk & VaR", "Metric": f"Parametric Gaussian CVaR ({cl_pct}%)", "Value": m["cvar_parametric"], "Formatted": f"{m['cvar_parametric']:+.2%}", "Description": "Analytical Gaussian Expected Shortfall"},
            {"Category": "Tail Risk & VaR", "Metric": "Omega Ratio", "Value": m["omega_ratio"], "Formatted": f"{m['omega_ratio']:.2f}", "Description": "Probability weighted gains over losses"},
            {"Category": "Tail Risk & VaR", "Metric": "Tail Ratio", "Value": m["tail_ratio"], "Formatted": f"{m['tail_ratio']:.2f}", "Description": "95th percentile / abs(5th percentile)"},

            # Distribution
            {"Category": "Distribution", "Metric": "Skewness", "Value": m["skewness"], "Formatted": f"{m['skewness']:.2f}", "Description": "3rd standardized moment (asymmetry)"},
            {"Category": "Distribution", "Metric": "Excess Kurtosis", "Value": m["excess_kurtosis"], "Formatted": f"{m['excess_kurtosis']:.2f}", "Description": "4th moment fat-tail metric (normal=0)"},
            {"Category": "Distribution", "Metric": "Gain / Loss Ratio", "Value": m["gain_loss_ratio"], "Formatted": f"{m['gain_loss_ratio']:.2f}", "Description": "Average win / average loss"},
        ]

        if self.has_benchmark:
            rows.extend([
                {"Category": "Benchmark-Relative", "Metric": "Realized Beta", "Value": m["realized_beta"], "Formatted": f"{m['realized_beta']:.2f}", "Description": f"Sensitivity to {self.benchmark_name}"},
                {"Category": "Benchmark-Relative", "Metric": "Jensen's Alpha", "Value": m["jensens_alpha"], "Formatted": f"{m['jensens_alpha']:+.2%}", "Description": f"Annualized excess return over CAPM"},
                {"Category": "Benchmark-Relative", "Metric": "Tracking Error", "Value": m["tracking_error"], "Formatted": f"{m['tracking_error']:.2%}", "Description": f"Annualized std dev of excess returns"},
                {"Category": "Benchmark-Relative", "Metric": "Information Ratio", "Value": m["information_ratio"], "Formatted": f"{m['information_ratio']:.2f}", "Description": "Active return / Tracking Error"},
            ])

        df_summary = pd.DataFrame(rows)
        return df_summary

    def print_dashboard(self, title: Optional[str] = None, width: int = 78) -> str:
        """Render a formatted terminal dashboard card displaying all key risk metrics.

        Parameters
        ----------
        title : str, optional
            Custom title for the dashboard header.
        width : int, default 78
            Total width of the terminal card.

        Returns
        -------
        str
            Rendered string output.
        """
        dash_title = title or f"PORTFOLIO RISK & PERFORMANCE DASHBOARD: {self.name.upper()}"
        card_width = max(width, 74)

        def box_border(left: str, fill: str, right: str) -> str:
            return left + fill * (card_width - 2) + right

        def center_text(text: str) -> str:
            t = f" {text.strip()} "
            pad = max(0, card_width - 2 - len(t))
            l_pad = pad // 2
            r_pad = pad - l_pad
            return "│" + " " * l_pad + t + " " * r_pad + "│"

        def section_divider(sec_title: str) -> str:
            st = f" {sec_title.strip()} "
            rem = max(0, card_width - 4 - len(st))
            return "├─" + st + "─" * rem + "┤"

        def row(label: str, val_str: str, desc: str = "") -> str:
            col1 = label[:26].ljust(26)
            col2 = val_str[:16].rjust(16)
            col3 = desc[:28].ljust(28)
            inner = f"│  {col1} {col2}   {col3}"
            inner = inner[: card_width - 1].ljust(card_width - 1) + "│"
            return inner

        m = self.metrics()
        cl_pct = int(self.confidence_level * 100)

        lines: List[str] = []
        lines.append(box_border("┌", "─", "┐"))
        lines.append(center_text(dash_title))
        lines.append(center_text(f"Periods: {m['periods_count']} | Ann Frequency: {self.periods_per_year} | Rf: {self.risk_free_rate:.1%}"))

        # Section 1: Returns & Performance
        lines.append(section_divider("PORTFOLIO PERFORMANCE & RETURN METRICS"))
        lines.append(row("Annualized Return (CAGR)", f"{m['annualized_return']:+.2%}", "Compound annual growth rate"))
        lines.append(row("Cumulative Total Return", f"{m['cumulative_return']:+.2%}", "Total wealth growth"))
        lines.append(row("Win Rate (Hit Ratio)", f"{m['win_rate']:.1%}", "Days with positive returns"))
        lines.append(row("Best / Worst Day", f"{m['best_day']:+.2%} / {m['worst_day']:+.2%}", "Daily return bounds"))

        # Section 2: Volatility & Downside Risk
        lines.append(section_divider("VOLATILITY & DOWNSIDE RISK METRICS"))
        lines.append(row("Annualized Volatility", f"{m['annualized_volatility']:.2%}", "Sample std dev (ann)"))
        lines.append(row("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}", f"Excess return / vol (Rf={self.risk_free_rate:.1%})"))
        lines.append(row("Sortino Ratio", f"{m['sortino_ratio']:.2f}", f"Excess return / downside dev"))
        lines.append(row("Calmar Ratio", f"{m['calmar_ratio']:.2f}", "Ann Return / Absolute Max DD"))
        lines.append(row("Max Drawdown", f"{m['max_drawdown']:+.2%}", "Deepest peak-to-trough drop"))
        lines.append(row("Max Drawdown Duration", f"{m['max_drawdown_duration']} days", "Longest recovery period"))

        # Section 3: Tail Risk & VaR
        lines.append(section_divider(f"TAIL RISK & VALUE AT RISK ({cl_pct}% CONFIDENCE)"))
        lines.append(row(f"Historical VaR ({cl_pct}%)", f"{m['var_historical']:+.2%}", f"Empirical {100-cl_pct}% cutoff"))
        lines.append(row(f"Parametric Gaussian VaR", f"{m['var_parametric']:+.2%}", "mu - z_alpha * sigma"))
        lines.append(row(f"Cornish-Fisher VaR", f"{m['var_cornish_fisher']:+.2%}", "Modified for skew/kurtosis"))
        lines.append(row(f"Monte Carlo VaR", f"{m['var_monte_carlo']:+.2%}", "Simulated 100k paths"))
        lines.append(row(f"Historical CVaR / ES", f"{m['cvar_historical']:+.2%}", "Expected Shortfall"))
        lines.append(row(f"Parametric CVaR / ES", f"{m['cvar_parametric']:+.2%}", "Gaussian Expected Shortfall"))
        lines.append(row("Omega Ratio", f"{m['omega_ratio']:.2f}", "Gains / losses probability mass"))
        lines.append(row("Tail Ratio", f"{m['tail_ratio']:.2f}", "95th pct / |5th pct|"))

        # Section 4: Distribution Characteristics
        lines.append(section_divider("RETURN DISTRIBUTION CHARACTERISTICS"))
        lines.append(row("Skewness", f"{m['skewness']:.2f}", "Negative = left fat tail"))
        lines.append(row("Excess Kurtosis", f"{m['excess_kurtosis']:.2f}", "Fat-tail metric (normal=0)"))
        lines.append(row("Gain / Loss Ratio", f"{m['gain_loss_ratio']:.2f}", "Avg win / avg loss"))

        # Section 5: Benchmark Relative
        if self.has_benchmark:
            lines.append(section_divider(f"BENCHMARK-RELATIVE METRICS ({self.benchmark_name})"))
            lines.append(row("Benchmark Ann Return", f"{m['benchmark_annualized_return']:+.2%}", f"{self.benchmark_name} CAGR"))
            lines.append(row("Benchmark Ann Vol", f"{m['benchmark_annualized_volatility']:.2%}", f"{self.benchmark_name} volatility"))
            lines.append(row("Realized Beta", f"{m['realized_beta']:.2f}", "Cov(Rp, Rb) / Var(Rb)"))
            lines.append(row("Jensen's Alpha", f"{m['jensens_alpha']:+.2%}", "Annualized alpha over CAPM"))
            lines.append(row("Tracking Error", f"{m['tracking_error']:.2%}", "std(Rp - Rb) * sqrt(252)"))
            lines.append(row("Information Ratio", f"{m['information_ratio']:.2f}", "Active Return / Tracking Error"))

        # Section 6: Allocation breakdown (if multiple assets)
        if self.asset_returns is not None and self.asset_returns.shape[1] > 1 and self.weights is not None:
            lines.append(section_divider("ASSET ALLOCATION BREAKDOWN"))
            for asset, wt in self.weights.items():
                lines.append(row(str(asset), f"{wt:.1%}", f"Target portfolio weight"))

        lines.append(box_border("└", "─", "┘"))

        newline = chr(10)
        rendered = newline.join(lines)
        print(rendered)
        return rendered
