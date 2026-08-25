"""Cross-Economy FX Carry Trade, Interest Rate Parity & Volatility Surface Engine (Project 49).

Implements Covered Interest Rate Parity (CIP), Uncovered Interest Rate Parity (UIP),
Fama Forward Rate Bias regression, Malz (1997) FX options volatility surface interpolation,
and systematic G10 & EM FX Carry Trade strategy with crash-risk volatility filtering.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ParityResult:
    """Results of CIP and UIP parity computations."""
    spot_rate: float
    forward_market: float
    r_domestic: float
    r_foreign: float
    tenor_years: float
    cip_theoretical_forward: float
    cip_basis_bps: float
    uip_expected_spot: float
    forward_premium_pct: float
    carry_yield_spread_pct: float
    is_cip_arbitrage_profitable: bool
    summary_dict: Dict[str, Any]


@dataclass
class FamaRegressionResult:
    """Results of Fama (1984) forward rate bias regression: (S_{t+1}-S_t)/S_t = alpha + beta * (F_t-S_t)/S_t."""
    alpha: float
    beta: float
    r_squared: float
    t_stat_beta_zero: float              # Test H0: beta = 0
    p_val_beta_zero: float
    t_stat_beta_one: float               # Test H0: beta = 1 (Pure UIP Hypothesis)
    p_val_beta_one: float
    std_error_beta: float
    is_forward_bias_present: bool        # True if beta < 1.0 (Forward rate puzzle)
    summary_dataframe: pd.DataFrame


@dataclass
class MalzVolSurfaceResult:
    """Malz (1997) FX implied volatility surface parametrization."""
    atm_vol: float                       # At-the-money straddle volatility
    risk_reversal_25: float              # 25-delta Risk Reversal (skew: Call vol - Put vol)
    butterfly_25: float                  # 25-delta Butterfly (kurtosis/smile curvature)
    call_25d_vol: float
    put_25d_vol: float
    vol_by_delta: Dict[float, float]
    skewness_proxy: float
    kurtosis_proxy: float
    summary_dataframe: pd.DataFrame


@dataclass
class FXCarryStrategyResult:
    """Backtest results for institutional FX carry strategy with crash-risk filtering."""
    dates: pd.DatetimeIndex
    cumulative_equity: pd.Series
    daily_returns: pd.Series
    spot_returns: pd.Series
    carry_returns: pd.Series
    weights_matrix: pd.DataFrame
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    win_rate: float
    annualized_turnover: float
    metrics_table: pd.DataFrame


class FXCarryParityEngine:
    """Engine for interest rate parity, forward bias econometrics, and systematic FX carry trading."""

    def __init__(self, default_transaction_cost_bps: float = 2.0):
        self.tx_cost_bps = default_transaction_cost_bps

    def calculate_interest_rate_parity(
        self,
        spot_rate: float,
        r_domestic: float,
        r_foreign: float,
        tenor_years: float = 1.0,
        forward_market: Optional[float] = None,
        borrowing_spread_bps: float = 10.0,
    ) -> ParityResult:
        """Calculates Covered Interest Parity (CIP) and Uncovered Interest Parity (UIP)."""
        # CIP Theoretical Forward: F = S * (1 + r_d * tau) / (1 + r_f * tau)
        cip_fwd = spot_rate * (1.0 + r_domestic * tenor_years) / (1.0 + r_foreign * tenor_years)
        
        fwd_actual = forward_market if forward_market is not None else cip_fwd
        
        # Cross-currency basis in basis points (F_actual - F_cip) / S * 10,000
        cip_basis_bps = (fwd_actual - cip_fwd) / spot_rate * 10000.0
        
        # UIP Expected Spot rate (identical to CIP under risk neutrality)
        uip_spot = cip_fwd
        
        forward_prem_pct = (fwd_actual - spot_rate) / spot_rate * 100.0
        carry_spread_pct = (r_foreign - r_domestic) * 100.0
        
        # Arbitrage check: basis exceeds transaction & borrowing frictions
        is_arb = abs(cip_basis_bps) > (borrowing_spread_bps + 2.0 * self.tx_cost_bps)

        summary = {
            "Spot_Rate": spot_rate,
            "Forward_Market": fwd_actual,
            "CIP_Theoretical_Forward": cip_fwd,
            "CIP_Basis_bps": cip_basis_bps,
            "Domestic_Rate_pct": r_domestic * 100.0,
            "Foreign_Rate_pct": r_foreign * 100.0,
            "Carry_Yield_Spread_pct": carry_spread_pct,
            "Forward_Premium_pct": forward_prem_pct,
            "Arbitrage_Profitable": is_arb,
        }

        return ParityResult(
            spot_rate=spot_rate,
            forward_market=fwd_actual,
            r_domestic=r_domestic,
            r_foreign=r_foreign,
            tenor_years=tenor_years,
            cip_theoretical_forward=cip_fwd,
            cip_basis_bps=cip_basis_bps,
            uip_expected_spot=uip_spot,
            forward_premium_pct=forward_prem_pct,
            carry_yield_spread_pct=carry_spread_pct,
            is_cip_arbitrage_profitable=is_arb,
            summary_dict=summary,
        )

    def fama_forward_rate_bias_regression(
        self,
        spot_series: pd.Series,
        forward_series: pd.Series,
        horizon_steps: int = 1,
    ) -> FamaRegressionResult:
        """Runs the Fama (1984) regression to test the Forward Rate Unbiasedness Hypothesis.
        
        Regression specification:
            (S_{t+1} - S_t) / S_t = alpha + beta * ((F_t - S_t) / S_t) + epsilon_{t+1}
            
        Under pure UIP, alpha = 0 and beta = 1.0.
        Empirically in currency markets, beta is frequently < 0 (the Forward Premium Puzzle).
        """
        common_idx = spot_series.dropna().index.intersection(forward_series.dropna().index)
        s = spot_series.loc[common_idx]
        f = forward_series.loc[common_idx]

        # Dependent variable: realized forward spot depreciation / appreciation
        y_realized_depreciation = (s.shift(-horizon_steps) - s) / s
        
        # Independent variable: forward premium / discount
        x_forward_premium = (f - s) / s

        # Drop NaN caused by shift
        valid_mask = ~(y_realized_depreciation.isna() | x_forward_premium.isna())
        y_clean = y_realized_depreciation[valid_mask].values
        x_clean = x_forward_premium[valid_mask].values

        n = len(y_clean)
        if n < 10:
            raise ValueError("Insufficient data points for Fama regression.")

        # OLS fit
        slope, intercept, r_value, p_val_zero, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value ** 2

        # Test H0: beta = 0
        t_stat_zero = slope / (std_err + 1e-8)

        # Test H0: beta = 1 (UIP hypothesis)
        t_stat_one = (slope - 1.0) / (std_err + 1e-8)
        p_val_one = 2.0 * (1.0 - stats.t.cdf(abs(t_stat_one), df=n - 2))

        is_bias_present = bool(slope < 1.0)

        df_summary = pd.DataFrame({
            "Parameter": ["Alpha (Intercept)", "Beta (Slope)", "R-Squared", "t-Stat (Beta=0)", "p-Val (Beta=0)", "t-Stat (Beta=1, UIP)", "p-Val (Beta=1)"],
            "Value": [
                f"{intercept:+.6f}",
                f"{slope:+.4f}",
                f"{r_squared:.4f}",
                f"{t_stat_zero:.2f}",
                f"{p_val_zero:.4e}",
                f"{t_stat_one:.2f}",
                f"{p_val_one:.4e}"
            ],
            "Interpretation": [
                "Unconditional drift",
                "Forward rate elasticity (< 0 confirms puzzle)",
                "Explanatory power",
                "Reject H0: beta=0" if p_val_zero < 0.05 else "Fail to reject",
                "Statistical significance",
                "Reject UIP (beta != 1)" if p_val_one < 0.05 else "Consistent with UIP",
                "UIP validity test"
            ]
        })

        return FamaRegressionResult(
            alpha=float(intercept),
            beta=float(slope),
            r_squared=float(r_squared),
            t_stat_beta_zero=float(t_stat_zero),
            p_val_beta_zero=float(p_val_zero),
            t_stat_beta_one=float(t_stat_one),
            p_val_beta_one=float(p_val_one),
            std_error_beta=float(std_err),
            is_forward_bias_present=is_bias_present,
            summary_dataframe=df_summary,
        )

    def fit_malz_vol_surface(
        self,
        atm_vol: float,
        risk_reversal_25: float,
        butterfly_25: float,
        delta_grid: Optional[List[float]] = None,
    ) -> MalzVolSurfaceResult:
        """Parametrizes FX implied volatility smile using the Malz (1997) delta model.
        
        Formula:
            sigma(Delta) = sigma_ATM - 2 * RR_25 * (Delta - 0.5) + 16 * BF_25 * (Delta - 0.5)^2
            
        where:
            - sigma_ATM is the at-the-money volatility
            - RR_25 = sigma(25d Call) - sigma(25d Put) captures distribution skewness / crash asymmetry
            - BF_25 = 0.5 * (sigma(25d Call) + sigma(25d Put)) - sigma_ATM captures kurtosis / fat tails
        """
        if delta_grid is None:
            delta_grid = [0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90]

        call_25d = atm_vol + butterfly_25 + 0.5 * risk_reversal_25
        put_25d = atm_vol + butterfly_25 - 0.5 * risk_reversal_25

        vols_by_delta = {}
        for delta in delta_grid:
            d_shift = delta - 0.50
            sigma_d = atm_vol - 2.0 * risk_reversal_25 * d_shift + 16.0 * butterfly_25 * (d_shift ** 2)
            vols_by_delta[delta] = float(np.clip(sigma_d, 0.01, 1.50))

        # Skewness proxy is proportional to -RR (high negative RR means deep out-of-the-money puts are expensive)
        skew_proxy = float(-risk_reversal_25 / (atm_vol + 1e-6))
        kurt_proxy = float(butterfly_25 / (atm_vol + 1e-6) * 4.0)

        df_summary = pd.DataFrame({
            "Delta": [f"{int(d*100)}D" if d != 0.5 else "50D (ATM)" for d in delta_grid],
            "Delta_Numeric": delta_grid,
            "Implied_Volatility_pct": [f"{vols_by_delta[d]*100:.2f}%" for d in delta_grid],
            "Vol_Decimal": [vols_by_delta[d] for d in delta_grid],
        })

        return MalzVolSurfaceResult(
            atm_vol=atm_vol,
            risk_reversal_25=risk_reversal_25,
            butterfly_25=butterfly_25,
            call_25d_vol=call_25d,
            put_25d_vol=put_25d,
            vol_by_delta=vols_by_delta,
            skewness_proxy=skew_proxy,
            kurtosis_proxy=kurt_proxy,
            summary_dataframe=df_summary,
        )

    def backtest_fx_carry_strategy(
        self,
        fx_spot_df: pd.DataFrame,
        interest_rates_df: pd.DataFrame,
        funding_currencies: Optional[List[str]] = None,
        target_currencies: Optional[List[str]] = None,
        vol_filter: bool = True,
        vol_lookback: int = 20,
        max_vol_threshold: float = 0.18,
        target_annual_vol: float = 0.10,
        rebalance_freq: int = 5,
        risk_free_rate: float = 0.02,
    ) -> FXCarryStrategyResult:
        """Backtests institutional multi-currency G10 & EM FX Carry Strategy with crash risk filtering.
        
        Strategy Mechanics:
            1. Ranks all currencies cross-sectionally by short-term interest rates.
            2. Goes Long Top N high-yielding target currencies (e.g. BRL, MXN, ZAR, INR).
            3. Goes Short Bottom N low-yielding funding currencies (e.g. USD, EUR, JPY, CHF).
            4. Applies dynamic volatility targeting and circuit-breaker de-leveraging when FX vol surges.
        """
        common_idx = fx_spot_df.index.intersection(interest_rates_df.index)
        spots = fx_spot_df.loc[common_idx]
        rates = interest_rates_df.loc[common_idx]

        if funding_currencies is None:
            funding_currencies = [c for c in ["USD", "EUR", "JPY", "CHF"] if c in spots.columns]
        if target_currencies is None:
            target_currencies = [c for c in ["BRL", "MXN", "INR", "ZAR", "TRY", "AUD", "NZD"] if c in spots.columns]

        all_currs = list(set(funding_currencies + target_currencies))
        available_currs = [c for c in all_currs if c in spots.columns and c in rates.columns]

        if len(available_currs) < 2:
            raise ValueError("Insufficient overlapping currencies in spot and rates matrices.")

        spots = spots[available_currs]
        rates = rates[available_currs]
        n_days, n_assets = spots.shape

        # Spot daily price returns (assuming quote is Domestic/Foreign or Foreign/USD)
        spot_pct_changes = spots.pct_change().fillna(0.0)
        daily_interest_yields = rates / 365.0

        weights = pd.DataFrame(0.0, index=common_idx, columns=available_currs)

        # Multi-currency ranking & position sizing
        for t in range(0, n_days, rebalance_freq):
            rates_row = rates.iloc[t]
            sorted_currs = rates_row.sort_values(ascending=True)

            n_select = max(1, len(sorted_currs) // 3)
            short_currs = sorted_currs.index[:n_select]
            long_currs = sorted_currs.index[-n_select:]

            w_row = pd.Series(0.0, index=available_currs)
            w_row[long_currs] = 0.5 / len(long_currs)
            w_row[short_currs] = -0.5 / len(short_currs)

            # Volatility filtering & crash protection
            if vol_filter and t >= vol_lookback:
                past_returns = spot_pct_changes.iloc[t - vol_lookback:t]
                port_vol = float((past_returns.dot(w_row)).std() * np.sqrt(252))

                if port_vol > max_vol_threshold:
                    # Emergency de-leverage: scale down by 75%
                    w_row = w_row * 0.25
                elif port_vol > 0.001:
                    vol_scale = min(1.5, target_annual_vol / port_vol)
                    w_row = w_row * vol_scale

            end_t = min(t + rebalance_freq, n_days)
            for step in range(t, end_t):
                weights.iloc[step] = w_row

        # Lag weights by 1 bar to prevent lookahead
        lagged_weights = weights.shift(1).fillna(0.0)

        # Turnover
        turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
        cost_drag = turnover * (self.tx_cost_bps / 10000.0)

        # Total Return Decomposition: Spot Return + Interest Carry Differential - Costs
        daily_spot_pnl = (lagged_weights * spot_pct_changes).sum(axis=1)
        daily_carry_pnl = (lagged_weights * daily_interest_yields).sum(axis=1)
        net_daily_returns = daily_spot_pnl + daily_carry_pnl - cost_drag

        cumulative_equity = (1.0 + net_daily_returns).cumprod()

        n_years = max(1.0 / 252.0, len(net_daily_returns) / 252.0)
        cagr = float((cumulative_equity.iloc[-1]) ** (1.0 / n_years) - 1.0)
        ann_vol = float(net_daily_returns.std() * np.sqrt(252))

        rf_daily = risk_free_rate / 252.0
        excess_ret = net_daily_returns - rf_daily
        sharpe = float(np.sqrt(252) * excess_ret.mean() / (net_daily_returns.std() + 1e-6))

        downside_ret = net_daily_returns[net_daily_returns < 0]
        downside_vol = float(downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 0 else ann_vol
        sortino = float(np.sqrt(252) * excess_ret.mean() / (downside_vol + 1e-6))

        peaks = cumulative_equity.cummax()
        drawdowns = (cumulative_equity - peaks) / peaks
        max_dd = float(drawdowns.min())
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0

        skewness = float(stats.skew(net_daily_returns))
        kurtosis = float(stats.kurtosis(net_daily_returns))
        win_rate = float((net_daily_returns > 0).sum() / (len(net_daily_returns) + 1e-6))
        turnover_ann = float(turnover.mean() * 252.0)

        metrics_df = pd.DataFrame({
            "Metric": [
                "Strategy CAGR",
                "Annualized Volatility",
                "Sharpe Ratio (Rf=2%)",
                "Sortino Ratio",
                "Calmar Ratio",
                "Maximum Drawdown",
                "Return Skewness (Crash Risk)",
                "Return Kurtosis (Fat Tails)",
                "Daily Win Rate",
                "Annual Carry Yield Spread",
                "Annualized Turnover",
                "Annual Cost Drag (bps)"
            ],
            "Value": [
                f"{cagr:+.2%}",
                f"{ann_vol:.2%}",
                f"{sharpe:.2f}",
                f"{sortino:.2f}",
                f"{calmar:.2f}",
                f"{max_dd:.2%}",
                f"{skewness:+.3f}",
                f"{kurtosis:+.3f}",
                f"{win_rate:.1%}",
                f"{daily_carry_pnl.mean() * 365 * 100:+.2f}%",
                f"{turnover_ann:.2f}",
                f"{turnover_ann * self.tx_cost_bps:.1f} bps"
            ]
        })

        return FXCarryStrategyResult(
            dates=common_idx,
            cumulative_equity=cumulative_equity,
            daily_returns=net_daily_returns,
            spot_returns=daily_spot_pnl,
            carry_returns=daily_carry_pnl,
            weights_matrix=lagged_weights,
            cagr=cagr,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate,
            annualized_turnover=turnover_ann,
            metrics_table=metrics_df,
        )
