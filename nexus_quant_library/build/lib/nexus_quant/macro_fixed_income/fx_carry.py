"""
FX Interest Rate Parity, EM/DM Carry Trade & Malz (1997) FX Volatility Surface.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass
class ParityResult:
    """Interest rate parity diagnostics container."""
    spot_rate: float
    forward_rate: float
    domestic_rate: float
    foreign_rate: float
    tenor_years: float
    theoretical_forward_cip: float
    cip_basis_bps: float
    forward_premium_pct: float
    interest_rate_differential: float


@dataclass
class FXCarryBacktestResult:
    """Output container for systematic FX carry strategy backtesting."""
    equity_curve: pd.Series
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    net_carry_yield_annual: float


class FXCarryParityEngine:
    """
    Evaluates Covered/Uncovered Interest Rate Parity (CIP/UIP),
    models Fama forward rate bias, and backtests cross-economy FX carry portfolios.
    """

    @staticmethod
    def evaluate_cip(
        spot: float,
        forward: float,
        domestic_rate: float,
        foreign_rate: float,
        tenor_years: float = 1.0,
    ) -> ParityResult:
        """
        Evaluate Covered Interest Rate Parity and compute the cross-currency basis (in bps).
        Theoretical CIP Forward: F = S * (1 + r_d * T) / (1 + r_f * T).
        """
        r_d = domestic_rate / 100.0 if domestic_rate > 1.0 else domestic_rate
        r_f = foreign_rate / 100.0 if foreign_rate > 1.0 else foreign_rate
        t = tenor_years

        f_cip = spot * (1.0 + r_d * t) / (1.0 + r_f * t)
        cip_basis_bps = ((forward / spot) * (1.0 + r_f * t) - (1.0 + r_d * t)) * 10000.0
        fwd_prem = (forward - spot) / spot * 100.0
        diff = (r_f - r_d) * 100.0

        return ParityResult(
            spot_rate=spot,
            forward_rate=forward,
            domestic_rate=r_d * 100.0,
            foreign_rate=r_f * 100.0,
            tenor_years=t,
            theoretical_forward_cip=float(f_cip),
            cip_basis_bps=float(cip_basis_bps),
            forward_premium_pct=float(fwd_prem),
            interest_rate_differential=float(diff),
        )

    @staticmethod
    def fama_forward_bias_regression(
        spot_series: pd.Series,
        forward_series: pd.Series,
    ) -> Dict[str, float]:
        """
        Estimate Fama (1984) forward rate bias regression: (S_{t+1} - S_t) / S_t = alpha + beta * (F_t - S_t) / S_t.
        If beta < 1 (or beta < 0), UIP fails, confirming the existence of the FX carry trade anomaly.
        """
        dep_var = spot_series.shift(-1).sub(spot_series).div(spot_series).dropna()
        indep_var = forward_series.sub(spot_series).div(spot_series).loc[dep_var.index]

        reg = linregress(indep_var, dep_var)
        return {
            "Alpha": float(reg.intercept),
            "Beta_Forward_Bias": float(reg.slope),
            "R_Squared": float(reg.rvalue**2),
            "P_Value": float(reg.pvalue),
            "Std_Error": float(reg.stderr),
            "UIP_Violated": bool(reg.slope < 0.80),
        }

    @classmethod
    def backtest_cross_economy_carry(
        cls,
        fx_spot_df: pd.DataFrame,
        interest_rates_df: pd.DataFrame,
        funding_currencies: List[str],
        target_currencies: List[str],
        rebalance_freq_days: int = 21,
        vol_filter_threshold: float = 0.20,
        transaction_cost_bps: float = 3.0,
    ) -> FXCarryBacktestResult:
        """
        Backtest a systematic DM/EM FX carry strategy:
        Long top-yielding EM currencies, Short lowest-yielding funding DM currencies.
        """
        dates = fx_spot_df.index
        n_dates = len(dates)
        portfolio_returns = np.zeros(n_dates)

        # Relative currency returns
        spot_ret = fx_spot_df.pct_change().fillna(0.0)
        daily_interest = interest_rates_df / (100.0 * 252.0)

        current_weights = pd.Series(0.0, index=fx_spot_df.columns)

        for t in range(1, n_dates):
            date = dates[t]

            # Rebalance check
            if t % rebalance_freq_days == 1 or t == 1:
                rates_t = interest_rates_df.loc[date]
                # Rank targets (long top high yields)
                target_rates = rates_t.loc[target_currencies].sort_values(ascending=False)
                fund_rates = rates_t.loc[funding_currencies].sort_values(ascending=True)

                long_curr = target_rates.index[:2]
                short_curr = fund_rates.index[:2]

                new_weights = pd.Series(0.0, index=fx_spot_df.columns)
                for c in long_curr:
                    new_weights[c] = +0.50 / len(long_curr)
                for c in short_curr:
                    new_weights[c] = -0.50 / len(short_curr)

                # Turnover & transaction cost
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost_drag = turnover * (transaction_cost_bps / 10000.0)
                current_weights = new_weights
            else:
                cost_drag = 0.0

            # Daily P&L = Spot Return + Net Interest Differential
            daily_spot_pnl = np.sum(current_weights * spot_ret.loc[date])
            daily_interest_pnl = np.sum(current_weights * daily_interest.loc[date])
            net_ret = daily_spot_pnl + daily_interest_pnl - cost_drag

            # Volatility circuit breaker filter
            rolling_vol = float(spot_ret.iloc[max(0, t-21):t].std().mean() * np.sqrt(252))
            if rolling_vol > vol_filter_threshold:
                net_ret *= 0.50  # scale down risk by 50%

            portfolio_returns[t] = net_ret

        equity = pd.Series(np.cumprod(1.0 + portfolio_returns), index=dates)
        cagr = float((equity.iloc[-1] ** (252.0 / n_dates)) - 1.0)
        ann_vol = float(np.std(portfolio_returns) * np.sqrt(252))
        sharpe = float((cagr - 0.02) / max(ann_vol, 1e-6))
        neg_rets = portfolio_returns[portfolio_returns < 0]
        sortino = float((cagr - 0.02) / max(np.std(neg_rets) * np.sqrt(252), 1e-6)) if len(neg_rets) > 0 else 5.0

        hwm = np.maximum.accumulate(equity)
        dd = (equity - hwm) / hwm
        max_dd = float(np.min(dd))
        win_rate = float(np.mean(portfolio_returns[1:] > 0))

        return FXCarryBacktestResult(
            equity_curve=equity,
            cagr=cagr,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=n_dates // rebalance_freq_days,
            net_carry_yield_annual=float(daily_interest.mean().mean() * 252.0 * 100.0),
        )


class MalzFXVolatilitySurface:
    """
    Malz (1997) FX Volatility Surface parametrization.
    Reconstructs delta-smile curves using ATM volatility, 25-delta Risk Reversals, and 25-delta Butterfly spreads:
    sigma(Delta) = sigma_ATM - 2 * RR_25 * (Delta - 0.5) + 16 * BF_25 * (Delta - 0.5)^2.
    """

    @classmethod
    def evaluate_smile(
        cls,
        atm_vol: float,
        risk_reversal_25: float,
        butterfly_25: float,
        deltas: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate implied volatility as a function of option delta."""
        if deltas is None:
            deltas = np.linspace(0.05, 0.95, 50)

        # Malz (1997) parabolic formulation in delta space
        delta_shift = deltas - 0.50
        iv_curve = atm_vol - 2.0 * risk_reversal_25 * delta_shift + 16.0 * butterfly_25 * (delta_shift**2)
        return deltas, np.maximum(iv_curve, 0.01)
