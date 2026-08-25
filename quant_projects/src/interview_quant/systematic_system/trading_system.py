"""Module 30: Full Production Systematic Trading System.

Integrates:
- Alpha Signal Engine (Multi-Factor Momentum & Mean-Reversion Ensemble)
- Risk Management Engine (Dynamic Volatility Targeting & Drawdown Circuit Breakers)
- Execution Engine (Almgren-Chriss Optimal Order Slicing & Friction Modeling)
- Stress Test Gating (Pre-trade tail risk verification)
- Out-of-Sample (OOS) Execution (2020-2024 compounding)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class AlmgrenChrissSchedule:
    total_shares: float
    time_horizon_hours: float
    n_intervals: int
    optimal_trade_list: np.ndarray
    inventory_trajectory: np.ndarray
    expected_cost_bps: float
    variance_of_cost: float


@dataclass
class StressGatingResult:
    passed: bool
    expected_gfc_loss_pct: float
    expected_covid_loss_pct: float
    max_stress_loss_pct: float
    max_allowed_loss_pct: float
    reason: str


@dataclass
class ProductionSystemResult:
    dates: pd.DatetimeIndex
    asset_prices: pd.DataFrame
    raw_signals: pd.DataFrame
    target_weights: pd.DataFrame
    circuit_breaker_active: pd.Series
    leverage_series: pd.Series
    daily_turnover: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    cumulative_equity: pd.Series
    drawdown_series: pd.Series
    metrics: Dict[str, Any]
    almgren_chriss_execution: AlmgrenChrissSchedule
    stress_gating: StressGatingResult


class ProductionTradingSystem:
    def __init__(
        self,
        target_annual_vol: float = 0.10,
        vol_lookback_days: int = 40,
        max_asset_weight: float = 0.25,
        max_gross_leverage: float = 1.8,
        drawdown_warning_level: float = 0.08,    # 8% drawdown -> scale leverage to 50%
        drawdown_circuit_breaker: float = 0.15,  # 15% drawdown -> halt & cooling off
        turnover_smoothing: float = 0.15,        # Exponential weight smoothing factor
        fee_bps: float = 5.0,
        half_spread_bps: float = 2.5,
        borrow_cost_bps: float = 50.0,
        risk_free_rate: float = 0.02,
    ):
        self.target_annual_vol = target_annual_vol
        self.vol_lookback_days = vol_lookback_days
        self.max_asset_weight = max_asset_weight
        self.max_gross_leverage = max_gross_leverage
        self.drawdown_warning_level = drawdown_warning_level
        self.drawdown_circuit_breaker = drawdown_circuit_breaker
        self.turnover_smoothing = turnover_smoothing
        self.fee_bps = fee_bps
        self.half_spread_bps = half_spread_bps
        self.borrow_cost_bps = borrow_cost_bps
        self.risk_free_rate = risk_free_rate

    # -------------------------------------------------------------------------
    # 1. ALPHA SIGNAL ENGINE (ENSEMBLE)
    # -------------------------------------------------------------------------
    def compute_alpha_signals(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generates ensemble alpha signals across momentum, mean reversion, and trend."""
        if isinstance(prices_df.columns, pd.MultiIndex):
            close = prices_df.xs("Close", level="Field", axis=1)
        else:
            close = prices_df

        tickers = list(close.columns)
        signals = pd.DataFrame(0.0, index=close.index, columns=tickers)

        for t in tickers:
            p = close[t]
            # 1. Multi-horizon Time-Series Momentum (TSMOM)
            r_1m = p.pct_change(21).fillna(0.0)
            r_3m = p.pct_change(63).fillna(0.0)
            r_6m = p.pct_change(126).fillna(0.0)
            r_12m = p.pct_change(252).fillna(0.0)

            mom_score = 0.25 * np.sign(r_1m) + 0.35 * np.sign(r_3m) + 0.25 * np.sign(r_6m) + 0.15 * np.sign(r_12m)

            # 2. Moving Average Trend Filter (50d vs 200d)
            sma_20 = p.rolling(20).mean().bfill()
            sma_50 = p.rolling(50).mean().bfill()
            sma_200 = p.rolling(200).mean().bfill()
            trend_filter = np.where(sma_50 > sma_200, 0.8, -0.2)

            # 3. Short-term Pullback / Mean Reversion in Trend
            std_20 = p.rolling(20).std().bfill().clip(lower=1e-4)
            z_score = (p - sma_20) / std_20
            mr_score = np.where((sma_50 > sma_200) & (z_score < -1.0), 0.5, 0.0)

            composite = 0.60 * mom_score + 0.25 * trend_filter + 0.15 * mr_score
            signals[t] = np.clip(composite, -1.0, 1.0)

        return signals.fillna(0.0)

    # -------------------------------------------------------------------------
    # 2. RISK MANAGEMENT ENGINE & CIRCUIT BREAKER
    # -------------------------------------------------------------------------
    def size_positions_with_risk_control(
        self,
        prices_df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Applies dynamic volatility targeting and trailing drawdown circuit breaker with cooling off."""
        if isinstance(prices_df.columns, pd.MultiIndex):
            close = prices_df.xs("Close", level="Field", axis=1)
        else:
            close = prices_df

        returns = close.pct_change().fillna(0.0)
        n_days = len(close)
        tickers = list(close.columns)

        raw_weights = pd.DataFrame(0.0, index=close.index, columns=tickers)
        circuit_active = pd.Series(False, index=close.index)

        running_equity = 100000.0
        hwm = running_equity
        cooling_off_counter = 0

        # Rolling volatility estimation
        rolling_vols = returns.rolling(self.vol_lookback_days).std() * np.sqrt(252.0)
        rolling_vols = rolling_vols.fillna(0.18).clip(lower=0.08)

        for i in range(1, n_days):
            date = close.index[i]
            date_prev = close.index[i-1]

            # 1. Update running equity
            if i > 1:
                w_prev = raw_weights.loc[date_prev].values
                r_curr = returns.loc[date].values
                day_ret = np.dot(w_prev, r_curr)
                running_equity = max(1000.0, running_equity * (1.0 + day_ret))
                hwm = max(hwm, running_equity)

            drawdown = (running_equity - hwm) / hwm

            # 2. Evaluate Circuit Breaker status
            multiplier = 1.0
            if cooling_off_counter > 0:
                cooling_off_counter -= 1
                multiplier = 0.50
                circuit_active.iloc[i] = True
            elif drawdown <= -self.drawdown_circuit_breaker:
                multiplier = 0.0
                cooling_off_counter = 10
                circuit_active.iloc[i] = True
                hwm = running_equity * 1.05
            elif drawdown <= -self.drawdown_warning_level:
                multiplier = 0.60
                circuit_active.iloc[i] = True

            # 3. Size positions by volatility target
            sig = signals_df.loc[date].values
            vols = rolling_vols.loc[date].values
            w_raw = sig * (self.target_annual_vol / (vols * np.sqrt(len(tickers)) * 0.5)) * multiplier

            # Enforce single asset cap and gross leverage cap
            w_capped = np.clip(w_raw, -self.max_asset_weight, self.max_asset_weight)
            gross_lev = np.sum(np.abs(w_capped))
            if gross_lev > self.max_gross_leverage:
                w_capped = w_capped * (self.max_gross_leverage / gross_lev)

            raw_weights.loc[date] = w_capped

        # Apply exponential turnover smoothing
        weights = raw_weights.ewm(alpha=self.turnover_smoothing).mean()
        leverage_series = weights.abs().sum(axis=1)

        return weights, circuit_active, leverage_series

    # -------------------------------------------------------------------------
    # 3. EXECUTION ENGINE: ALMGREN-CHRISS OPTIMAL SLICING
    # -------------------------------------------------------------------------
    def compute_almgren_chriss_schedule(
        self,
        total_shares: float = 100_000.0,
        time_horizon_hours: float = 6.5,
        n_intervals: int = 13,
        daily_vol: float = 0.015,
        temporary_impact_eta: float = 2.5e-6,
        permanent_impact_gamma: float = 1.0e-7,
        risk_aversion_lambda: float = 1.0e-5,
    ) -> AlmgrenChrissSchedule:
        """Computes Almgren-Chriss (2000) optimal trade trajectory minimizing cost + risk."""
        T = time_horizon_hours
        N = n_intervals
        dt = T / N

        kappa = np.sqrt(risk_aversion_lambda * (daily_vol**2) / max(temporary_impact_eta, 1e-9))
        kappa = max(0.1, min(kappa, 10.0))

        t_grid = np.linspace(0, T, N + 1)
        denom = np.sinh(kappa * T)
        if abs(denom) < 1e-9:
            inventory = np.linspace(total_shares, 0.0, N + 1)
        else:
            inventory = (np.sinh(kappa * (T - t_grid)) / denom) * total_shares

        trade_list = -np.diff(inventory)

        permanent_cost = 0.5 * permanent_impact_gamma * (total_shares**2)
        temp_cost = temporary_impact_eta * np.sum(trade_list**2 / dt)
        expected_cost = permanent_cost + temp_cost
        expected_cost_bps = float((expected_cost / (total_shares * 100.0)) * 10000.0)
        variance_cost = float(risk_aversion_lambda * (daily_vol**2) * np.sum(inventory[:-1]**2 * dt))

        return AlmgrenChrissSchedule(
            total_shares=total_shares,
            time_horizon_hours=time_horizon_hours,
            n_intervals=n_intervals,
            optimal_trade_list=trade_list,
            inventory_trajectory=inventory,
            expected_cost_bps=expected_cost_bps,
            variance_of_cost=variance_cost,
        )

    # -------------------------------------------------------------------------
    # 4. STRESS TEST GATING (PRE-TRADE RISK CHECK)
    # -------------------------------------------------------------------------
    def evaluate_stress_gating(
        self,
        target_weights: pd.Series,
        max_allowed_loss_pct: float = 0.15,
    ) -> StressGatingResult:
        """Pre-trade risk gating against extreme historical shocks."""
        w = target_weights.values
        gfc_shock_vector = np.array([-0.20, -0.28, -0.25, -0.22, -0.35, -0.24, -0.28, -0.30, -0.15, 0.08])
        if len(w) <= len(gfc_shock_vector):
            shocks_gfc = gfc_shock_vector[:len(w)]
        else:
            shocks_gfc = np.full(len(w), -0.20)

        covid_shock_vector = np.array([-0.30, -0.25, -0.20, -0.18, -0.32, -0.22, -0.20, -0.35, -0.45, 0.12])
        if len(w) <= len(covid_shock_vector):
            shocks_covid = covid_shock_vector[:len(w)]
        else:
            shocks_covid = np.full(len(w), -0.25)

        loss_gfc = float(np.abs(np.dot(w, shocks_gfc)))
        loss_covid = float(np.abs(np.dot(w, shocks_covid)))
        max_stress_loss = max(loss_gfc, loss_covid)

        passed = max_stress_loss <= max_allowed_loss_pct
        reason = "PASSED: Expected stress loss within tolerance" if passed else "FAILED: Stress loss exceeds threshold"

        return StressGatingResult(
            passed=passed,
            expected_gfc_loss_pct=loss_gfc,
            expected_covid_loss_pct=loss_covid,
            max_stress_loss_pct=max_stress_loss,
            max_allowed_loss_pct=max_allowed_loss_pct,
            reason=reason,
        )

    # -------------------------------------------------------------------------
    # 5. OUT-OF-SAMPLE END-TO-END EXECUTION
    # -------------------------------------------------------------------------
    def run_systematic_system(
        self,
        prices_df: pd.DataFrame,
        oos_start_date: str = "2020-01-01",
    ) -> ProductionSystemResult:
        """Executes full production system and generates OOS equity compounding."""
        if isinstance(prices_df.columns, pd.MultiIndex):
            close = prices_df.xs("Close", level="Field", axis=1)
        else:
            close = prices_df

        # 1. Compute alpha signals
        raw_signals = self.compute_alpha_signals(prices_df)

        # 2. Risk management & position sizing
        weights, circuit_active, leverage_series = self.size_positions_with_risk_control(close, raw_signals)

        # 3. Execution & backtest accounting
        returns = close.pct_change().fillna(0.0)
        w_lagged = weights.shift(1).fillna(0.0)
        gross_ret = (w_lagged * returns).sum(axis=1)

        delta_w = weights.diff().abs().fillna(weights.abs())
        turnover = delta_w.sum(axis=1)

        cost_bps = (self.fee_bps + self.half_spread_bps) / 10000.0
        slippage_gamma = 1e-4
        quadratic_slippage = 0.5 * slippage_gamma * (delta_w**2).sum(axis=1)
        linear_costs = cost_bps * turnover
        short_borrow = w_lagged.clip(upper=0.0).abs().sum(axis=1) * (self.borrow_cost_bps / 10000.0 / 252.0)

        total_costs = linear_costs + quadratic_slippage + short_borrow
        net_ret = gross_ret - total_costs

        # Filter to Out-of-Sample period if requested
        if pd.Timestamp(oos_start_date) in net_ret.index or pd.Timestamp(oos_start_date) <= net_ret.index[-1]:
            oos_idx = net_ret.loc[oos_start_date:].index
        else:
            oos_idx = net_ret.index

        net_ret_oos = net_ret.loc[oos_idx]
        gross_ret_oos = gross_ret.loc[oos_idx]
        turnover_oos = turnover.loc[oos_idx]

        cum_eq = (1.0 + net_ret_oos).cumprod() * 100000.0
        hwm = cum_eq.cummax()
        dd = (cum_eq - hwm) / hwm

        n_days = len(net_ret_oos)
        tot_ret = float(cum_eq.iloc[-1] / cum_eq.iloc[0] - 1.0)
        if 1.0 + tot_ret > 0.0:
            cagr = float((1.0 + tot_ret) ** (252.0 / max(1, n_days)) - 1.0)
        else:
            cagr = -1.0

        ann_vol = float(net_ret_oos.std() * np.sqrt(252.0))
        sharpe = float((cagr - self.risk_free_rate) / max(ann_vol, 1e-6))
        max_dd = float(dd.min())
        calmar = float(cagr / max(abs(max_dd), 1e-6))

        win_days = net_ret_oos[net_ret_oos > 0.0]
        loss_days = net_ret_oos[net_ret_oos < 0.0]
        win_rate = float(len(win_days) / max(1, len(net_ret_oos[net_ret_oos != 0.0])))
        profit_factor = float(win_days.sum() / max(abs(loss_days.sum()), 1e-6))

        metrics = {
            "CAGR": cagr,
            "Total Return": tot_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_dd,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Average Leverage": float(leverage_series.loc[oos_idx].mean()),
            "Annualized Turnover": float(turnover_oos.mean() * 252.0),
        }

        ac_schedule = self.compute_almgren_chriss_schedule(total_shares=50000.0, daily_vol=max(ann_vol, 0.10) / np.sqrt(252.0))
        stress_gate = self.evaluate_stress_gating(weights.iloc[-1])

        return ProductionSystemResult(
            dates=oos_idx,
            asset_prices=close.loc[oos_idx],
            raw_signals=raw_signals.loc[oos_idx],
            target_weights=weights.loc[oos_idx],
            circuit_breaker_active=circuit_active.loc[oos_idx],
            leverage_series=leverage_series.loc[oos_idx],
            daily_turnover=turnover_oos,
            gross_returns=gross_ret_oos,
            net_returns=net_ret_oos,
            cumulative_equity=cum_eq,
            drawdown_series=dd,
            metrics=metrics,
            almgren_chriss_execution=ac_schedule,
            stress_gating=stress_gate,
        )
