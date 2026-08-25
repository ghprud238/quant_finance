"""Crypto Perpetual Futures, Funding Rate Arbitrage & Basis Trading (Project 44).

Implements exchange 8-hour funding rate mechanics, premium index calculation,
and delta-neutral Cash-and-Carry basis arbitrage simulation with margin requirements,
liquidation buffers, and funding cashflow compounding.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class FundingRateSnapshot:
    """Represents a single 8-hour funding rate calculation interval."""
    timestamp: pd.Timestamp
    index_price: float
    mark_price: float
    impact_bid: float
    impact_ask: float
    interest_rate: float
    premium_index: float
    funding_rate_8h: float
    annualized_yield_pct: float


@dataclass
class FundingStatistics:
    """Comprehensive statistical metrics of a historical funding rate series."""
    mean_rate_8h: float
    median_rate_8h: float
    std_rate_8h: float
    annualized_yield_apy: float
    positive_funding_pct: float
    max_rate_8h: float
    min_rate_8h: float
    n_periods: int

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Metric": [
                "Mean 8h Funding Rate",
                "Annualized Funding Yield (APY)",
                "Median 8h Funding Rate",
                "Volatility (8h Std Dev)",
                "Positive Funding Frequency (%)",
                "Max 8h Rate (Peak Bull)",
                "Min 8h Rate (Deep Discount)",
                "Total 8h Intervals",
            ],
            "Value": [
                f"{self.mean_rate_8h:+.4%}",
                f"{self.annualized_yield_apy:+.2f}%",
                f"{self.median_rate_8h:+.4%}",
                f"{self.std_rate_8h:.4%}",
                f"{self.positive_funding_pct:.1f}%",
                f"{self.max_rate_8h:+.4%}",
                f"{self.min_rate_8h:+.4%}",
                f"{self.n_periods}",
            ]
        })


@dataclass
class BasisTradeResult:
    """Comprehensive performance and risk analytics for Cash-and-Carry basis arbitrage."""
    initial_capital_usd: float
    final_equity_usd: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    total_funding_collected_usd: float
    net_funding_yield_apy: float
    margin_calls_count: int
    peak_margin_utilization_pct: float
    equity_curve: pd.Series
    funding_cashflows: pd.Series
    margin_ratios: pd.Series
    drawdowns: pd.Series

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Metric": [
                "Initial Capital ($)",
                "Final Portfolio Equity ($)",
                "Total Return (%)",
                "CAGR (Annual Return)",
                "Annualized Volatility",
                "Sharpe Ratio (Rf=2%)",
                "Sortino Ratio",
                "Calmar Ratio",
                "Maximum Drawdown (%)",
                "Cumulative Funding Collected ($)",
                "Net Annualized Funding Yield (APY)",
                "Peak Margin Utilization (%)",
                "Margin Call Trigger Events",
            ],
            "Value": [
                f"${self.initial_capital_usd:,.2f}",
                f"${self.final_equity_usd:,.2f}",
                f"{self.total_return_pct:+.2f}%",
                f"{self.cagr_pct:+.2f}%",
                f"{self.annualized_volatility_pct:.2f}%",
                f"{self.sharpe_ratio:.2f}",
                f"{self.sortino_ratio:.2f}",
                f"{self.calmar_ratio:.2f}",
                f"{self.max_drawdown_pct:.2f}%",
                f"${self.total_funding_collected_usd:+,.2f}",
                f"{self.net_funding_yield_apy:+.2f}%",
                f"{self.peak_margin_utilization_pct:.1f}%",
                f"{self.margin_calls_count}",
            ]
        })


class PerpetualFundingEngine:
    """Calculates exchange perpetual futures funding rates and premium indices."""

    def __init__(
        self,
        default_interest_rate_8h: float = 0.0001,  # 0.01% standard base rate per 8h
        clamp_bounds: Tuple[float, float] = (-0.0075, 0.0075),  # +/- 0.75% standard clamp
    ):
        self.default_interest_rate_8h = default_interest_rate_8h
        self.clamp_bounds = clamp_bounds

    def compute_funding_rate(
        self,
        index_price: float,
        impact_bid: float,
        impact_ask: float,
        interest_rate_8h: Optional[float] = None,
    ) -> float:
        """Calculates 8-hour perpetual funding rate according to exchange mechanics:
        Premium Index = [max(0, Impact Bid - Index Price) - max(0, Index Price - Impact Ask)] / Index Price
        Funding Rate = Clamp(Premium Index + Clamp(Interest Rate - Premium Index, -0.05%, +0.05%), -0.75%, +0.75%)
        """
        if index_price <= 0.0:
            return 0.0
        
        ir = interest_rate_8h if interest_rate_8h is not None else self.default_interest_rate_8h
        
        premium_index = (max(0.0, impact_bid - index_price) - max(0.0, index_price - impact_ask)) / index_price
        rate_diff_clamped = np.clip(ir - premium_index, -0.0005, 0.0005)
        funding_rate = np.clip(premium_index + rate_diff_clamped, self.clamp_bounds[0], self.clamp_bounds[1])
        return float(funding_rate)

    def analyze_funding_series(self, funding_rates_8h: pd.Series) -> FundingStatistics:
        """Computes statistical metrics of historical funding rate data."""
        rates = funding_rates_8h.dropna().values
        if len(rates) == 0:
            return FundingStatistics(0, 0, 0, 0, 0, 0, 0, 0)
        
        mean_r = float(np.mean(rates))
        median_r = float(np.median(rates))
        std_r = float(np.std(rates))
        apy = mean_r * 3 * 365 * 100.0
        pos_pct = float(np.sum(rates > 0) / len(rates) * 100.0)
        max_r = float(np.max(rates))
        min_r = float(np.min(rates))

        return FundingStatistics(
            mean_rate_8h=mean_r,
            median_rate_8h=median_r,
            std_rate_8h=std_r,
            annualized_yield_apy=apy,
            positive_funding_pct=pos_pct,
            max_rate_8h=max_r,
            min_rate_8h=min_r,
            n_periods=len(rates),
        )


class CashAndCarryBasisTrader:
    """Simulates Delta-Neutral Cash-and-Carry basis arbitrage strategy."""

    def __init__(
        self,
        initial_capital_usd: float = 1_000_000.0,
        spot_allocation_pct: float = 0.50,  # 50% capital in spot, 50% in margin for short perp (2x leverage)
        initial_margin_rate: float = 0.10,  # 10% Initial Margin (10x max leverage)
        maintenance_margin_rate: float = 0.05,  # 5% Maintenance Margin (20x liquidation threshold)
        spot_trading_fee_bps: float = 5.0,  # 0.05% spot taker fee
        perp_trading_fee_bps: float = 2.0,  # 0.02% perp maker/taker fee
        staking_yield_apy: float = 0.035,   # +3.5% staking yield on spot collateral (e.g. stETH)
        risk_free_rate: float = 0.02,
    ):
        self.initial_capital_usd = initial_capital_usd
        self.spot_allocation_pct = spot_allocation_pct
        self.initial_margin_rate = initial_margin_rate
        self.maintenance_margin_rate = maintenance_margin_rate
        self.spot_trading_fee_bps = spot_trading_fee_bps
        self.perp_trading_fee_bps = perp_trading_fee_bps
        self.staking_yield_apy = staking_yield_apy
        self.risk_free_rate = risk_free_rate

    def backtest(
        self,
        df_market: pd.DataFrame,
    ) -> BasisTradeResult:
        """Executes full backtest of delta-neutral cash-and-carry basis strategy."""
        timestamps = df_market["Timestamp"].values if "Timestamp" in df_market.columns else df_market.index
        spot_prices = df_market["Spot_Price"].values
        perp_prices = df_market["Perp_Price"].values
        funding_rates = df_market["Funding_Rate_8h"].values
        n_steps = len(df_market)

        # Initial Position Sizing
        spot_capital = self.initial_capital_usd * self.spot_allocation_pct
        margin_cash = self.initial_capital_usd * (1.0 - self.spot_allocation_pct)

        # Pay entry fees
        spot_entry_fee = spot_capital * (self.spot_trading_fee_bps * 1e-4)
        spot_units = (spot_capital - spot_entry_fee) / spot_prices[0]
        
        # Delta-neutral: short perp notional equal to spot notional
        perp_notional_initial = spot_units * perp_prices[0]
        perp_entry_fee = perp_notional_initial * (self.perp_trading_fee_bps * 1e-4)
        margin_cash -= perp_entry_fee

        # Tracking arrays
        equity_series = np.zeros(n_steps)
        funding_cashflows = np.zeros(n_steps)
        margin_ratios = np.zeros(n_steps)
        margin_calls = 0
        peak_utilization = 0.0

        cum_funding = 0.0
        current_spot_units = spot_units
        current_perp_units = spot_units
        entry_perp_price = perp_prices[0]

        staking_yield_8h = (self.staking_yield_apy / (3 * 365))

        for t in range(n_steps):
            s_price = spot_prices[t]
            p_price = perp_prices[t]
            f_rate = funding_rates[t]

            # 1. Accrue Spot Staking Yield
            current_spot_units *= (1.0 + staking_yield_8h)
            spot_value = current_spot_units * s_price

            # 2. Perpetual Short Unrealized PnL
            perp_notional = current_perp_units * p_price
            perp_unrealized_pnl = current_perp_units * (entry_perp_price - p_price)

            # 3. 8-Hour Funding Cashflow: Long receives -f_rate, Short receives +f_rate
            # When funding is positive (longs pay shorts), short receives positive payment
            funding_payment = perp_notional * f_rate
            cum_funding += funding_payment
            margin_cash += funding_payment
            funding_cashflows[t] = funding_payment

            # 4. Total Portfolio Equity
            total_equity = spot_value + margin_cash + perp_unrealized_pnl
            equity_series[t] = total_equity

            # 5. Margin Ratio on Perpetual Account
            perp_margin_equity = margin_cash + perp_unrealized_pnl
            margin_ratio = perp_margin_equity / perp_notional if perp_notional > 0 else 1.0
            margin_ratios[t] = margin_ratio

            # Margin utilization
            required_maintenance = perp_notional * self.maintenance_margin_rate
            utilization = (required_maintenance / perp_margin_equity * 100.0) if perp_margin_equity > 0 else 100.0
            peak_utilization = max(peak_utilization, utilization)

            # Margin Call Check
            if margin_ratio <= self.maintenance_margin_rate:
                margin_calls += 1
                # Emergency rebalancing: transfer cash from spot liquidation to margin
                emergency_rebalance_usd = perp_notional * self.initial_margin_rate
                spot_liquidate_units = emergency_rebalance_usd / s_price
                current_spot_units = max(0.0, current_spot_units - spot_liquidate_units)
                current_perp_units = current_spot_units  # re-hedge delta
                margin_cash += emergency_rebalance_usd

        # Performance Analytics
        equity_pd = pd.Series(equity_series, index=timestamps)
        final_equity = equity_series[-1]
        total_ret = (final_equity - self.initial_capital_usd) / self.initial_capital_usd * 100.0

        n_years = n_steps / (3 * 365)
        cagr = ((final_equity / self.initial_capital_usd) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else total_ret

        # Daily resampled returns for Sharpe/Sortino
        daily_equity = equity_pd.resample("D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        ann_vol = float(daily_returns.std() * np.sqrt(365) * 100.0)
        excess_returns = daily_returns - (self.risk_free_rate / 365)
        sharpe = float(np.mean(excess_returns) / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0.0

        downside = daily_returns[daily_returns < 0]
        sortino = float(np.mean(excess_returns) / downside.std() * np.sqrt(365)) if len(downside) > 0 and downside.std() > 0 else 0.0

        # Drawdowns
        running_max = equity_pd.cummax()
        drawdown_series = (equity_pd - running_max) / running_max * 100.0
        max_dd = float(drawdown_series.min())

        calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
        net_funding_yield = (cum_funding / self.initial_capital_usd / n_years) * 100.0 if n_years > 0 else 0.0

        return BasisTradeResult(
            initial_capital_usd=self.initial_capital_usd,
            final_equity_usd=final_equity,
            total_return_pct=total_ret,
            cagr_pct=cagr,
            annualized_volatility_pct=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            total_funding_collected_usd=cum_funding,
            net_funding_yield_apy=net_funding_yield,
            margin_calls_count=margin_calls,
            peak_margin_utilization_pct=peak_utilization,
            equity_curve=equity_pd,
            funding_cashflows=pd.Series(funding_cashflows, index=timestamps),
            margin_ratios=pd.Series(margin_ratios, index=timestamps),
            drawdowns=drawdown_series,
        )
