"""On-Chain Blockchain Telemetry, MVRV, Exchange Flows & Whale Alpha Engine (Project 45).

Implements institutional on-chain metrics and systematic trading strategies:
1. MVRV Ratio & MVRV Z-Score (Market Cap vs Realized Cap).
2. Net Exchange Flow Imbalance (EFI) tracking liquidity exchange reserves.
3. Whale Wallet Accumulation Index (large tier address balance deltas).
4. Network Value to Transactions (NVT Ratio & NVT Signal).
5. Active Address Network Velocity.
6. Multi-Factor On-Chain Macro Regime Classifier.
7. Systematic On-Chain Quantitative Strategy with Dynamic Position Sizing.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class OnChainRegime(str, Enum):
    """Macro crypto market regime derived from on-chain telemetry."""
    ACCUMULATION_BOTTOM = "ACCUMULATION_BOTTOM"
    BULL_EXPANSION = "BULL_EXPANSION"
    OVERHEATED_EUPHORIA = "OVERHEATED_EUPHORIA"
    CAPITULATION_BEAR = "CAPITULATION_BEAR"


@dataclass
class OnChainMetrics:
    """Snapshot of raw and standardized on-chain telemetry metrics."""
    timestamp: pd.Timestamp
    price: float
    market_cap: float
    realized_cap: float
    mvrv: float
    mvrv_z_score: float
    exchange_inflows: float
    exchange_outflows: float
    exchange_net_flow: float
    exchange_flow_imbalance: float
    whale_balance: float
    whale_accumulation_index: float
    nvt_ratio: float
    nvt_signal: float
    active_addresses: float
    address_velocity: float
    regime: OnChainRegime
    composite_alpha_score: float


@dataclass
class OnChainBacktestResult:
    """Comprehensive performance report and analytics for on-chain strategy."""
    dates: pd.DatetimeIndex
    prices: pd.Series
    signals: pd.Series
    positions: pd.Series
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    cumulative_strategy: pd.Series
    cumulative_benchmark: pd.Series
    regimes: pd.Series
    metrics: Dict[str, Any]
    daily_df: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        """Returns structured performance metrics table."""
        rows = []
        for k, v in self.metrics.items():
            if isinstance(v, float):
                if any(term in k for term in ["CAGR", "Return", "Volatility", "Drawdown", "Win Rate", "Drag"]):
                    formatted_v = f"{v:+.2%}" if "Return" in k or "CAGR" in k else f"{v:.2%}"
                else:
                    formatted_v = f"{v:.4f}"
            else:
                formatted_v = str(v)
            rows.append({"Metric": k, "Value": formatted_v})
        return pd.DataFrame(rows)


class OnChainAlphaEngine:
    """Quantitative on-chain blockchain telemetry & alpha modeling engine.
    
    Extracts high-signal fundamental drivers from raw ledger flows, wallet distributions,
    and valuation metrics to forecast medium-to-long term crypto price trends.
    """

    def __init__(
        self,
        mvrv_lookback_std: int = 365,
        efi_smoothing_window: int = 7,
        whale_lookback_days: int = 30,
        whale_z_window: int = 180,
        nvt_fast_window: int = 28,
        nvt_slow_window: int = 90,
        velocity_fast_window: int = 14,
        velocity_slow_window: int = 60,
    ) -> None:
        self.mvrv_lookback_std = mvrv_lookback_std
        self.efi_smoothing_window = efi_smoothing_window
        self.whale_lookback_days = whale_lookback_days
        self.whale_z_window = whale_z_window
        self.nvt_fast_window = nvt_fast_window
        self.nvt_slow_window = nvt_slow_window
        self.velocity_fast_window = velocity_fast_window
        self.velocity_slow_window = velocity_slow_window

    # =========================================================================
    # 1. CORE ON-CHAIN METRICS COMPUTATION
    # =========================================================================

    def calculate_mvrv(
        self,
        market_cap: pd.Series,
        realized_cap: pd.Series,
        rolling_std_window: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculates MVRV Ratio and MVRV Z-Score.
        
        MVRV = Market Cap / Realized Cap
        MVRV Z-Score = (Market Cap - Realized Cap) / std(Market Cap)
        """
        window = rolling_std_window or self.mvrv_lookback_std
        safe_realized = realized_cap.replace(0, np.nan)
        mvrv = market_cap / safe_realized

        # Standard deviation of market cap (expanding with fallback)
        std_market_cap = market_cap.rolling(window=window, min_periods=14).std()
        std_market_cap = std_market_cap.fillna(market_cap.expanding(min_periods=2).std())
        std_market_cap = std_market_cap.replace(0, np.nan).fillna(1.0)

        mvrv_z_score = (market_cap - realized_cap) / std_market_cap
        return mvrv.fillna(1.0), mvrv_z_score.fillna(0.0)

    def calculate_exchange_flow_imbalance(
        self,
        inflows: pd.Series,
        outflows: pd.Series,
        smoothing_window: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculates Net Exchange Flow and Net Exchange Flow Imbalance (EFI).
        
        EFI = (Inflows - Outflows) / (Inflows + Outflows) in [-1.0, +1.0]
        Negative EFI indicates exchange reserve depletion (outflows/accumulation -> bullish).
        Positive EFI indicates exchange deposits (inflows/selling pressure -> bearish).
        """
        window = smoothing_window or self.efi_smoothing_window
        net_flow = inflows - outflows
        total_flow = inflows + outflows

        safe_total = total_flow.replace(0, np.nan)
        raw_efi = (net_flow / safe_total).fillna(0.0).clip(-1.0, 1.0)
        smoothed_efi = raw_efi.rolling(window=window, min_periods=1).mean().clip(-1.0, 1.0)

        return net_flow, smoothed_efi

    def calculate_whale_accumulation_index(
        self,
        whale_balances: pd.Series,
        lookback_days: Optional[int] = None,
        z_window: Optional[int] = None,
    ) -> pd.Series:
        """Calculates Whale Wallet Accumulation Index.
        
        Tracks percentage changes in supply held by addresses >= 1,000 BTC / >= 10,000 ETH
        standardized against historical accumulation velocity.
        """
        k = lookback_days or self.whale_lookback_days
        z_win = z_window or self.whale_z_window

        # Percentage change over lookback
        whale_pct_change = whale_balances.pct_change(k).fillna(0.0)

        # Standardize via rolling mean and std
        rolling_mean = whale_pct_change.rolling(z_win, min_periods=7).mean().fillna(0.0)
        rolling_std = whale_pct_change.rolling(z_win, min_periods=7).std().replace(0, np.nan).fillna(1.0)

        whale_index = ((whale_pct_change - rolling_mean) / rolling_std).fillna(0.0)
        return whale_index.clip(-4.0, 4.0)

    def calculate_nvt_metrics(
        self,
        market_cap: pd.Series,
        tx_volume: pd.Series,
        fast_window: Optional[int] = None,
        slow_window: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculates Network Value to Transactions (NVT) Ratio & NVT Signal.
        
        NVT Ratio = Market Cap / Tx Volume
        NVT Signal = Market Cap / MA_slow(Tx Volume)
        """
        fast_w = fast_window or self.nvt_fast_window
        slow_w = slow_window or self.nvt_slow_window

        safe_vol = tx_volume.replace(0, np.nan)
        nvt_ratio = (market_cap / safe_vol).ffill().fillna(50.0)

        smoothed_vol = tx_volume.rolling(slow_w, min_periods=7).mean().replace(0, np.nan)
        nvt_signal = (market_cap / smoothed_vol).ffill().fillna(50.0)

        return nvt_ratio, nvt_signal

    def calculate_address_velocity(
        self,
        active_addresses: pd.Series,
        lookback_fast: Optional[int] = None,
        lookback_slow: Optional[int] = None,
    ) -> pd.Series:
        """Calculates Active Address Velocity (Network Momentum).
        
        Velocity = (MA_fast(Active) - MA_slow(Active)) / MA_slow(Active)
        """
        fast_w = lookback_fast or self.velocity_fast_window
        slow_w = lookback_slow or self.velocity_slow_window

        fast_ma = active_addresses.rolling(fast_w, min_periods=1).mean()
        slow_ma = active_addresses.rolling(slow_w, min_periods=1).mean()

        safe_slow = slow_ma.replace(0, np.nan)
        velocity = ((fast_ma - slow_ma) / safe_slow).fillna(0.0)
        return velocity.clip(-1.0, 1.0)

    # =========================================================================
    # 2. MULTI-FACTOR REGIME CLASSIFIER & COMPOSITE ALPHA
    # =========================================================================

    def classify_regimes(
        self,
        mvrv: pd.Series,
        mvrv_z: pd.Series,
        efi: pd.Series,
        whale_acc: pd.Series,
        addr_velocity: pd.Series,
    ) -> pd.Series:
        """Classifies market into 4 distinct macro on-chain regimes."""
        regimes = pd.Series(index=mvrv.index, dtype=object)

        # Vectorized rule evaluation
        for idx in mvrv.index:
            m = mvrv.loc[idx]
            mz = mvrv_z.loc[idx]
            e = efi.loc[idx]
            w = whale_acc.loc[idx]
            v = addr_velocity.loc[idx]

            # 1. Euphoria / Blow-off top
            if m >= 3.0 or mz >= 3.5 or (m >= 2.5 and e >= 0.20):
                regimes.loc[idx] = OnChainRegime.OVERHEATED_EUPHORIA
            # 2. Deep Accumulation / Value Bottom
            elif m <= 1.15 or mz <= 0.20 or (m <= 1.30 and (w >= 0.5 or e <= -0.10)):
                regimes.loc[idx] = OnChainRegime.ACCUMULATION_BOTTOM
            # 3. Capitulation / Bear Bleed
            elif (m <= 1.60 and e >= 0.05 and v <= -0.05) or (mz < 0.8 and e > 0.15):
                regimes.loc[idx] = OnChainRegime.CAPITULATION_BEAR
            # 4. Standard Bull Expansion
            else:
                regimes.loc[idx] = OnChainRegime.BULL_EXPANSION

        return regimes

    def compute_composite_signal(
        self,
        mvrv: pd.Series,
        mvrv_z: pd.Series,
        efi: pd.Series,
        whale_acc: pd.Series,
        nvt_signal: pd.Series,
        addr_velocity: pd.Series,
    ) -> pd.Series:
        """Computes continuous composite on-chain alpha score in [-1.0, +1.0]."""
        # 1. Valuation Component: Low MVRV is bullish, High MVRV is bearish
        s_mvrv = -np.tanh((mvrv - 1.75) / 0.85)

        # 2. Exchange Flow Component: Outflows (negative EFI) are bullish (+), Inflows are bearish (-)
        s_efi = -np.tanh(efi * 3.0)

        # 3. Whale Component: Positive whale stacking is bullish
        s_whale = np.tanh(whale_acc * 0.75)

        # 4. Active Address Velocity: Positive network expansion is bullish
        s_vel = np.tanh(addr_velocity * 4.0)

        # 5. NVT Component: Standardized NVT (lower NVT indicates higher relative on-chain velocity)
        nvt_mean = nvt_signal.rolling(365, min_periods=30).mean().fillna(nvt_signal.mean())
        nvt_std = nvt_signal.rolling(365, min_periods=30).std().replace(0, np.nan).fillna(nvt_signal.std())
        s_nvt = -np.tanh(((nvt_signal - nvt_mean) / nvt_std).fillna(0.0) * 0.5)

        # Weighted combination
        composite = (
            0.30 * s_mvrv +
            0.25 * s_efi +
            0.20 * s_whale +
            0.15 * s_vel +
            0.10 * s_nvt
        )
        return composite.clip(-1.0, 1.0)

    # =========================================================================
    # 3. SYNTHETIC MULTI-CYCLE DATA GENERATOR
    # =========================================================================

    def generate_synthetic_onchain_data(
        self,
        n_days: int = 1500,
        initial_price: float = 10000.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generates realistic multi-cycle crypto price action & on-chain metrics."""
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

        # Multi-cycle regime phases (Bear bottom -> Bull run -> Euphoria top -> Crash -> Accumulation)
        t = np.linspace(0, 4 * np.pi, n_days)
        cycle_drift = 0.0006 + 0.0012 * np.sin(t - np.pi / 4) + 0.0004 * np.sin(2 * t)
        cycle_vol = 0.025 + 0.015 * np.abs(np.sin(t))

        price_returns = np.random.normal(cycle_drift, cycle_vol)
        # Inject realistic flash volatility spikes / liquidations
        crash_indices = [int(n_days * 0.35), int(n_days * 0.70)]
        for ci in crash_indices:
            price_returns[ci : ci + 5] -= 0.08

        prices = initial_price * np.cumprod(1.0 + price_returns)
        supply = 18.5e6 + np.linspace(0, 1.5e6, n_days)  # Circulating supply
        market_cap = prices * supply

        # Realized Cap (lags market cap, smoother, represents aggregate cost basis)
        realized_price = pd.Series(prices).ewm(span=180, min_periods=1).mean().values
        # Compress realized price drift during parabolic tops
        realized_cap = realized_price * supply * 0.92

        # Exchange Inflows & Outflows
        base_flow = 15000 + 5000 * np.sin(t)
        # Inflows spike near cycle tops / panic selloffs
        price_accel = pd.Series(price_returns).rolling(14, min_periods=1).mean().values
        inflow_shock = np.maximum(0.0, -price_accel * 40000) + np.maximum(0.0, (prices / realized_price - 2.5) * 15000)
        outflow_shock = np.maximum(0.0, price_accel * 35000) + np.maximum(0.0, (1.2 - prices / realized_price) * 20000)

        inflows = np.maximum(500.0, base_flow + inflow_shock + np.random.exponential(4000, n_days))
        outflows = np.maximum(500.0, base_flow + outflow_shock + np.random.exponential(4000, n_days))

        # Whale Balances (addresses >= 1,000 BTC)
        whale_base = 7.8e6
        whale_accumulation_trend = np.cumsum(np.where(prices < realized_price * 1.2, 350, -250))
        whale_balances = whale_base + whale_accumulation_trend + np.random.normal(0, 15000, n_days)

        # On-Chain Transaction Volume & Active Addresses
        tx_vol_usd = market_cap * (0.015 + 0.010 * np.abs(price_returns) + 0.005 * np.maximum(0, np.sin(t)))
        active_addresses = 700000 + 400000 * (prices / initial_price)**0.6 + np.random.normal(0, 30000, n_days)

        df = pd.DataFrame({
            "Price": np.round(prices, 2),
            "Market_Cap": market_cap,
            "Realized_Cap": realized_cap,
            "Exchange_Inflows": np.round(inflows, 2),
            "Exchange_Outflows": np.round(outflows, 2),
            "Whale_Balance": np.round(whale_balances, 2),
            "Tx_Volume_USD": np.round(tx_vol_usd, 2),
            "Active_Addresses": np.round(np.maximum(100000, active_addresses), 0),
        }, index=dates)

        return df

    # =========================================================================
    # 4. SYSTEMATIC STRATEGY BACKTESTING & EVALUATION
    # =========================================================================

    def backtest_strategy(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,
        max_leverage: float = 1.5,
        allow_short: bool = True,
    ) -> OnChainBacktestResult:
        """Executes full quantitative on-chain systematic strategy backtest."""
        df = data.copy()

        # Compute all on-chain indicators
        mvrv, mvrv_z = self.calculate_mvrv(df["Market_Cap"], df["Realized_Cap"])
        net_flow, efi = self.calculate_exchange_flow_imbalance(df["Exchange_Inflows"], df["Exchange_Outflows"])
        whale_acc = self.calculate_whale_accumulation_index(df["Whale_Balance"])
        nvt_ratio, nvt_signal = self.calculate_nvt_metrics(df["Market_Cap"], df["Tx_Volume_USD"])
        addr_velocity = self.calculate_address_velocity(df["Active_Addresses"])

        regimes = self.classify_regimes(mvrv, mvrv_z, efi, whale_acc, addr_velocity)
        composite_signal = self.compute_composite_signal(mvrv, mvrv_z, efi, whale_acc, nvt_signal, addr_velocity)

        # Position Sizing Logic
        target_positions = pd.Series(0.0, index=df.index)

        for i, idx in enumerate(df.index):
            regime = regimes.loc[idx]
            sig = composite_signal.loc[idx]

            if regime == OnChainRegime.ACCUMULATION_BOTTOM:
                # Strong long conviction with leverage
                pos = np.clip(1.0 + 0.5 * max(0.0, sig), 0.5, max_leverage)
            elif regime == OnChainRegime.BULL_EXPANSION:
                # Scaled long exposure according to alpha score
                pos = np.clip(0.8 + 0.4 * sig, 0.2, 1.2)
            elif regime == OnChainRegime.OVERHEATED_EUPHORIA:
                # De-risk to cash or short
                pos = -0.5 if (allow_short and sig < -0.3) else 0.0
            elif regime == OnChainRegime.CAPITULATION_BEAR:
                # Protective posture or tactical short
                pos = -0.3 if (allow_short and sig < -0.4) else 0.0
            else:
                pos = np.clip(sig, -1.0 if allow_short else 0.0, 1.0)

            target_positions.loc[idx] = pos

        # Strictly lagged positions to eliminate lookahead bias (trade executed next bar)
        exec_positions = target_positions.shift(1).fillna(0.0)

        # Asset returns
        asset_returns = df["Price"].pct_change().fillna(0.0)

        # Turnover and transaction cost deduction
        turnover = exec_positions.diff().abs().fillna(0.0)
        cost_drag = turnover * (transaction_cost_bps / 10000.0)

        strategy_returns = exec_positions * asset_returns - cost_drag
        benchmark_returns = asset_returns

        cum_strategy = (1.0 + strategy_returns).cumprod()
        cum_benchmark = (1.0 + benchmark_returns).cumprod()

        # Information Coefficient (Signal vs Forward 5-Day Returns)
        fwd_5d_ret = df["Price"].pct_change(5).shift(-5)
        valid_mask = composite_signal.notna() & fwd_5d_ret.notna()
        if valid_mask.sum() > 30:
            ic, ic_p = pearsonr(composite_signal[valid_mask], fwd_5d_ret[valid_mask])
            rank_ic, rank_ic_p = spearmanr(composite_signal[valid_mask], fwd_5d_ret[valid_mask])
        else:
            ic, rank_ic = 0.0, 0.0

        # Performance Metrics
        n_years = len(df) / 365.25
        cagr = (cum_strategy.iloc[-1]) ** (1.0 / max(0.1, n_years)) - 1.0
        bench_cagr = (cum_benchmark.iloc[-1]) ** (1.0 / max(0.1, n_years)) - 1.0

        ann_vol = strategy_returns.std() * np.sqrt(365)
        bench_vol = benchmark_returns.std() * np.sqrt(365)

        rf = 0.03
        sharpe = (cagr - rf) / max(1e-4, ann_vol)
        bench_sharpe = (bench_cagr - rf) / max(1e-4, bench_vol)

        downside_returns = strategy_returns[strategy_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 5 else ann_vol
        sortino = (cagr - rf) / max(1e-4, downside_vol)

        # Drawdowns
        running_max = cum_strategy.cummax()
        drawdowns = (cum_strategy - running_max) / running_max
        max_dd = drawdowns.min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

        win_rate = (strategy_returns > 0).sum() / max(1, (strategy_returns != 0).sum())
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / max(1e-6, gross_loss)

        metrics = {
            "Strategy CAGR": cagr,
            "Benchmark CAGR (Buy & Hold)": bench_cagr,
            "Strategy Annualized Volatility": ann_vol,
            "Benchmark Volatility": bench_vol,
            "Sharpe Ratio (Rf=3%)": sharpe,
            "Benchmark Sharpe Ratio": bench_sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Maximum Drawdown": max_dd,
            "Daily Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Information Coefficient (IC)": ic,
            "Rank IC (Spearman)": rank_ic,
            "Annualized Portfolio Turnover": turnover.mean() * 365,
            "Annual Transaction Cost Drag": cost_drag.mean() * 365,
        }

        daily_df = pd.DataFrame({
            "Price": df["Price"],
            "Market_Cap": df["Market_Cap"],
            "Realized_Cap": df["Realized_Cap"],
            "MVRV": mvrv,
            "MVRV_Z_Score": mvrv_z,
            "Exchange_Net_Flow": net_flow,
            "Exchange_Flow_Imbalance": efi,
            "Whale_Accumulation_Index": whale_acc,
            "NVT_Signal": nvt_signal,
            "Address_Velocity": addr_velocity,
            "OnChain_Regime": regimes,
            "Composite_Alpha": composite_signal,
            "Target_Position": target_positions,
            "Executed_Position": exec_positions,
            "Strategy_Daily_Return": strategy_returns,
            "Benchmark_Daily_Return": benchmark_returns,
            "Cumulative_Strategy": cum_strategy,
            "Cumulative_Benchmark": cum_benchmark,
            "Drawdown": drawdowns,
        }, index=df.index)

        return OnChainBacktestResult(
            dates=df.index,
            prices=df["Price"],
            signals=composite_signal,
            positions=exec_positions,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            cumulative_strategy=cum_strategy,
            cumulative_benchmark=cum_benchmark,
            regimes=regimes,
            metrics=metrics,
            daily_df=daily_df,
        )
