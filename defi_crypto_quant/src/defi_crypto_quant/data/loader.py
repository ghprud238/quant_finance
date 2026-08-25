"""Data loader and synthetic market simulator for DeFi & Crypto Quantitative Finance.

Generates realistic:
1. High-frequency tick and hourly OHLCV spot data (ETH/USDC, BTC/USDT, SOL/USDC) with volatility regimes and jumps.
2. Uniswap v3 concentrated liquidity tick distributions across price ranges.
3. 8-Hour Perpetual Futures funding rates, premium indices, and basis spreads.
4. On-chain valuation metrics (MVRV, Realized Cap, Exchange Flows, Whale Accumulation).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def generate_crypto_market_data(
    tickers: Optional[List[str]] = None,
    n_days: int = 365,
    freq: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic hourly/daily crypto price series with volatility clustering and jump diffusion."""
    if tickers is None:
        tickers = ["ETH/USDC", "BTC/USDT", "SOL/USDC"]
        
    rng = np.random.default_rng(seed)
    
    base_params = {
        "ETH/USDC": {"s0": 3000.0, "mu": 0.35, "vol": 0.65, "jump_intensity": 0.08, "jump_mean": -0.02, "jump_vol": 0.06},
        "BTC/USDT": {"s0": 65000.0, "mu": 0.25, "vol": 0.50, "jump_intensity": 0.05, "jump_mean": -0.015, "jump_vol": 0.04},
        "SOL/USDC": {"s0": 150.0, "mu": 0.45, "vol": 0.85, "jump_intensity": 0.12, "jump_mean": -0.03, "jump_vol": 0.08},
    }
    
    date_range = pd.date_range("2024-01-01", periods=n_days * 24 if freq == "1h" else n_days, freq=freq)
    n_steps = len(date_range)
    dt = 1.0 / (365 * 24) if freq == "1h" else 1.0 / 365
    
    records = {}
    
    for ticker in tickers:
        p = base_params.get(ticker, {"s0": 100.0, "mu": 0.20, "vol": 0.60, "jump_intensity": 0.05, "jump_mean": -0.02, "jump_vol": 0.05})
        s0 = p["s0"]
        mu = p["mu"]
        vol_base = p["vol"]
        
        regimes = np.zeros(n_steps, dtype=int)
        vol_series = np.zeros(n_steps)
        curr_regime = 0
        
        prices = np.zeros(n_steps)
        prices[0] = s0
        
        for t in range(1, n_steps):
            if curr_regime == 0 and rng.random() < 0.01:
                curr_regime = 1
            elif curr_regime == 1 and rng.random() < 0.05:
                curr_regime = 0
            regimes[t] = curr_regime
            
            vol = vol_base * (1.8 if curr_regime == 1 else 1.0)
            vol_series[t] = vol
            
            z = rng.normal(0, 1)
            jump_count = rng.poisson(p["jump_intensity"] * dt * 365)
            jump_factor = np.exp(rng.normal(p["jump_mean"], p["jump_vol"])) if jump_count > 0 else 1.0
            
            ret = (mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
            prices[t] = max(1.0, prices[t-1] * np.exp(ret) * jump_factor)
            
        high = prices * (1.0 + np.abs(rng.normal(0, 0.003, n_steps)))
        low = prices * (1.0 - np.abs(rng.normal(0, 0.003, n_steps)))
        open_p = np.roll(prices, 1)
        open_p[0] = s0
        volume = rng.lognormal(mean=np.log(1000 * s0), sigma=0.6, size=n_steps)
        
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            val = {"Open": open_p, "High": high, "Low": low, "Close": prices, "Volume": volume}[field]
            records[(ticker, field)] = val
            
    df = pd.DataFrame(records, index=date_range)
    df.columns.names = ["Ticker", "Field"]
    return df


def generate_uniswap_v3_liquidity_ticks(
    spot_price: float = 3000.0,
    fee_tier_bps: int = 30,
    price_range_pct: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic Uniswap v3 active tick distribution around spot price."""
    rng = np.random.default_rng(seed)
    tick_spacing = {5: 10, 30: 60, 100: 200}.get(fee_tier_bps, 60)
    
    current_tick = int(np.floor(np.log(spot_price) / np.log(1.0001)))
    current_tick = (current_tick // tick_spacing) * tick_spacing
    
    min_price = spot_price * (1.0 - price_range_pct)
    max_price = spot_price * (1.0 + price_range_pct)
    
    min_tick = int(np.floor(np.log(min_price) / np.log(1.0001)))
    min_tick = (min_tick // tick_spacing) * tick_spacing
    
    max_tick = int(np.ceil(np.log(max_price) / np.log(1.0001)))
    max_tick = (max_tick // tick_spacing) * tick_spacing
    
    ticks = np.arange(min_tick, max_tick + tick_spacing, tick_spacing)
    prices = 1.0001 ** ticks
    
    dist_from_spot = np.abs(ticks - current_tick)
    sigma_ticks = 4000.0
    
    base_liquidity = 1e8 * np.exp(-0.5 * (dist_from_spot / sigma_ticks)**2)
    noise = rng.lognormal(mean=0, sigma=0.4, size=len(ticks))
    liquidity_gross = np.maximum(1e5, base_liquidity * noise)
    
    liquidity_net = np.zeros(len(ticks))
    for i in range(len(ticks) // 2):
        pos_size = liquidity_gross[i] * 0.4
        lower_idx = i
        upper_idx = min(len(ticks) - 1, i + int(rng.integers(5, 20)))
        liquidity_net[lower_idx] += pos_size
        liquidity_net[upper_idx] -= pos_size
        
    df = pd.DataFrame({
        "Tick": ticks,
        "Price": prices,
        "Sqrt_Price_X96": (np.sqrt(prices) * (2**96)).astype(object),
        "Liquidity_Gross": np.round(liquidity_gross, 2),
        "Liquidity_Net": np.round(liquidity_net, 2),
        "Is_Current_Tick": ticks == current_tick,
    })
    return df


def generate_perpetual_funding_data(
    ticker: str = "ETH-PERP",
    n_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic 8-hour perpetual funding rate timestamps, index prices, mark prices, and basis."""
    rng = np.random.default_rng(seed)
    
    n_intervals = n_days * 3
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_intervals, freq="8h")
    
    spot_ret = rng.normal(0.0003, 0.015, n_intervals)
    index_prices = 3000.0 * np.cumprod(1.0 + spot_ret)
    
    market_sentiment = np.sin(np.linspace(0, 4 * np.pi, n_intervals)) * 0.002
    premium_noise = rng.normal(0, 0.0008, n_intervals)
    premium_index = market_sentiment + premium_noise
    
    mark_prices = index_prices * (1.0 + premium_index)
    
    interest_rate_8h = 0.0001
    funding_rate_8h = np.clip(premium_index + np.clip(interest_rate_8h - premium_index, -0.0005, 0.0005), -0.0075, 0.0075)
    annualized_funding = funding_rate_8h * 3 * 365
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Index_Price": np.round(index_prices, 2),
        "Mark_Price": np.round(mark_prices, 2),
        "Basis_USD": np.round(mark_prices - index_prices, 2),
        "Basis_Bps": np.round((mark_prices / index_prices - 1.0) * 10000, 2),
        "Funding_Rate_8h": funding_rate_8h,
        "Annualized_Funding_Rate": annualized_funding,
    })
    return df


def generate_onchain_metrics_data(
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic daily on-chain valuation metrics: MVRV, Realized Cap, Exchange Reserves, Active Addresses."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    ret = rng.normal(0.001, 0.03, n_days)
    prices = 45000.0 * np.cumprod(1.0 + ret)
    
    realized_price = pd.Series(prices).rolling(90, min_periods=1).mean().values * (1.0 + rng.normal(0, 0.02, n_days))
    mvrv_ratio = prices / realized_price
    
    base_exchange_reserves = 2.2e6
    exchange_flow = rng.normal(-150, 1200, n_days)
    exchange_reserves = base_exchange_reserves + np.cumsum(exchange_flow)
    
    whale_holdings = 8.5e6 + np.cumsum(rng.normal(200, 800, n_days))
    active_addresses = np.maximum(400000, 900000 + (prices / 100) * 3 + rng.normal(0, 40000, n_days))
    
    df = pd.DataFrame({
        "Date": dates,
        "Market_Price": np.round(prices, 2),
        "Realized_Price": np.round(realized_price, 2),
        "MVRV_Ratio": np.round(mvrv_ratio, 3),
        "Exchange_Reserves": np.round(exchange_reserves, 0),
        "Net_Exchange_Inflow": np.round(exchange_flow, 0),
        "Whale_Holdings": np.round(whale_holdings, 0),
        "Active_Addresses": np.round(active_addresses, 0),
    }, index=dates)
    return df


# =========================================================================
# SUBAGENT 2 HELPERS (Module 43 & 44)
# =========================================================================

"""Synthetic data generation for DEX pools, funding rates, and mempool transactions."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def generate_synthetic_dex_pools() -> Dict[str, Dict[str, Any]]:
    """Generates cross-DEX liquidity pool configurations for spatial and triangular arbitrage."""
    return {
        "uniswap_v2_eth_usdc": {
            "venue": "Uniswap v2",
            "token_a": "WETH",
            "token_b": "USDC",
            "reserve_a": 2500.0,      # WETH
            "reserve_b": 7_500_000.0, # USDC (000 / WETH)
            "fee": 0.003,             # 0.30%
        },
        "sushiswap_eth_usdc": {
            "venue": "Sushiswap",
            "token_a": "WETH",
            "token_b": "USDC",
            "reserve_a": 1800.0,      # WETH
            "reserve_b": 5_580_000.0, # USDC (100 / WETH - mispriced higher)
            "fee": 0.003,             # 0.30%
        },
        "curve_eth_steth": {
            "venue": "Curve",
            "token_a": "WETH",
            "token_b": "stETH",
            "reserve_a": 10000.0,
            "reserve_b": 9950.0,
            "fee": 0.0004,            # 0.04%
        },
        "uniswap_v2_btc_usdc": {
            "venue": "Uniswap v2",
            "token_a": "WBTC",
            "token_b": "USDC",
            "reserve_a": 120.0,
            "reserve_b": 7_800_000.0, # 5,000 / WBTC
            "fee": 0.003,
        },
        "uniswap_v2_wbtc_eth": {
            "venue": "Uniswap v2",
            "token_a": "WBTC",
            "token_b": "WETH",
            "reserve_a": 80.0,
            "reserve_b": 1720.0,      # 21.5 WETH / WBTC
            "fee": 0.003,
        },
    }


def generate_synthetic_funding_rates(
    n_periods: int = 1095,  # 1095 8-hour intervals = 1 year
    base_rate_annual: float = 0.12,
    volatility: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic 8-hour funding rates and spot/perpetual price series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01 00:00:00", periods=n_periods, freq="8h")
    mean_8h_rate = base_rate_annual / (3 * 365)
    
    regimes = np.zeros(n_periods)
    current_regime = 0
    for t in range(n_periods):
        if rng.random() < 0.02:
            current_regime = rng.choice([-1, 0, 1], p=[0.25, 0.50, 0.25])
        regimes[t] = current_regime
        
    rates = np.zeros(n_periods)
    spot_prices = np.zeros(n_periods)
    spot_price = 3000.0
    
    for t in range(n_periods):
        reg = regimes[t]
        if reg == 1:
            target_mean = 0.0006  # +6 bps per 8h (~65% APY)
        elif reg == -1:
            target_mean = -0.00015  # -1.5 bps per 8h (-16% APY)
        else:
            target_mean = mean_8h_rate  # +1.1 bps per 8h (12% APY)
            
        rate = rng.normal(target_mean, 0.00015)
        rates[t] = np.clip(rate, -0.0075, 0.0075)
        
        price_ret = rng.normal(0.0001, 0.012)
        spot_price *= (1.0 + price_ret)
        spot_prices[t] = spot_price
        
    perp_prices = spot_prices * (1.0 + rates * 3.0)
    
    return pd.DataFrame({
        "Timestamp": dates,
        "Spot_Price": spot_prices,
        "Perp_Price": perp_prices,
        "Funding_Rate_8h": rates,
        "Annualized_Yield": rates * 3 * 365,
    })


def generate_synthetic_mempool_swaps() -> List[Dict[str, Any]]:
    """Generates pending victim transactions in the mempool for sandwich simulation."""
    return [
        {
            "tx_hash": "0xabc123victim01",
            "pool": "uniswap_v2_eth_usdc",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in": 250_000.0,
            "max_slippage_pct": 0.01,
            "gas_price_gwei": 30.0,
        },
        {
            "tx_hash": "0xabc123victim02",
            "pool": "uniswap_v2_eth_usdc",
            "token_in": "WETH",
            "token_out": "USDC",
            "amount_in": 120.0,
            "max_slippage_pct": 0.005,
            "gas_price_gwei": 25.0,
        },
    ]
