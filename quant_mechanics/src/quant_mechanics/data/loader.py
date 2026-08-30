"""Data generation and loading utilities for quantitative mechanics."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class OrderBookLevel:
    """Represents a single price-size level in an order book."""
    price: float
    size: float


@dataclass
class BinaryContractOrderBook:
    """Level 2 order book for a binary prediction market contract (Yes/No)."""
    contract_id: str
    event_title: str
    venue: str
    timestamp_ns: int
    yes_bids: List[OrderBookLevel]
    yes_asks: List[OrderBookLevel]
    no_bids: List[OrderBookLevel]
    no_asks: List[OrderBookLevel]
    fee_rate: float
    gas_cost_usd: float


def generate_prediction_market_order_books(
    seed: int = 42
) -> Dict[str, Dict[str, BinaryContractOrderBook]]:
    """Generates synthetic Level 2 order books for Polymarket and Kalshi.
    
    Creates binary contracts across multiple themes (US Election, Fed Rate Decision,
    NBA Championship) with realistic mispricing windows, order book depth ladders,
    and venue fee/gas structures.
    
    Returns:
        Dict[contract_name, Dict[venue_name, BinaryContractOrderBook]]
    """
    np.random.seed(seed)
    
    contracts = [
        ("FED_RATE_CUT_SEP", "Fed Cuts Rates by >= 25bps in September 2026"),
        ("PRES_ELECTION_DEM", "Democratic Party Wins 2028 US Presidential Election"),
        ("NBA_FINALS_BOS", "Boston Celtics Win NBA Championship"),
        ("GLOBAL_AI_TREATY", "UN Ratifies Global AI Safety Treaty by 2027"),
    ]
    
    result = {}
    base_timestamps = 1_770_000_000_000_000_000  # nanoseconds
    
    for i, (cid, title) in enumerate(contracts):
        # True latent probability
        p_true = 0.35 + 0.12 * i
        
        # Venue A: Polymarket (Polygon gas ~ $0.015, 0% trading fee, crypto taker)
        # Venue B: Kalshi (CFTC regulated, maker/taker fee ~ 1.0% / 0.01 fee, zero gas)
        
        # Create subtle mispricing between venues
        noise_poly = np.random.normal(0.0, 0.02)
        noise_kalshi = np.random.normal(0.0, 0.02)
        
        # Polymarket mid
        p_mid_poly = np.clip(p_true + noise_poly, 0.08, 0.92)
        # Kalshi mid
        p_mid_kalshi = np.clip(p_true + noise_kalshi, 0.08, 0.92)
        
        # For the first contract, engineer an intentional cross-venue arbitrage
        if i == 0:
            p_mid_poly = 0.46
            p_mid_kalshi = 0.58
        
        # Build ladders
        def build_ladder(mid_p: float, spread: float = 0.02, n_levels: int = 5):
            half_s = spread / 2.0
            best_bid = round(max(0.01, mid_p - half_s), 2)
            best_ask = round(min(0.99, mid_p + half_s), 2)
            if best_bid >= best_ask:
                best_ask = round(best_bid + 0.01, 2)
                
            bids = []
            asks = []
            for lvl in range(n_levels):
                b_price = round(max(0.01, best_bid - lvl * 0.01), 2)
                b_size = float(np.random.randint(500, 5000) * (lvl + 1))
                bids.append(OrderBookLevel(price=b_price, size=b_size))
                
                a_price = round(min(0.99, best_ask + lvl * 0.01), 2)
                a_size = float(np.random.randint(500, 5000) * (lvl + 1))
                asks.append(OrderBookLevel(price=a_price, size=a_size))
            return bids, asks
        
        poly_yes_bids, poly_yes_asks = build_ladder(p_mid_poly, spread=0.02)
        # In binary markets, No Ask = 1 - Yes Bid, No Bid = 1 - Yes Ask
        poly_no_bids = [OrderBookLevel(price=round(1.0 - a.price, 2), size=a.size) for a in poly_yes_asks]
        poly_no_asks = [OrderBookLevel(price=round(1.0 - b.price, 2), size=b.size) for b in poly_yes_bids]
        
        kalshi_yes_bids, kalshi_yes_asks = build_ladder(p_mid_kalshi, spread=0.03)
        kalshi_no_bids = [OrderBookLevel(price=round(1.0 - a.price, 2), size=a.size) for a in kalshi_yes_asks]
        kalshi_no_asks = [OrderBookLevel(price=round(1.0 - b.price, 2), size=b.size) for b in kalshi_yes_bids]
        
        book_poly = BinaryContractOrderBook(
            contract_id=cid,
            event_title=title,
            venue="Polymarket",
            timestamp_ns=base_timestamps + int(np.random.randint(100, 500) * 1e6),
            yes_bids=poly_yes_bids,
            yes_asks=poly_yes_asks,
            no_bids=poly_no_bids,
            no_asks=poly_no_asks,
            fee_rate=0.000,       # 0% on Polymarket CTF exchange
            gas_cost_usd=0.015,   # Polygon PoS L2 gas fee
        )
        
        book_kalshi = BinaryContractOrderBook(
            contract_id=cid,
            event_title=title,
            venue="Kalshi",
            timestamp_ns=base_timestamps + int(np.random.randint(100, 500) * 1e6),
            yes_bids=kalshi_yes_bids,
            yes_asks=kalshi_yes_asks,
            no_bids=kalshi_no_bids,
            no_asks=kalshi_no_asks,
            fee_rate=0.010,       # 1.0% transaction fee
            gas_cost_usd=0.000,   # Regulated exchange, no crypto gas
        )
        
        result[cid] = {
            "Polymarket": book_poly,
            "Kalshi": book_kalshi,
        }
        
    return result


def generate_macro_news_and_trades(
    n_days: int = 5,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generates high-frequency timestamped trade prints and scheduled macro announcements.
    
    Creates a 2-variate mutually exciting event sequence:
      - Node 0: Macro News Announcements (Scheduled CPI, NFP, Fed releases + breaking headlines)
      - Node 1: High-Frequency Volatility / Price Jump Events
      
    Returns:
        Dict with keys:
          'macro_news_times': 1D array of timestamps in seconds
          'price_jump_times': 1D array of timestamps in seconds
          'horizon': total duration in seconds
    """
    np.random.seed(seed)
    horizon = n_days * 86400.0  # 5 days in seconds
    
    # Generate scheduled macro announcements (approx 2 per day at 8:30 AM and 2:00 PM EST)
    scheduled_news = []
    for d in range(n_days):
        t_day_start = d * 86400.0
        # 8:30 AM = 8.5 * 3600 = 30600s
        scheduled_news.append(t_day_start + 30600.0 + np.random.normal(0, 2.0))
        # 2:00 PM = 14.0 * 3600 = 50400s
        scheduled_news.append(t_day_start + 50400.0 + np.random.normal(0, 2.0))
        
    # Unscheduled breaking news (Poisson process with rate ~ 1 per 6 hours)
    n_unscheduled = np.random.poisson(horizon / (6 * 3600.0))
    unscheduled = np.sort(np.random.uniform(0, horizon, n_unscheduled))
    
    macro_news_times = np.sort(np.concatenate([scheduled_news, unscheduled]))
    
    # Generate trade jump times triggered by macro news plus endogenous trade clustering
    # Base rate of trades: 1 jump every 300 seconds
    # News excitation: each news item excites a cascade of trade jumps
    jump_times = []
    
    # Exogenous trade jumps
    n_base = int(horizon / 400.0)
    current_t = 0.0
    for _ in range(n_base):
        current_t += np.random.exponential(400.0)
        if current_t < horizon:
            jump_times.append(current_t)
            
    # News-induced cascades (decay beta = 0.005, alpha = 0.8)
    for t_news in macro_news_times:
        n_cascades = np.random.poisson(15)  # surge in volatility events
        cascade_dt = np.random.exponential(scale=120.0, size=n_cascades)
        for dt in cascade_dt:
            t_event = t_news + dt
            if t_event < horizon:
                jump_times.append(t_event)
                # Endogenous secondary cascade
                if np.random.rand() < 0.4:
                    jump_times.append(t_event + np.random.exponential(30.0))
                    
    price_jump_times = np.sort(np.array(jump_times))
    
    return {
        "macro_news_times": macro_news_times,
        "price_jump_times": price_jump_times,
        "horizon": horizon,
    }


def generate_option_chains(
    spot: float = 100.0,
    r: float = 0.045,
    q: float = 0.010,
    seed: int = 42
) -> pd.DataFrame:
    """Generates realistic live option chain tables with volatility skew and crash risk."""
    from scipy.stats import norm
    
    np.random.seed(seed)
    expiries = [0.0833, 0.25, 0.50, 1.00]  # 1M, 3M, 6M, 1Y
    strikes = [70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130]
    
    rows = []
    for T in expiries:
        for K in strikes:
            # Implied volatility skew (Gatheral SVI-like parametric curve)
            log_m = np.log(K / spot)
            atm_vol = 0.20 + 0.02 * np.sqrt(T)
            skew = -0.15 * log_m / np.sqrt(T)
            smile = 0.08 * (log_m ** 2) / T
            sigma = float(np.clip(atm_vol + skew + smile, 0.08, 0.65))
            
            # Black-Scholes analytical prices
            d1 = (np.log(spot / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_theo = spot * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_theo = K * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(-q * T) * norm.cdf(-d1)
            
            # Spread and quotes
            half_spread_c = max(0.05, 0.02 * call_theo)
            call_bid = round(max(0.01, call_theo - half_spread_c), 2)
            call_ask = round(call_theo + half_spread_c, 2)
            
            half_spread_p = max(0.05, 0.02 * put_theo)
            put_bid = round(max(0.01, put_theo - half_spread_p), 2)
            put_ask = round(put_theo + half_spread_p, 2)
            
            rows.append({
                "Expiry_Years": T,
                "Days_To_Expiry": int(T * 365),
                "Strike": K,
                "Moneyness_K_over_S": round(K / spot, 4),
                "Implied_Vol": round(sigma, 4),
                "Call_Bid": call_bid,
                "Call_Ask": call_ask,
                "Call_Mid": round(0.5 * (call_bid + call_ask), 2),
                "Put_Bid": put_bid,
                "Put_Ask": put_ask,
                "Put_Mid": round(0.5 * (put_bid + put_ask), 2),
            })
            
    return pd.DataFrame(rows)


def generate_multi_strategy_returns(
    n_strategies: int = 1000,
    n_days: int = 504,
    n_true_alpha: int = 25,
    seed: int = 42
) -> pd.DataFrame:
    """Generates 1,000 synthetic strategy return paths (noise trials vs true alpha).
    
    Includes realistic non-normal skewness and fat-tailed Student-t kurtosis.
    """
    np.random.seed(seed)
    
    # 975 pure noise strategies (true mean = 0.0)
    # 25 genuine alpha strategies (true annual Sharpe = 1.2 to 2.2)
    daily_returns = np.zeros((n_days, n_strategies))
    
    # Common market factor to introduce cross-strategy correlation
    market_innovations = np.random.standard_t(df=4.5, size=n_days) * (0.16 / np.sqrt(252))
    
    for s in range(n_strategies):
        is_true_alpha = (s < n_true_alpha)
        
        # Alpha daily drift
        if is_true_alpha:
            true_sharpe = np.random.uniform(1.2, 2.2)
            daily_drift = (true_sharpe * 0.15) / 252.0
        else:
            daily_drift = 0.0
            
        # Strategy daily volatility ~ 15% annual
        daily_vol = np.random.uniform(0.10, 0.20) / np.sqrt(252)
        
        # Idiosyncratic Student-t innovations (fat tails)
        idio = np.random.standard_t(df=4.0, size=n_days) * daily_vol
        
        # Market exposure beta
        beta = np.random.uniform(-0.3, 0.3) if is_true_alpha else np.random.uniform(-0.8, 0.8)
        
        daily_returns[:, s] = daily_drift + beta * market_innovations + idio
        
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    strategy_cols = [f"Strat_{i:04d}" + ("_TRUE" if i < n_true_alpha else "_NOISE") for i in range(n_strategies)]
    
    return pd.DataFrame(daily_returns, index=dates, columns=strategy_cols)
