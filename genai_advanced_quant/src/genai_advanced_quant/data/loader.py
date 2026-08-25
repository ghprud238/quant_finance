"""Comprehensive Data Loader & Synthetic Generation Suite for Frontier Quant Projects (31-35).

Generates and loads:
1. Multi-Year SEC 10-K Filings (Item 1A Risk Factors & Item 7 MD&A) with narrative shifts.
2. Market Option Surface Data (Moneyness K/S x Expiry T x IV / Price).
3. High-Frequency Tick Trade Data for VPIN and Order Flow Toxicity.
4. Multi-Firm Supply-Chain Knowledge Graphs & Revenue Links.
5. In-Sample & Out-of-Sample Non-Stationary Return Matrices for Wasserstein DRO.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


# =========================================================================
# 1. SEC 10-K FILINGS GENERATOR (Module 31)
# =========================================================================

def generate_synthetic_sec_filings(
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, Dict[int, Dict[str, str]]]:
    """Generates realistic multi-year SEC 10-K filings (Item 1A Risk Factors & Item 7 MD&A)."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "TSLA", "META", "JPM", "XOM", "PFE"]
    if years is None:
        years = [2021, 2022, 2023, 2024]
        
    rng = np.random.default_rng(seed)
    
    base_mda_templates = {
        "AAPL": "Our business, results of operations and financial condition have been and could continue to be adversely affected by global economic conditions, supply chain disruptions, geopolitical tensions, and fluctuations in foreign currency exchange rates. Net sales increased due to higher demand for our hardware ecosystem and expanding high-margin subscription services. Operating margin remained resilient driven by product mix optimization.",
        "MSFT": "We generate revenue by offering cloud infrastructure, enterprise software platforms, and productivity suites. Commercial cloud revenue experienced double-digit growth led by enterprise Azure deployments and AI service integrations. Research and development expenses increased as we expand hyperscale datacenter capacity and strategic computing partnerships.",
        "NVDA": "Our graphics processing units and accelerated computing platforms power high-performance artificial intelligence, data centers, gaming, and robotics. Data center compute revenue surged substantially reflecting the transition from general-purpose computing to accelerated computing and generative AI deployment. Supply chain wafer allocation and substrate packaging constraints remain critical operational risks.",
        "TSLA": "We design, manufacture, deploy, and sell fully electric vehicles, energy generation and storage systems, and autonomous software. Automotive gross margin was impacted by vehicle pricing adjustments, manufacturing ramp costs for next-generation platforms, and heightened global competition. Regulatory scrutiny surrounding automated driving software remains an ongoing legal uncertainty.",
        "JPM": "Net interest income was supported by higher benchmark interest rates and disciplined asset-liability management. Consumer banking credit loss provisions normalized as deposit margins stabilized. Investment banking fees reflected macroeconomic uncertainty and dampened debt underwriting volumes.",
    }
    
    base_risk_templates = {
        "AAPL": "Global supply chain disruptions, component shortages, semiconductor fabrication concentration in specific regions, and intellectual property litigation could materially impact product launch timelines and gross margins.",
        "MSFT": "Intense competition in cloud infrastructure, cybersecurity vulnerabilities, regulatory scrutiny of technology acquisitions, and potential failures in artificial intelligence model governance pose material risks to our business.",
        "NVDA": "Export control restrictions on advanced semiconductor hardware, customer concentration among cloud service providers, foundry fabrication dependence, and cyclicality in semiconductor demand could adversely affect operating results.",
        "TSLA": "Uncertainty regarding vehicle production scalability, autonomous driving regulatory approval, raw material battery price volatility, and key person dependency represent substantial risk factors.",
        "JPM": "Credit default cycles, counterparty risk, systemic liquidity crises, monetary policy shifts, and evolving capital adequacy regulations could reduce net interest income and trading revenues.",
    }
    
    drift_modifications = {
        2022: {
            "NVDA": {"mda": "Export control regulations implemented by the US Department of Commerce restricted shipments of advanced A100 and H100 integrated circuits, requiring product redesigns for affected regional markets. Data center revenue outlook reflects revised geopolitical compliance requirements.",
                     "risk": "Geopolitical restrictions, licensing requirements, and potential expansion of trade embargoes on high-performance accelerators could eliminate access to significant global customer segments and reduce long-term earnings."},
            "TSLA": {"mda": "We initiated aggressive price reductions across all model variants to stimulate order velocity amidst rising interest rates. Energy storage deployments expanded rapidly, partially offsetting margin compression in the automotive division.",
                     "risk": "Price competition in primary markets has impaired automotive gross margins. Substantial capital commitments for battery cathode manufacturing and litigation regarding autopilot safety claims could result in adverse financial outcomes."}
        },
        2023: {
            "NVDA": {"mda": "Demand for our Hopper architecture and HGX platforms exceeded expectations exponentially due to broad enterprise adoption of large language models. Operating income increased more than twelve-fold year-over-year. We secured additional packaging capacity with multiple OSAT partners.",
                     "risk": "Concentration risk has intensified as a small number of hyperscale cloud providers represent an outsized percentage of total compute demand. Hardware obsolescence and rapid competitor ASIC development could alter competitive positioning."},
            "META": {"mda": "We executed a company-wide year of efficiency, reducing workforce headcount by 24% and restructuring infrastructure investments. Advertising impressions increased while family of apps daily active users reached record highs.",
                     "risk": "Platform tracking policy changes by mobile operating systems continue to create measurement frictions in ad monetization. Substantial operating losses in Reality Labs hardware initiatives may not generate commensurate economic returns."}
        },
        2024: {
            "AAPL": {"mda": "Regulatory authorities in the European Union enforced the Digital Markets Act, mandating alternative app marketplace access and fee restructuring. Services growth remained robust while research and development focused on spatial computing and on-device machine learning.",
                     "risk": "Antitrust litigation filed by the US Department of Justice and regulatory enforcement actions in multiple jurisdictions could force material alterations to our App Store business model and integrated platform monetization."}
        }
    }
    
    filings: Dict[str, Dict[int, Dict[str, str]]] = {}
    
    for ticker in tickers:
        filings[ticker] = {}
        base_m = base_mda_templates.get(ticker, f"{ticker} produces commercial goods and services. Revenue grew steadily in accordance with general market conditions. Operating expenses were managed prudently across key operating units.")
        base_r = base_risk_templates.get(ticker, f"{ticker} faces market competition, macroeconomic cyclicality, inflation in input costs, foreign exchange risks, and regulatory compliance obligations in all operating jurisdictions.")
        
        curr_m = base_m
        curr_r = base_r
        
        for yr in years:
            if yr in drift_modifications and ticker in drift_modifications[yr]:
                mod = drift_modifications[yr][ticker]
                curr_m = curr_m + " " + mod.get("mda", "")
                curr_r = curr_r + " " + mod.get("risk", "")
            else:
                if rng.random() > 0.65:
                    curr_m = curr_m + f" During fiscal year {yr}, operating performance aligned with strategic guidance."
                if rng.random() > 0.70:
                    curr_r = curr_r + f" In {yr}, interest rate fluctuations and macroeconomic tightening were monitored by risk committees."
                    
            filings[ticker][yr] = {
                "Item_1A_Risk_Factors": curr_r,
                "Item_7_MDA": curr_m,
                "Full_Text": "ITEM 1A. RISK FACTORS\n" + curr_r + "\n\nITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n" + curr_m
            }
            
    return filings


def load_sec_filings(data_dir: Optional[str] = None) -> Dict[str, Dict[int, Dict[str, str]]]:
    """Loads or generates synthetic SEC filings."""
    return generate_synthetic_sec_filings()


# =========================================================================
# 2. MARKET OPTION SURFACE GENERATOR (Module 32)
# =========================================================================

def generate_market_option_surface(
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic market option surface data with volatility smile/skew."""
    rng = np.random.default_rng(seed)
    
    strikes = np.array([70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 130.0])
    expiries = np.array([0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00])
    
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.35, -0.70
    
    records = []
    for T in expiries:
        for K in strikes:
            moneyness = K / spot
            log_m = np.log(moneyness)
            
            base_iv = np.sqrt(theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T))
            skew_adjustment = rho * xi / (2 * kappa) * (log_m / np.sqrt(T))
            smile_curvature = 0.5 * (xi ** 2) / (4 * kappa ** 2) * (log_m ** 2 / T)
            
            iv = max(0.08, base_iv + skew_adjustment + smile_curvature + rng.normal(0, 0.002))
            
            d1 = (np.log(spot / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            
            call_price = spot * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(-q * T) * norm.cdf(-d1)
            
            spread = max(0.05, 0.015 * call_price)
            call_bid = max(0.01, call_price - spread / 2)
            call_ask = call_price + spread / 2
            
            records.append({
                "Expiry": T,
                "Strike": K,
                "Moneyness": moneyness,
                "Implied_Vol": iv,
                "Market_IV": iv,
                "Call_Price": call_price,
                "Call_Mid": call_price,
                "Call_Bid": call_bid,
                "Call_Ask": call_ask,
                "Put_Price": put_price,
                "Put_Mid": put_price,
                "Spot": spot,
                "Rate": r,
                "Dividend": q
            })
            
    df = pd.DataFrame(records)
    return df


def load_market_option_surface(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads market option surface."""
    return generate_market_option_surface()


# =========================================================================
# 3. HIGH-FREQUENCY TICK TRADE DATA FOR VPIN (Module 33)
# =========================================================================

def generate_tick_trade_data(
    n_ticks: int = 15000,
    initial_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic high-frequency tick data with an injected liquidity shock / flash crash."""
    np.random.seed(seed)
    times = pd.date_range("2024-05-06 09:30:00", periods=n_ticks, freq="250ms")

    prices = np.zeros(n_ticks)
    volumes = np.zeros(n_ticks)
    prices[0] = initial_price

    p = initial_price
    for t in range(1, n_ticks):
        if 8000 <= t <= 9500:
            drift = -0.0015
            vol = 0.0040
            trade_vol = np.random.exponential(800) + 400
        else:
            drift = 0.00001
            vol = 0.0008
            trade_vol = np.random.exponential(150) + 10

        ret = np.random.normal(drift, vol)
        p = max(5.0, p * (1.0 + ret))
        prices[t] = p
        volumes[t] = trade_vol

    return pd.DataFrame({
        "Timestamp": times,
        "Price": np.round(prices, 2),
        "Volume": np.round(volumes, 0),
    })


def load_vpin_sample_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads VPIN sample tick data."""
    return generate_tick_trade_data()


# =========================================================================
# 4. SUPPLY-CHAIN KNOWLEDGE GRAPH (Module 34)
# =========================================================================

def generate_supply_chain_network() -> Dict[str, Any]:
    """Generates customer-supplier graph with revenue linkages and price series."""
    customers = ["AAPL", "BA", "NVDA", "AMZN", "TSLA", "MSFT"]
    suppliers = ["TSM", "QCOM", "AVGO", "CRUS", "SWKS", "SPR", "HWM", "TDG", "MRVL", "ALB", "ATSG", "AAOI"]
    all_tickers = customers + suppliers

    links = [
        ("TSM", "AAPL", 0.25), ("CRUS", "AAPL", 0.76), ("SWKS", "AAPL", 0.55), ("AVGO", "AAPL", 0.20),
        ("TSM", "NVDA", 0.15), ("MRVL", "NVDA", 0.10), ("AAOI", "NVDA", 0.42),
        ("SPR", "BA", 0.80),   ("HWM", "BA", 0.35),   ("TDG", "BA", 0.25),
        ("ALB", "TSLA", 0.22),  ("ATSG", "AMZN", 0.38),
    ]

    dates = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n_days = len(dates)
    np.random.seed(42)

    prices_dict = {}
    for t in all_tickers:
        mu = 0.12 / 252
        sigma = 0.25 / np.sqrt(252)
        shocks = np.random.normal(mu, sigma, n_days)
        prices_dict[t] = 100.0 * np.cumprod(1.0 + shocks)

    prices_df = pd.DataFrame(prices_dict, index=dates)

    return {
        "customers": customers,
        "suppliers": suppliers,
        "all_tickers": all_tickers,
        "links": links,
        "prices": prices_df,
    }


def load_supply_chain_market_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads supply chain prices."""
    net = generate_supply_chain_network()
    return net["prices"]


# =========================================================================
# 5. IN-SAMPLE & OUT-OF-SAMPLE DATA FOR WASSERSTEIN DRO (Module 35)
# =========================================================================

def load_dro_returns_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates/loads in-sample and out-of-sample regime-shifted return matrices."""
    np.random.seed(42)
    assets = ["SPY", "QQQ", "TLT", "GLD", "USO", "HYG"]
    
    n_train = 500
    mu_train = np.array([0.12, 0.18, 0.04, 0.06, 0.05, 0.07]) / 252
    vol_train = np.array([0.15, 0.20, 0.12, 0.14, 0.28, 0.09]) / np.sqrt(252)
    train_returns = pd.DataFrame(
        np.random.normal(mu_train, vol_train, (n_train, len(assets))),
        columns=assets,
        index=pd.date_range("2020-01-01", periods=n_train, freq="B"),
    )

    n_test = 250
    mu_test = np.array([0.02, -0.15, -0.12, 0.15, 0.35, -0.05]) / 252
    vol_test = np.array([0.22, 0.30, 0.18, 0.16, 0.38, 0.14]) / np.sqrt(252)
    test_returns = pd.DataFrame(
        np.random.normal(mu_test, vol_test, (n_test, len(assets))),
        columns=assets,
        index=pd.date_range("2022-01-01", periods=n_test, freq="B"),
    )

    return train_returns, test_returns
