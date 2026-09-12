"""
Unified Data Engine for Nexus Quant Platform.
Generates comprehensive synthetic datasets across all 11 quantitative domains:
1. Equities OHLCV & Cross-Sectional Factors
2. Macro Yield Curves & Sovereign Spreads
3. Crypto & Perpetual Futures Funding Rates
4. Carbon Allowances & Fuel-Switching Spreads
5. Options Volatility Surfaces
6. Alternative Satellite & SEC Disclosures
7. Prediction Market L2 Order Books
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class UnifiedDataEngine:
    """Master generator for multi-domain quantitative market datasets."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_equity_ohlcv(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        Generate correlated multi-asset daily OHLCV series.
        Includes market regime shifts (2020 crash, 2022 rate hike, 2023 tech rally).
        """
        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "SPY"]

        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n_days = len(dates)
        n_assets = len(tickers)

        # Baseline annual drift and volatility
        base_drifts = np.array([0.18, 0.16, 0.14, 0.15, 0.35, 0.11])[:n_assets]
        base_vols = np.array([0.24, 0.22, 0.25, 0.28, 0.40, 0.16])[:n_assets]

        # Empirical correlation matrix
        corr = np.array([
            [1.00, 0.75, 0.68, 0.65, 0.60, 0.85],
            [0.75, 1.00, 0.72, 0.70, 0.64, 0.88],
            [0.68, 0.72, 1.00, 0.68, 0.58, 0.80],
            [0.65, 0.70, 0.68, 1.00, 0.55, 0.78],
            [0.60, 0.64, 0.58, 0.55, 1.00, 0.75],
            [0.85, 0.88, 0.80, 0.78, 0.75, 1.00],
        ])[:n_assets, :n_assets]

        L = np.linalg.cholesky(corr)
        dt = 1.0 / 252.0

        # Simulate correlated daily log returns with jumps
        z = self.rng.standard_normal((n_days, n_assets))
        correlated_z = z @ L.T

        daily_returns = np.zeros((n_days, n_assets))
        for t, date in enumerate(dates):
            vol_mult = 1.0
            drift_adj = 0.0
            # 2020 COVID Crash
            if pd.Timestamp("2020-02-20") <= date <= pd.Timestamp("2020-03-23"):
                vol_mult = 3.5
                drift_adj = -0.60
            # 2022 Inflation & Rate Hike
            elif pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2022-10-31"):
                vol_mult = 1.6
                drift_adj = -0.20
            # 2023-2024 AI Expansion
            elif date >= pd.Timestamp("2023-01-01"):
                vol_mult = 1.1
                drift_adj = 0.10

            current_vols = base_vols * vol_mult
            current_drifts = base_drifts + drift_adj
            daily_returns[t] = (current_drifts - 0.5 * current_vols**2) * dt + current_vols * np.sqrt(dt) * correlated_z[t]

        # Initial prices
        start_prices = np.array([45.0, 85.0, 55.0, 60.0, 15.0, 270.0])[:n_assets]
        price_paths = np.zeros((n_days, n_assets))
        price_paths[0] = start_prices
        for t in range(1, n_days):
            price_paths[t] = price_paths[t-1] * np.exp(daily_returns[t])

        # Construct multi-index dataframe (Ticker, Field)
        tuples = []
        for ticker in tickers:
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                tuples.append((ticker, field))
        columns = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])

        data_matrix = np.zeros((n_days, n_assets * 5))
        for i, ticker in enumerate(tickers):
            closes = price_paths[:, i]
            # Realistic intraday High/Low/Open spreads
            intraday_vol = base_vols[i] * np.sqrt(dt) * 0.8
            opens = closes * (1.0 + self.rng.normal(0, intraday_vol * 0.4, n_days))
            highs = np.maximum(opens, closes) * (1.0 + np.abs(self.rng.normal(0, intraday_vol * 0.7, n_days)))
            lows = np.minimum(opens, closes) * (1.0 - np.abs(self.rng.normal(0, intraday_vol * 0.7, n_days)))
            vols = self.rng.lognormal(16.5, 0.4, n_days)

            data_matrix[:, i*5 + 0] = opens
            data_matrix[:, i*5 + 1] = highs
            data_matrix[:, i*5 + 2] = lows
            data_matrix[:, i*5 + 3] = closes
            data_matrix[:, i*5 + 4] = vols

        df = pd.DataFrame(data_matrix, index=dates, columns=columns)
        return df

    def generate_macro_yield_data(
        self,
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """Generate daily US Treasury par yields (1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)."""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        maturities = [1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        col_names = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

        n_days = len(dates)
        yields = np.zeros((n_days, len(maturities)))

        # Level, Slope, Curvature latent factors
        for t, date in enumerate(dates):
            if date < pd.Timestamp("2020-03-01"):
                beta0 = 2.8; beta1 = -1.2; beta2 = 0.5
            elif date < pd.Timestamp("2022-01-01"):
                beta0 = 1.8; beta1 = -1.6; beta2 = 0.8
            elif date < pd.Timestamp("2023-06-01"):
                beta0 = 4.2; beta1 = 0.8; beta2 = -1.0 # 2s10s inverted
            else:
                beta0 = 4.5; beta1 = -0.3; beta2 = -0.5

            lam = 1.5
            for j, tau in enumerate(maturities):
                factor1 = (1.0 - np.exp(-tau / lam)) / (tau / lam)
                factor2 = factor1 - np.exp(-tau / lam)
                noise = self.rng.normal(0, 0.02)
                yields[t, j] = beta0 + beta1 * factor1 + beta2 * factor2 + noise

        return pd.DataFrame(yields, index=dates, columns=col_names)

    def generate_crypto_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """Generate daily BTC, ETH spot prices, MVRV, and 8-hour perpetual funding rates."""
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        n_days = len(dates)

        btc_ret = self.rng.normal(0.0015, 0.038, n_days)
        btc_price = 10000.0 * np.exp(np.cumsum(btc_ret))

        eth_ret = btc_ret * 1.15 + self.rng.normal(0, 0.025, n_days)
        eth_price = 300.0 * np.exp(np.cumsum(eth_ret))

        # Realized cap and MVRV
        realized_btc = btc_price * (0.65 + 0.20 * np.sin(np.linspace(0, 10, n_days)))
        mvrv = btc_price / realized_btc
        funding_rate_8h = (btc_ret * 0.15 + self.rng.normal(0.0003, 0.0004, n_days)).clip(-0.0075, 0.0075)

        return pd.DataFrame({
            "BTC_Price": btc_price,
            "ETH_Price": eth_price,
            "Realized_BTC": realized_btc,
            "MVRV": mvrv,
            "Funding_Rate_8h": funding_rate_8h,
        }, index=dates)

    def generate_carbon_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """Generate EU ETS EUA carbon allowance prices and spark/dark spread inputs."""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n_days = len(dates)

        carbon_price = 25.0 + np.cumsum(self.rng.normal(0.04, 1.2, n_days)).clip(15.0, 105.0)
        gas_ttf = 18.0 + np.cumsum(self.rng.normal(0.02, 1.5, n_days)).clip(10.0, 150.0)
        coal_ara = 12.0 + np.cumsum(self.rng.normal(0.01, 0.8, n_days)).clip(8.0, 60.0)
        power_base = 45.0 + 1.8 * gas_ttf + 0.35 * carbon_price + self.rng.normal(0, 3.0, n_days)

        return pd.DataFrame({
            "EUA_Carbon_Price": carbon_price,
            "TTF_Gas_Price": gas_ttf,
            "ARA_Coal_Price": coal_ara,
            "Power_Baseload": power_base,
        }, index=dates)
