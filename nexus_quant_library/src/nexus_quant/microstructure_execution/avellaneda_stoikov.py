"""
Avellaneda-Stoikov (2008) High-Frequency Market Making with Optimal Inventory Risk Control.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class MarketMakingQuotes:
    """Optimal bid and ask quotes with reservation price."""
    mid_price: float
    reservation_price: float
    bid_price: float
    ask_price: float
    bid_half_spread: float
    ask_half_spread: float
    total_spread: float
    current_inventory: int


@dataclass
class MMSimulationResult:
    """Output container for market making session simulation."""
    mid_prices: np.ndarray
    inventory_path: np.ndarray
    cash_path: np.ndarray
    pnl_path: np.ndarray
    total_pnl: float
    sharpe_ratio: float
    mean_inventory: float
    max_inventory_held: int
    total_trades_filled: int
    metrics_summary: pd.DataFrame


class AvellanedaStoikovMarketMaker:
    """
    Avellaneda-Stoikov (2008) Optimal High-Frequency Market Making Engine.
    Dynamically adjusts quote spreads around reservation price r(s, q, t) to mitigate inventory risk.
    """

    def __init__(
        self,
        risk_aversion_gamma: float = 0.10,
        order_book_liquidity_kappa: float = 1.50,
        trade_arrival_intensity_A: float = 140.0,
        asset_volatility_sigma: float = 0.25,
        time_horizon_T: float = 1.0,
        max_inventory_limit: int = 25,
    ):
        self.gamma = risk_aversion_gamma
        self.kappa = order_book_liquidity_kappa
        self.A = trade_arrival_intensity_A
        self.sigma = asset_volatility_sigma
        self.T = time_horizon_T
        self.max_inventory = max_inventory_limit

    def reservation_price(self, spot: float, inventory: int, current_time: float) -> float:
        """
        Reservation (indifference) price:
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t).
        """
        time_left = max(0.0, self.T - current_time)
        return float(spot - inventory * self.gamma * (self.sigma**2) * time_left)

    def optimal_quotes(
        self,
        spot: float,
        inventory: int,
        current_time: float,
    ) -> MarketMakingQuotes:
        """
        Compute optimal bid and ask quotes:
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t).
        delta^a + delta^b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa).
        """
        time_left = max(0.0, self.T - current_time)
        r_p = self.reservation_price(spot, inventory, current_time)

        # Baseline half-spread under zero inventory
        half_spread_base = (1.0 / self.gamma) * np.log(1.0 + (self.gamma / self.kappa))
        time_spread_adj = 0.5 * self.gamma * (self.sigma**2) * time_left

        d_bid = half_spread_base + time_spread_adj
        d_ask = half_spread_base + time_spread_adj

        p_bid = r_p - d_bid
        p_ask = r_p + d_ask

        # If inventory limit reached, pull quotes
        if inventory >= self.max_inventory:
            p_bid = -1e6  # do not bid further
        elif inventory <= -self.max_inventory:
            p_ask = 1e6   # do not offer further

        return MarketMakingQuotes(
            mid_price=spot,
            reservation_price=r_p,
            bid_price=p_bid,
            ask_price=p_ask,
            bid_half_spread=d_bid,
            ask_half_spread=d_ask,
            total_spread=d_bid + d_ask,
            current_inventory=inventory,
        )

    def simulate_session(
        self,
        initial_price: float = 100.0,
        n_steps: int = 1000,
        seed: int = 42,
    ) -> MMSimulationResult:
        """
        Simulate high-frequency trading session with Poisson order arrivals and mid-price Brownian motion.
        """
        np.random.seed(seed)
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)

        mid_prices = np.zeros(n_steps + 1)
        inventories = np.zeros(n_steps + 1, dtype=int)
        cash_balances = np.zeros(n_steps + 1)
        portfolio_values = np.zeros(n_steps + 1)

        mid_prices[0] = initial_price
        portfolio_values[0] = 0.0

        current_inv = 0
        current_cash = 0.0
        trades_filled = 0

        for t in range(n_steps):
            time_now = t * dt
            s_curr = mid_prices[t]

            # Mid-price random walk: dS = sigma * dW
            z = np.random.standard_normal()
            s_next = s_curr + self.sigma * s_curr * sqrt_dt * z
            mid_prices[t + 1] = s_next

            # Compute optimal quotes
            quotes = self.optimal_quotes(s_curr, current_inv, time_now)

            # Order fill probabilities based on distance from mid-price
            delta_b = s_curr - quotes.bid_price
            delta_a = quotes.ask_price - s_curr

            lambda_b = self.A * np.exp(-self.kappa * max(delta_b, 0.01))
            lambda_a = self.A * np.exp(-self.kappa * max(delta_a, 0.01))

            prob_fill_bid = 1.0 - np.exp(-lambda_b * dt)
            prob_fill_ask = 1.0 - np.exp(-lambda_a * dt)

            # Execution checks
            if current_inv < self.max_inventory and np.random.rand() < prob_fill_bid:
                # Buy fill: increase inventory by 1, pay bid price
                current_inv += 1
                current_cash -= quotes.bid_price
                trades_filled += 1

            if current_inv > -self.max_inventory and np.random.rand() < prob_fill_ask:
                # Sell fill: decrease inventory by 1, receive ask price
                current_inv -= 1
                current_cash += quotes.ask_price
                trades_filled += 1

            inventories[t + 1] = current_inv
            cash_balances[t + 1] = current_cash
            # Mark-to-market P&L = Cash + Inventory * Current Mid
            portfolio_values[t + 1] = current_cash + current_inv * s_next

        total_pnl = portfolio_values[-1]
        pnl_diffs = np.diff(portfolio_values)
        sharpe = float(np.mean(pnl_diffs) / max(np.std(pnl_diffs), 1e-6) * np.sqrt(n_steps))

        summary_df = pd.DataFrame({
            "Metric": [
                "Total Realized P&L ($)",
                "Annualized Quoting Sharpe",
                "Mean Inventory Maintained",
                "Max Absolute Inventory",
                "Total Trades Filled",
            ],
            "Value": [
                f"${total_pnl:+,.2f}",
                f"{sharpe:.2f}",
                f"{np.mean(inventories):+.2f}",
                f"{np.max(np.abs(inventories))}",
                f"{trades_filled}",
            ]
        })

        return MMSimulationResult(
            mid_prices=mid_prices,
            inventory_path=inventories,
            cash_path=cash_balances,
            pnl_path=portfolio_values,
            total_pnl=float(total_pnl),
            sharpe_ratio=float(sharpe),
            mean_inventory=float(np.mean(inventories)),
            max_inventory_held=int(np.max(np.abs(inventories))),
            total_trades_filled=trades_filled,
            metrics_summary=summary_df,
        )
