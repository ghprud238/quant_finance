"""Transaction cost modeling and execution drag for systematic backtesting.

Supports:
- Linear brokerage/exchange fees (basis points)
- Bid-ask spread crossing (half-spread bps)
- Non-linear quadratic market impact / slippage: C(Delta w) = (fees + half_spread) * |Delta w| + 0.5 * gamma * (Delta w)^2
"""

from typing import Union, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TransactionCostModel:
    """Realistic institutional transaction cost and slippage model.

    Attributes:
        fee_bps (float): Flat linear brokerage/exchange fee in basis points (1 bps = 0.0001).
        half_spread_bps (float): Half of the average bid-ask spread in basis points.
        market_impact_gamma (float): Non-linear quadratic market impact coefficient.
        borrow_cost_bps (float): Annualized short-borrow fee in basis points (pro-rated daily).
    """

    def __init__(
        self,
        fee_bps: float = 5.0,
        half_spread_bps: float = 2.5,
        market_impact_gamma: float = 0.0,
        borrow_cost_bps: float = 50.0,
    ) -> None:
        self.fee_bps = fee_bps
        self.half_spread_bps = half_spread_bps
        self.market_impact_gamma = market_impact_gamma
        self.borrow_cost_bps = borrow_cost_bps

        # Convert bps to decimal multipliers
        self.linear_rate = (self.fee_bps + self.half_spread_bps) / 10_000.0
        self.daily_borrow_rate = (self.borrow_cost_bps / 10_000.0) / 252.0

    def compute_turnover(
        self,
        weights: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculates portfolio turnover (Delta w) between consecutive rebalancing dates."""
        if isinstance(weights, pd.DataFrame):
            initial_val = weights.iloc[0].abs()
            weight_diff = weights.diff().fillna(initial_val)
            turnover = weight_diff.abs().sum(axis=1)
        else:
            initial_val = abs(float(weights.iloc[0])) if len(weights) > 0 else 0.0
            weight_diff = weights.diff().fillna(initial_val)
            turnover = weight_diff.abs()
        return turnover

    def compute_trade_cost(
        self,
        delta_w: Union[float, np.ndarray, pd.Series, pd.DataFrame],
    ) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
        """Computes transaction cost as a fraction of portfolio equity for a given weight delta.

        Formula:
            Cost(Delta w) = Linear_Rate * |Delta w| + 0.5 * Gamma * (Delta w)^2
        """
        abs_delta = np.abs(delta_w)
        linear_cost = self.linear_rate * abs_delta
        if self.market_impact_gamma > 0.0:
            impact_cost = 0.5 * self.market_impact_gamma * (abs_delta ** 2)
        else:
            impact_cost = 0.0
        return linear_cost + impact_cost

    def compute_borrow_cost(
        self,
        weights: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Computes pro-rated short borrow financing costs on short positions (w < 0)."""
        if isinstance(weights, pd.DataFrame):
            short_exposure = np.maximum(-weights, 0.0).sum(axis=1)
        else:
            short_exposure = np.maximum(-weights, 0.0)
        return short_exposure * self.daily_borrow_rate

    def apply_costs(
        self,
        gross_returns: pd.Series,
        weights: Union[pd.Series, pd.DataFrame],
    ) -> Tuple[pd.Series, pd.Series]:
        """Deducts rebalancing transaction costs and short borrow costs from gross returns.

        Returns:
            net_returns (pd.Series): Net return series after all execution costs.
            cost_series (pd.Series): Total daily cost drag series.
        """
        turnover = self.compute_turnover(weights)
        trade_costs = self.compute_trade_cost(turnover)
        borrow_costs = self.compute_borrow_cost(weights)

        total_costs = trade_costs + borrow_costs
        net_returns = gross_returns - total_costs
        return net_returns, total_costs

    def cost_breakdown(
        self,
        weights: Union[pd.Series, pd.DataFrame],
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """Returns annual cost breakdown statistics given a weight trajectory."""
        turnover = self.compute_turnover(weights)
        ann_turnover = float(turnover.mean() * periods_per_year)

        linear_drag = self.linear_rate * turnover.mean() * periods_per_year
        if self.market_impact_gamma > 0.0:
            impact_drag = 0.5 * self.market_impact_gamma * (turnover ** 2).mean() * periods_per_year
        else:
            impact_drag = 0.0

        borrow_drag = float(self.compute_borrow_cost(weights).mean() * periods_per_year)
        total_drag = float(linear_drag + impact_drag + borrow_drag)

        return {
            "annualized_turnover": ann_turnover,
            "annualized_fee_spread_drag_bps": float(linear_drag * 10_000.0),
            "annualized_market_impact_drag_bps": float(impact_drag * 10_000.0),
            "annualized_borrow_drag_bps": float(borrow_drag * 10_000.0),
            "total_annualized_drag_bps": float(total_drag * 10_000.0),
            "average_cost_per_turnover_bps": float((total_drag / ann_turnover * 10_000.0) if ann_turnover > 0 else 0.0),
        }
