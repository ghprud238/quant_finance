"""
Level 2 Limit Order Book with FIFO Price-Time Priority, Matching Engine, OBI & Micro-Price.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class Order:
    """Represents a limit, market, or cancellation order."""
    order_id: str
    side: str        # 'buy' or 'sell'
    price: float     # Limit price (0.0 for market orders)
    volume: float    # Quantity of shares/contracts
    timestamp: float
    order_type: str = "limit"  # 'limit', 'market', 'cancel'


@dataclass
class Trade:
    """Represents an executed trade print."""
    trade_id: str
    maker_order_id: str
    taker_order_id: str
    price: float
    volume: float
    timestamp: float
    taker_side: str


class LimitOrderBook:
    """
    Continuous double auction Level 2 Limit Order Book.
    Maintains Bid ladder (sorted descending) and Ask ladder (sorted ascending) with FIFO deques.
    """

    def __init__(self, name: str = "LOB_MAIN"):
        self.name = name
        # Price -> deque of Orders (FIFO queue per price level)
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}
        self.orders_by_id: Dict[str, Order] = {}
        self.trade_history: List[Trade] = []
        self._trade_counter = 0

    @property
    def best_bid(self) -> Optional[float]:
        """Current highest bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())

    @property
    def best_ask(self) -> Optional[float]:
        """Current lowest ask price."""
        if not self.asks:
            return None
        return min(self.asks.keys())

    @property
    def spread(self) -> Optional[float]:
        """Bid-Ask spread: best_ask - best_bid."""
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return float(ba - bb)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid-point price between best bid and best ask."""
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return float((bb + ba) / 2.0)
        return None

    @property
    def total_bid_volume(self) -> float:
        """Total volume resting on the bid ladder."""
        return sum(sum(o.volume for o in q) for q in self.bids.values())

    @property
    def total_ask_volume(self) -> float:
        """Total volume resting on the ask ladder."""
        return sum(sum(o.volume for o in q) for q in self.asks.values())

    @property
    def order_book_imbalance(self) -> float:
        """
        Order Book Imbalance (OBI) at top of book:
        I = (V_b - V_a) / (V_b + V_a) in [-1.0, +1.0].
        """
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return 0.0
        vb = sum(o.volume for o in self.bids[bb])
        va = sum(o.volume for o in self.asks[ba])
        denom = vb + va
        if denom <= 1e-8:
            return 0.0
        return float((vb - va) / denom)

    @property
    def micro_price(self) -> Optional[float]:
        """
        Volume-Weighted Micro-Price:
        P_micro = (V_b * P_a + V_a * P_b) / (V_b + V_a).
        """
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return self.mid_price
        vb = sum(o.volume for o in self.bids[bb])
        va = sum(o.volume for o in self.asks[ba])
        denom = vb + va
        if denom <= 1e-8:
            return self.mid_price
        return float((vb * ba + va * bb) / denom)

    def add_limit_order(self, order: Order) -> List[Trade]:
        """
        Insert limit order into book. If price crosses spread, matches immediately against resting liquidity.
        """
        executed_trades: List[Trade] = []
        remaining_vol = order.volume
        side = order.side.lower()

        if side == "buy":
            # Match against resting asks if order.price >= best_ask
            while remaining_vol > 1e-8 and self.best_ask is not None and order.price >= self.best_ask:
                ba = self.best_ask
                queue = self.asks[ba]
                maker = queue[0]
                fill_vol = min(remaining_vol, maker.volume)

                self._trade_counter += 1
                t = Trade(
                    trade_id=f"T{self._trade_counter}",
                    maker_order_id=maker.order_id,
                    taker_order_id=order.order_id,
                    price=ba,
                    volume=fill_vol,
                    timestamp=order.timestamp,
                    taker_side="buy",
                )
                executed_trades.append(t)
                self.trade_history.append(t)

                maker.volume -= fill_vol
                remaining_vol -= fill_vol

                if maker.volume <= 1e-8:
                    queue.popleft()
                    self.orders_by_id.pop(maker.order_id, None)
                    if not queue:
                        del self.asks[ba]

            # Place remaining volume on bid ladder
            if remaining_vol > 1e-8:
                order.volume = remaining_vol
                p_level = order.price
                if p_level not in self.bids:
                    self.bids[p_level] = deque()
                self.bids[p_level].append(order)
                self.orders_by_id[order.order_id] = order

        else:
            # Match against resting bids if order.price <= best_bid
            while remaining_vol > 1e-8 and self.best_bid is not None and order.price <= self.best_bid:
                bb = self.best_bid
                queue = self.bids[bb]
                maker = queue[0]
                fill_vol = min(remaining_vol, maker.volume)

                self._trade_counter += 1
                t = Trade(
                    trade_id=f"T{self._trade_counter}",
                    maker_order_id=maker.order_id,
                    taker_order_id=order.order_id,
                    price=bb,
                    volume=fill_vol,
                    timestamp=order.timestamp,
                    taker_side="sell",
                )
                executed_trades.append(t)
                self.trade_history.append(t)

                maker.volume -= fill_vol
                remaining_vol -= fill_vol

                if maker.volume <= 1e-8:
                    queue.popleft()
                    self.orders_by_id.pop(maker.order_id, None)
                    if not queue:
                        del self.bids[bb]

            # Place remaining volume on ask ladder
            if remaining_vol > 1e-8:
                order.volume = remaining_vol
                p_level = order.price
                if p_level not in self.asks:
                    self.asks[p_level] = deque()
                self.asks[p_level].append(order)
                self.orders_by_id[order.order_id] = order

        return executed_trades

    def execute_market_order(self, side: str, volume: float, timestamp: float) -> Tuple[List[Trade], float]:
        """
        Execute market order walking the book depth until filled or book depleted.
        Returns list of executed trades and total filled volume.
        """
        side_clean = side.lower()
        dummy_price = float("inf") if side_clean == "buy" else 0.0
        m_order = Order(
            order_id=f"MKT_{self._trade_counter+1}",
            side=side_clean,
            price=dummy_price,
            volume=volume,
            timestamp=timestamp,
            order_type="market",
        )
        trades = self.add_limit_order(m_order)
        filled = sum(t.volume for t in trades)
        return trades, filled

    def cancel_order(self, order_id: str) -> bool:
        """Remove a resting order from the book."""
        if order_id not in self.orders_by_id:
            return False
        order = self.orders_by_id.pop(order_id)
        p = order.price
        side = order.side.lower()

        target_dict = self.bids if side == "buy" else self.asks
        if p in target_dict:
            queue = target_dict[p]
            queue = deque(o for o in queue if o.order_id != order_id)
            if queue:
                target_dict[p] = queue
            else:
                del target_dict[p]
            return True
        return False

    def get_snapshot_table(self, depth: int = 5) -> pd.DataFrame:
        """Return formatted Level 2 price depth ladder DataFrame."""
        sorted_bids = sorted(self.bids.keys(), reverse=True)[:depth]
        sorted_asks = sorted(self.asks.keys())[:depth]

        bid_prices = sorted_bids + [np.nan] * (depth - len(sorted_bids))
        bid_vols = [sum(o.volume for o in self.bids[p]) for p in sorted_bids] + [np.nan] * (depth - len(sorted_bids))

        ask_prices = sorted_asks + [np.nan] * (depth - len(sorted_asks))
        ask_vols = [sum(o.volume for o in self.asks[p]) for p in sorted_asks] + [np.nan] * (depth - len(sorted_asks))

        return pd.DataFrame({
            "Bid_Volume": bid_vols,
            "Bid_Price": bid_prices,
            "Ask_Price": ask_prices,
            "Ask_Volume": ask_vols,
        })
