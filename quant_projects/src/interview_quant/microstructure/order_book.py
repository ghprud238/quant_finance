"""Limit Order Book with Level 2 Matching Engine, FIFO Price-Time Priority, and Microstructure Properties."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np
import pandas as pd


@dataclass
class Order:
    """Represents an individual order submitted to the Limit Order Book."""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    volume: float
    timestamp: float
    order_type: str = 'limit'  # 'limit', 'market', or 'cancel'

    def __post_init__(self):
        self.side = self.side.lower()
        self.order_type = self.order_type.lower()
        if self.side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {self.side}. Must be 'buy' or 'sell'.")
        if self.order_type not in ('limit', 'market', 'cancel'):
            raise ValueError(f"Invalid order_type: {self.order_type}.")
        if self.volume <= 0:
            raise ValueError(f"Volume must be strictly positive, got {self.volume}.")
        if self.order_type == 'limit' and self.price <= 0:
            raise ValueError(f"Price must be strictly positive for limit orders, got {self.price}.")


@dataclass
class MatchResult:
    """Represents a trade execution resulting from order matching."""
    trade_id: str
    price: float
    volume: float
    buyer_id: str
    seller_id: str
    timestamp: float
    is_maker_buyer: bool


@dataclass
class Level2Snapshot:
    """Snapshot of Level 2 market depth."""
    bids: List[Tuple[float, float]]  # [(price, volume), ...] descending by price
    asks: List[Tuple[float, float]]  # [(price, volume), ...] ascending by price
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    micro_price: Optional[float]
    order_book_imbalance: float


class LimitOrderBook:
    """Continuous double-auction Limit Order Book with FIFO Price-Time Priority.

    Maintains separate price ladders for Bids (sorted descending) and Asks (sorted ascending).
    At each price level, orders are queued in a FIFO deque to ensure strict time priority.
    """

    def __init__(self, name: str = "LOB"):
        self.name = name
        # Bids: price -> deque[Order] (prices sorted descending)
        self._bids: Dict[float, deque[Order]] = {}
        # Asks: price -> deque[Order] (prices sorted ascending)
        self._asks: Dict[float, deque[Order]] = {}
        # Order location lookup for fast O(1) cancellations: order_id -> (side, price)
        self._order_lookup: Dict[str, Tuple[str, float]] = {}
        self._trade_counter = 0
        self._last_trade_price: Optional[float] = None
        self._last_trade_timestamp: float = 0.0

    @property
    def best_bid(self) -> Optional[float]:
        """Highest resting bid price in the book."""
        valid_prices = [p for p, q in self._bids.items() if len(q) > 0]
        return max(valid_prices) if valid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest resting ask price in the book."""
        valid_prices = [p for p, q in self._asks.items() if len(q) > 0]
        return min(valid_prices) if valid_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid-price between best bid and best ask: (P_b + P_a) / 2."""
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb or ba or self._last_trade_price

    @property
    def spread(self) -> Optional[float]:
        """Bid-Ask spread: P_a - P_b."""
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return max(0.0, ba - bb)
        return None

    @property
    def total_bid_volume(self) -> float:
        """Aggregate resting volume across all bid levels."""
        return sum(sum(o.volume for o in q) for q in self._bids.values())

    @property
    def total_ask_volume(self) -> float:
        """Aggregate resting volume across all ask levels."""
        return sum(sum(o.volume for o in q) for q in self._asks.values())

    @property
    def best_bid_volume(self) -> float:
        """Resting volume at the best bid level."""
        bb = self.best_bid
        if bb is not None and bb in self._bids:
            return sum(o.volume for o in self._bids[bb])
        return 0.0

    @property
    def best_ask_volume(self) -> float:
        """Resting volume at the best ask level."""
        ba = self.best_ask
        if ba is not None and ba in self._asks:
            return sum(o.volume for o in self._asks[ba])
        return 0.0

    @property
    def order_book_imbalance(self) -> float:
        """Level 1 Order Book Imbalance (OBI): I = (V_b - V_a) / (V_b + V_a) in [-1, +1]."""
        vb = self.best_bid_volume
        va = self.best_ask_volume
        total = vb + va
        if total > 0:
            return (vb - va) / total
        return 0.0

    @property
    def micro_price(self) -> Optional[float]:
        """Volume-Weighted Mid-Price (Micro-Price): P_micro = (V_b * P_a + V_a * P_b) / (V_b + V_a)."""
        bb = self.best_bid
        ba = self.best_ask
        vb = self.best_bid_volume
        va = self.best_ask_volume
        if bb is not None and ba is not None:
            total_vol = vb + va
            if total_vol > 0:
                return (vb * ba + va * bb) / total_vol
            return (bb + ba) / 2.0
        return self.mid_price

    def add_limit_order(self, order: Order) -> List[MatchResult]:
        """Inserts a limit order into the book, matching across opposing levels if crossing."""
        trades: List[MatchResult] = []
        rem_vol = order.volume

        if order.side == 'buy':
            # Cross with asks while order.price >= best_ask
            while rem_vol > 1e-8 and self.best_ask is not None and order.price >= self.best_ask:
                best_ask_price = self.best_ask
                ask_queue = self._asks[best_ask_price]

                while rem_vol > 1e-8 and len(ask_queue) > 0:
                    resting_order = ask_queue[0]
                    trade_vol = min(rem_vol, resting_order.volume)
                    self._trade_counter += 1

                    trades.append(MatchResult(
                        trade_id=f"T{self._trade_counter:07d}",
                        price=best_ask_price,
                        volume=trade_vol,
                        buyer_id=order.order_id,
                        seller_id=resting_order.order_id,
                        timestamp=order.timestamp,
                        is_maker_buyer=False,
                    ))

                    self._last_trade_price = best_ask_price
                    self._last_trade_timestamp = order.timestamp
                    rem_vol -= trade_vol
                    resting_order.volume -= trade_vol

                    if resting_order.volume <= 1e-8:
                        ask_queue.popleft()
                        self._order_lookup.pop(resting_order.order_id, None)

                if len(ask_queue) == 0:
                    self._asks.pop(best_ask_price, None)

            # Resting residual volume
            if rem_vol > 1e-8:
                order.volume = rem_vol
                p = round(order.price, 4)
                if p not in self._bids:
                    self._bids[p] = deque()
                self._bids[p].append(order)
                self._order_lookup[order.order_id] = ('buy', p)

        else:  # sell order
            # Cross with bids while order.price <= best_bid
            while rem_vol > 1e-8 and self.best_bid is not None and order.price <= self.best_bid:
                best_bid_price = self.best_bid
                bid_queue = self._bids[best_bid_price]

                while rem_vol > 1e-8 and len(bid_queue) > 0:
                    resting_order = bid_queue[0]
                    trade_vol = min(rem_vol, resting_order.volume)
                    self._trade_counter += 1

                    trades.append(MatchResult(
                        trade_id=f"T{self._trade_counter:07d}",
                        price=best_bid_price,
                        volume=trade_vol,
                        buyer_id=resting_order.order_id,
                        seller_id=order.order_id,
                        timestamp=order.timestamp,
                        is_maker_buyer=True,
                    ))

                    self._last_trade_price = best_bid_price
                    self._last_trade_timestamp = order.timestamp
                    rem_vol -= trade_vol
                    resting_order.volume -= trade_vol

                    if resting_order.volume <= 1e-8:
                        bid_queue.popleft()
                        self._order_lookup.pop(resting_order.order_id, None)

                if len(bid_queue) == 0:
                    self._bids.pop(best_bid_price, None)

            # Resting residual volume
            if rem_vol > 1e-8:
                order.volume = rem_vol
                p = round(order.price, 4)
                if p not in self._asks:
                    self._asks[p] = deque()
                self._asks[p].append(order)
                self._order_lookup[order.order_id] = ('sell', p)

        return trades

    def execute_market_order(self, side: str, volume: float, timestamp: float = 0.0, order_id: Optional[str] = None) -> Tuple[List[MatchResult], float]:
        """Executes a market order walking through available depth on the opposing side."""
        side = side.lower()
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid side: {side}")
        if volume <= 0:
            return [], 0.0

        if order_id is None:
            order_id = f"MKT_{self._trade_counter+1}"

        trades: List[MatchResult] = []
        rem_vol = volume

        if side == 'buy':
            while rem_vol > 1e-8 and self.best_ask is not None:
                best_p = self.best_ask
                queue = self._asks[best_p]

                while rem_vol > 1e-8 and len(queue) > 0:
                    maker = queue[0]
                    trade_vol = min(rem_vol, maker.volume)
                    self._trade_counter += 1

                    trades.append(MatchResult(
                        trade_id=f"T{self._trade_counter:07d}",
                        price=best_p,
                        volume=trade_vol,
                        buyer_id=order_id,
                        seller_id=maker.order_id,
                        timestamp=timestamp,
                        is_maker_buyer=False,
                    ))

                    self._last_trade_price = best_p
                    self._last_trade_timestamp = timestamp
                    rem_vol -= trade_vol
                    maker.volume -= trade_vol

                    if maker.volume <= 1e-8:
                        queue.popleft()
                        self._order_lookup.pop(maker.order_id, None)

                if len(queue) == 0:
                    self._asks.pop(best_p, None)

        else:  # sell market order
            while rem_vol > 1e-8 and self.best_bid is not None:
                best_p = self.best_bid
                queue = self._bids[best_p]

                while rem_vol > 1e-8 and len(queue) > 0:
                    maker = queue[0]
                    trade_vol = min(rem_vol, maker.volume)
                    self._trade_counter += 1

                    trades.append(MatchResult(
                        trade_id=f"T{self._trade_counter:07d}",
                        price=best_p,
                        volume=trade_vol,
                        buyer_id=maker.order_id,
                        seller_id=order_id,
                        timestamp=timestamp,
                        is_maker_buyer=True,
                    ))

                    self._last_trade_price = best_p
                    self._last_trade_timestamp = timestamp
                    rem_vol -= trade_vol
                    maker.volume -= trade_vol

                    if maker.volume <= 1e-8:
                        queue.popleft()
                        self._order_lookup.pop(maker.order_id, None)

                if len(queue) == 0:
                    self._bids.pop(best_p, None)

        filled_volume = volume - rem_vol
        return trades, filled_volume

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an existing resting order by order_id."""
        if order_id not in self._order_lookup:
            return False

        side, price = self._order_lookup.pop(order_id)
        book = self._bids if side == 'buy' else self._asks

        if price in book:
            queue = book[price]
            for order in list(queue):
                if order.order_id == order_id:
                    queue.remove(order)
                    break
            if len(queue) == 0:
                book.pop(price, None)
            return True

        return False

    def get_level2_snapshot(self, depth: int = 10) -> Level2Snapshot:
        """Returns structured Level 2 depth snapshot."""
        # Sorted bids (descending)
        bids_sorted = []
        for p in sorted(self._bids.keys(), reverse=True):
            vol = sum(o.volume for o in self._bids[p])
            if vol > 1e-8:
                bids_sorted.append((p, vol))
            if len(bids_sorted) >= depth:
                break

        # Sorted asks (ascending)
        asks_sorted = []
        for p in sorted(self._asks.keys()):
            vol = sum(o.volume for o in self._asks[p])
            if vol > 1e-8:
                asks_sorted.append((p, vol))
            if len(asks_sorted) >= depth:
                break

        return Level2Snapshot(
            bids=bids_sorted,
            asks=asks_sorted,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            mid_price=self.mid_price,
            spread=self.spread,
            micro_price=self.micro_price,
            order_book_imbalance=self.order_book_imbalance,
        )

    def get_snapshot_table(self, depth: int = 5) -> pd.DataFrame:
        """Returns formatted Level 2 DataFrame matching the screenshot table."""
        snap = self.get_level2_snapshot(depth=depth)
        rows = []

        for i in range(depth):
            bid_p = snap.bids[i][0] if i < len(snap.bids) else np.nan
            bid_v = snap.bids[i][1] if i < len(snap.bids) else np.nan
            ask_p = snap.asks[i][0] if i < len(snap.asks) else np.nan
            ask_v = snap.asks[i][1] if i < len(snap.asks) else np.nan

            rows.append({
                'Level': i + 1,
                'Bid_Volume': bid_v,
                'Bid_Price': bid_p,
                'Ask_Price': ask_p,
                'Ask_Volume': ask_v,
            })

        return pd.DataFrame(rows)
