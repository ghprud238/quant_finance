"""High-Frequency Market Microstructure & Poisson Order Flow Simulator."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from .order_book import Order, LimitOrderBook, MatchResult


class MarketMicrostructureSimulator:
    """Simulates continuous double auction order book dynamics with Poisson order arrivals,
    cancellations, market executions, and temporary/permanent market impact.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        lambda_limit: float = 5.0,     # Limit order arrival rate (events per unit time)
        lambda_market: float = 2.0,    # Market order arrival rate
        lambda_cancel: float = 1.5,    # Cancellation rate
        perm_impact_coeff: float = 0.0001,  # Permanent market impact parameter (Kyle's lambda)
        name: str = "Simulated_LOB",
    ):
        self.tick_size = tick_size
        self.lambda_limit = lambda_limit
        self.lambda_market = lambda_market
        self.lambda_cancel = lambda_cancel
        self.perm_impact = perm_impact_coeff
        self.name = name

    def initialize_book(
        self,
        initial_mid: float = 100.0,
        initial_half_spread_ticks: int = 2,
        initial_depth: int = 5,
        base_volume: float = 100.0,
    ) -> LimitOrderBook:
        """Initializes an order book with symmetric depth around the starting mid-price."""
        lob = LimitOrderBook(name=self.name)
        half_spread = initial_half_spread_ticks * self.tick_size

        for i in range(initial_depth):
            bid_p = round(initial_mid - half_spread - i * self.tick_size, 4)
            ask_p = round(initial_mid + half_spread + i * self.tick_size, 4)
            vol = round(base_volume * (1.0 + 0.1 * i), 0)

            lob.add_limit_order(Order(
                order_id=f"INIT_BID_{i+1}",
                side='buy',
                price=bid_p,
                volume=vol,
                timestamp=0.0,
            ))
            lob.add_limit_order(Order(
                order_id=f"INIT_ASK_{i+1}",
                side='sell',
                price=ask_p,
                volume=vol,
                timestamp=0.0,
            ))

        return lob

    def simulate(
        self,
        n_events: int = 1000,
        initial_mid: float = 100.0,
        seed: Optional[int] = 42,
    ) -> Tuple[LimitOrderBook, pd.DataFrame]:
        """Runs a discrete-event microstructure simulation over n_events."""
        if seed is not None:
            np.random.seed(seed)

        lob = self.initialize_book(initial_mid=initial_mid)
        events_log: List[Dict[str, Any]] = []

        current_time = 0.0
        total_rate = self.lambda_limit + self.lambda_market + self.lambda_cancel
        order_counter = 0

        p_limit = self.lambda_limit / total_rate
        p_market = self.lambda_market / total_rate

        for step in range(n_events):
            # Inter-arrival time ~ Exp(total_rate)
            dt = np.random.exponential(1.0 / total_rate)
            current_time += dt
            order_counter += 1

            mid = lob.mid_price or initial_mid
            spread = lob.spread or (2 * self.tick_size)

            event_rand = np.random.rand()
            order_id = f"ORD_{order_counter:07d}"

            if event_rand < p_limit:
                # Limit Order Arrival
                event_type = 'LIMIT'
                side = 'buy' if np.random.rand() < 0.5 else 'sell'
                vol = float(np.random.choice([10, 25, 50, 100, 200], p=[0.3, 0.3, 0.2, 0.15, 0.05]))

                # Distance from mid follows exponential decay
                ticks_from_mid = int(np.random.exponential(scale=3.0)) + 1
                if side == 'buy':
                    price = round(max(self.tick_size, mid - ticks_from_mid * self.tick_size), 4)
                else:
                    price = round(mid + ticks_from_mid * self.tick_size, 4)

                order = Order(
                    order_id=order_id,
                    side=side,
                    price=price,
                    volume=vol,
                    timestamp=current_time,
                )
                trades = lob.add_limit_order(order)
                trade_p = trades[0].price if trades else np.nan
                trade_v = sum(t.volume for t in trades) if trades else 0.0

            elif event_rand < p_limit + p_market:
                # Market Order Execution
                event_type = 'MARKET'
                side = 'buy' if np.random.rand() < 0.5 else 'sell'
                vol = float(np.random.choice([10, 25, 50, 100], p=[0.4, 0.3, 0.2, 0.1]))
                trades, filled_vol = lob.execute_market_order(
                    side=side,
                    volume=vol,
                    timestamp=current_time,
                    order_id=order_id,
                )
                trade_p = trades[0].price if trades else np.nan
                trade_v = filled_vol
                price = trade_p if not np.isnan(trade_p) else mid

            else:
                # Cancellation of a random resting order
                event_type = 'CANCEL'
                side = 'buy' if np.random.rand() < 0.5 else 'sell'
                price = np.nan
                vol = 0.0
                trade_p = np.nan
                trade_v = 0.0

                # Pick an active order to cancel
                if lob._order_lookup:
                    cand_ids = [oid for oid, (s, _) in lob._order_lookup.items() if s == side]
                    if cand_ids:
                        target_id = np.random.choice(cand_ids)
                        lob.cancel_order(target_id)

            # Log snapshot state
            snap = lob.get_level2_snapshot(depth=5)
            events_log.append({
                'timestamp': current_time,
                'event_type': event_type,
                'side': side if 'side' in locals() else '',
                'price': price,
                'volume': vol,
                'trade_price': trade_p,
                'trade_volume': trade_v,
                'best_bid': snap.best_bid,
                'best_ask': snap.best_ask,
                'mid_price': snap.mid_price,
                'spread': snap.spread,
                'micro_price': snap.micro_price,
                'order_book_imbalance': snap.order_book_imbalance,
                'total_bid_volume': lob.total_bid_volume,
                'total_ask_volume': lob.total_ask_volume,
            })

        df_log = pd.DataFrame(events_log)
        return lob, df_log
