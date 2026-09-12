"""
Example 02: High-Frequency Trading & Market Making Desk Workflow
================================================================
"""
import nexus_quant as nq

# 1. Level 2 Limit Order Book & Micro-Price
lob = nq.microstructure.order_book.LimitOrderBook(name="HFT_LOB_BTC")
lob.add_limit_order(nq.microstructure.order_book.Order("B1", "buy", price=99.95, volume=1500.0, timestamp=1.0))
lob.add_limit_order(nq.microstructure.order_book.Order("A1", "sell", price=100.05, volume=800.0, timestamp=1.0))

print("L2 Limit Order Book State:")
print(f"  - Mid-Price:    ${lob.mid_price:.2f}")
print(f"  - Spread:       ${lob.spread:.2f}")
print(f"  - Micro-Price:  ${lob.micro_price:.4f}")
print(f"  - Book Imbalance: {lob.order_book_imbalance:+.2f}")

# 2. Avellaneda-Stoikov Inventory-Skewed Quoting Engine
mm = nq.microstructure.avellaneda_stoikov.AvellanedaStoikovMarketMaker(
    risk_aversion_gamma=0.10,
    order_book_liquidity_kappa=1.50,
)
quotes = mm.optimal_quotes(spot=100.0, inventory=5, current_time=0.5)
print("\nAvellaneda-Stoikov Quoting (with +5 Long Inventory):")
print(f"  - Reservation (Indifference) Price: ${quotes.reservation_price:.2f} (skewed downward to dump long inventory)")
print(f"  - Optimal Bid Quote: ${quotes.bid_price:.2f}")
print(f"  - Optimal Ask Quote: ${quotes.ask_price:.2f}")

# 3. Almgren-Chriss Optimal Order Slicing Trajectory
ac = nq.microstructure.optimal_execution.AlmgrenChrissModel(
    total_shares=1_000_000,
    horizon=1.0,
    n_intervals=20,
    volatility=0.30,
)
traj = ac.solve_trajectory(risk_aversion=1e-6)
print("\nAlmgren-Chriss Execution Trajectory:")
print(f"  - Expected Shortfall Cost: ${traj.expected_shortfall:,.2f}")
print(f"  - Trajectory Half-Life:    {traj.half_life:.2f} days")
print(f"  - Initial Interval Slicing: {traj.trade_sizes[0]:,.0f} shares")
