"""Supply-Chain Knowledge Graph & GNN Spillover Momentum (Module 34).

Implementation of economic network alpha models inspired by:
- Cohen & Frazzini (2008): 'Economic Links and Predictable Returns' (Journal of Finance)
- Kipf & Welling (2016): 'Semi-Supervised Classification with Graph Convolutional Networks' (ICLR)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class SupplyChainLink:
    """Directed supplier-to-customer economic revenue link."""
    supplier: str
    customer: str
    revenue_pct: float  # Percentage of supplier revenue derived from customer (0.0 to 1.0)
    relationship_type: str = "supplier"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier": self.supplier,
            "customer": self.customer,
            "revenue_pct": self.revenue_pct,
            "relationship_type": self.relationship_type,
        }


@dataclass
class SpilloverSignal:
    """Cross-sectional lead-lag spillover signal for a given company."""
    date: Any
    supplier: str
    customer_momentum: float
    gcn_propagated_score: float
    residual_alpha_signal: float
    target_weight: float


@dataclass
class GraphAlphaResult:
    """Container for supply-chain knowledge graph alpha results, metrics, and backtests."""
    signals_df: pd.DataFrame
    centrality_df: pd.DataFrame
    equity_curve: pd.Series
    standalone_momentum_curve: pd.Series
    benchmark_curve: pd.Series
    ic_series: pd.Series
    metrics: Dict[str, Any]
    links_summary: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        """Returns formatted DataFrame summary of strategy metrics vs standalone momentum."""
        rows = [
            {"Metric": "Strategy Annualized Return (CAGR)", "Value": f"{self.metrics.get('cagr', 0.0):+.2%}"},
            {"Metric": "Strategy Annualized Volatility", "Value": f"{self.metrics.get('volatility', 0.0):.2%}"},
            {"Metric": "Strategy Sharpe Ratio (Rf=2%)", "Value": f"{self.metrics.get('sharpe_ratio', 0.0):.2f}"},
            {"Metric": "Strategy Sortino Ratio", "Value": f"{self.metrics.get('sortino_ratio', 0.0):.2f}"},
            {"Metric": "Strategy Maximum Drawdown", "Value": f"{self.metrics.get('max_drawdown', 0.0):.2%}"},
            {"Metric": "Daily Win Rate", "Value": f"{self.metrics.get('win_rate', 0.0):.1%}"},
            {"Metric": "Mean Information Coefficient (IC)", "Value": f"{self.metrics.get('mean_ic', 0.0):+.4f}"},
            {"Metric": "IC Information Ratio (IR_IC)", "Value": f"{self.metrics.get('ic_ir', 0.0):.2f}"},
            {"Metric": "Standalone Momentum Sharpe", "Value": f"{self.metrics.get('standalone_sharpe', 0.0):.2f}"},
            {"Metric": "Correlation to Traditional Momentum", "Value": f"{self.metrics.get('corr_to_momentum', 0.0):.2f}"},
        ]
        return pd.DataFrame(rows)


class SupplyChainNetwork:
    """Directed economic network graph representation with adjacency matrix operations."""

    def __init__(self, links: List[SupplyChainLink], all_nodes: Optional[List[str]] = None) -> None:
        self.links = links

        # Extract unique nodes
        if all_nodes is not None:
            self.nodes = sorted(list(set(all_nodes)))
        else:
            node_set = set()
            for link in links:
                node_set.add(link.supplier)
                node_set.add(link.customer)
            self.nodes = sorted(list(node_set))

        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)

        # Build raw adjacency matrix A (row = supplier, col = customer)
        self.adjacency_matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for link in links:
            if link.supplier in self.node_to_idx and link.customer in self.node_to_idx:
                i = self.node_to_idx[link.supplier]
                j = self.node_to_idx[link.customer]
                self.adjacency_matrix[i, j] = link.revenue_pct

        # Build normalized GCN propagation matrix
        self.gcn_normalized_adj = self._compute_gcn_laplacian(self.adjacency_matrix)

    @staticmethod
    def _compute_gcn_laplacian(A: np.ndarray) -> np.ndarray:
        """Computes Kipf-Welling GCN normalized adjacency matrix: D~^{-1/2} A~ D~^{-1/2}."""
        M = A.shape[0]
        A_tilde = A + np.eye(M)  # Add self-loops
        d_tilde = np.sum(A_tilde, axis=1)
        # Avoid division by zero
        d_inv_sqrt = np.power(np.maximum(d_tilde, 1e-12), -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def compute_pagerank(self, damping: float = 0.85, max_iter: int = 200, tol: float = 1e-8) -> pd.Series:
        """Computes PageRank centrality to identify systemic demand hubs in supply chain."""
        M = self.n_nodes
        if M == 0:
            return pd.Series()

        # Adjacency: edge from supplier to customer (A[i, j] > 0 means i supplies j)
        # In supply-chain flow, economic influence flows customer -> supplier or supplier -> customer
        # We model flow from suppliers to their major customer demand hubs (A_adj)
        A = self.adjacency_matrix.copy()

        # Row-stochastic transition matrix
        row_sums = A.sum(axis=1, keepdims=True)
        # For dangling nodes (no outgoing edges), distribute uniformly
        dangling_mask = (row_sums.flatten() == 0)
        
        P = np.zeros_like(A)
        non_dangling = ~dangling_mask
        P[non_dangling] = A[non_dangling] / row_sums[non_dangling]
        P[dangling_mask] = 1.0 / M

        # Power iteration
        v = np.ones(M) / M
        teleport = np.ones(M) / M

        for _ in range(max_iter):
            v_next = damping * (v @ P) + (1.0 - damping) * teleport
            v_next = v_next / np.sum(v_next)  # Ensure exact normalization
            if np.linalg.norm(v_next - v, 1) < tol:
                break
            v = v_next

        return pd.Series(v, index=self.nodes, name="PageRank")

    def compute_concentration_hhi(self) -> pd.Series:
        """Computes Herfindahl-Hirschman Index (HHI) of customer revenue concentration."""
        # Sum of squared revenue percentages per supplier
        hhi = np.sum(np.square(self.adjacency_matrix), axis=1)
        return pd.Series(hhi, index=self.nodes, name="Customer_HHI")

    def get_centrality_table(self) -> pd.DataFrame:
        """Returns structured centrality and vulnerability analysis table."""
        pr = self.compute_pagerank()
        hhi = self.compute_concentration_hhi()
        in_degree = np.sum(self.adjacency_matrix > 0, axis=1)  # Number of customers
        out_degree = np.sum(self.adjacency_matrix > 0, axis=0)  # Number of suppliers

        df = pd.DataFrame({
            "Ticker": self.nodes,
            "PageRank_Centrality": pr.values,
            "Customer_Count": in_degree,
            "Supplier_Count": out_degree,
            "Customer_Concentration_HHI": hhi.values,
            "Total_Customer_Revenue_Pct": np.sum(self.adjacency_matrix, axis=1),
        })
        return df.sort_values(by="PageRank_Centrality", ascending=False).reset_index(drop=True)


class SupplyChainGraphAlpha:
    """Supply-Chain Knowledge Graph Alpha & GNN Lead-Lag Momentum Engine.

    Propagates customer earnings surprises, sales revisions, and short-term price shocks
    downstream to suppliers via graph convolutional message passing.
    """

    def __init__(
        self,
        network: Optional[SupplyChainNetwork] = None,
        n_gcn_layers: int = 2,
        lead_lag_window: int = 5,
        rebalance_freq: int = 5,
        target_vol: float = 0.12,
        transaction_cost_bps: float = 5.0,
    ) -> None:
        self.network = network if network is not None else self.build_sample_supply_chain_network()
        self.n_gcn_layers = n_gcn_layers
        self.lead_lag_window = lead_lag_window
        self.rebalance_freq = rebalance_freq
        self.target_vol = target_vol
        self.transaction_cost_bps = transaction_cost_bps

    @staticmethod
    def build_sample_supply_chain_network() -> SupplyChainNetwork:
        """Constructs realistic institutional supply-chain graph connecting major ecosystem nodes."""
        links = [
            # Apple (AAPL) Ecosystem
            SupplyChainLink(supplier="SWKS", customer="AAPL", revenue_pct=0.58),
            SupplyChainLink(supplier="QRVO", customer="AAPL", revenue_pct=0.48),
            SupplyChainLink(supplier="CRUS", customer="AAPL", revenue_pct=0.76),
            SupplyChainLink(supplier="TSM", customer="AAPL", revenue_pct=0.25),
            SupplyChainLink(supplier="HON", customer="AAPL", revenue_pct=0.15),
            # Nvidia (NVDA) Ecosystem
            SupplyChainLink(supplier="TSM", customer="NVDA", revenue_pct=0.35),
            SupplyChainLink(supplier="SMCI", customer="NVDA", revenue_pct=0.65),
            SupplyChainLink(supplier="DELL", customer="NVDA", revenue_pct=0.20),
            SupplyChainLink(supplier="VRT", customer="NVDA", revenue_pct=0.30),
            # Boeing (BA) Aerospace Ecosystem
            SupplyChainLink(supplier="SPR", customer="BA", revenue_pct=0.72),
            SupplyChainLink(supplier="HXL", customer="BA", revenue_pct=0.32),
            SupplyChainLink(supplier="TGI", customer="BA", revenue_pct=0.38),
            SupplyChainLink(supplier="TDG", customer="BA", revenue_pct=0.28),
            # Amazon (AMZN) Logistics & Cloud
            SupplyChainLink(supplier="ATSG", customer="AMZN", revenue_pct=0.38),
            SupplyChainLink(supplier="AAOI", customer="AMZN", revenue_pct=0.42),
            # Tesla (TSLA) Automotive & Energy
            SupplyChainLink(supplier="ALB", customer="TSLA", revenue_pct=0.22),
            SupplyChainLink(supplier="MGA", customer="TSLA", revenue_pct=0.18),
            # Microsoft (MSFT) Cloud Ecosystem
            SupplyChainLink(supplier="DELL", customer="MSFT", revenue_pct=0.18),
            SupplyChainLink(supplier="VRT", customer="MSFT", revenue_pct=0.25),
        ]
        return SupplyChainNetwork(links=links)

    def graph_convolution_message_passing(
        self,
        node_features: np.ndarray,
        n_layers: int = 2,
        activation: bool = True,
    ) -> np.ndarray:
        """Executes multi-hop Graph Convolutional Network (GCN) layer propagation.

        Formula:
            H^{(l+1)} = ReLU( D~^{-1/2} A~ D~^{-1/2} H^{(l)} W^{(l)} )
        """
        A_norm = self.network.gcn_normalized_adj
        H = node_features.copy()

        for layer in range(n_layers):
            # Graph aggregation
            H = A_norm @ H
            if activation:
                H = np.maximum(H, 0.0)  # ReLU non-linearity

        return H

    def compute_lead_lag_signals(
        self,
        prices_df: pd.DataFrame,
        customer_mom_window: int = 5,
    ) -> pd.DataFrame:
        """Generates customer-to-supplier lead-lag spillover signals across time.

        Calculates customer momentum, propagates shocks via GCN message passing,
        and isolates residual alpha by neutralizing against supplier's own past momentum.

        Args:
            prices_df: DataFrame of daily prices with tickers as columns.
            customer_mom_window: Lookback window W for customer return calculation.

        Returns:
            DataFrame of cross-sectional alpha scores for all universe tickers.
        """
        # Align tickers with network nodes
        universe = [t for t in self.network.nodes if t in prices_df.columns]
        if not universe:
            raise ValueError("None of the network tickers were found in prices_df.")

        p = prices_df[universe]
        returns_1d = p.pct_change().fillna(0.0)
        returns_mom = p.pct_change(customer_mom_window).fillna(0.0)

        # Adjacency matrix restricted to available universe
        A = np.zeros((len(universe), len(universe)))
        for i, sup in enumerate(universe):
            for j, cust in enumerate(universe):
                if sup in self.network.node_to_idx and cust in self.network.node_to_idx:
                    idx_i = self.network.node_to_idx[sup]
                    idx_j = self.network.node_to_idx[cust]
                    A[i, j] = self.network.adjacency_matrix[idx_i, idx_j]

        A_norm = SupplyChainNetwork._compute_gcn_laplacian(A)

        # Vectorized signal computation across time steps
        signals = np.zeros((len(p), len(universe)))

        for t in range(customer_mom_window, len(p)):
            # Step 1: Customer momentum feature vector at t-1 (lagged to prevent lookahead)
            mom_t = returns_mom.iloc[t - 1].values.reshape(-1, 1)

            # Step 2: Direct Customer Impact (Tier-1 Linkage)
            direct_spillover = A @ mom_t

            # Step 3: Multi-Hop GCN Propagation (Tier-1 + Tier-2 Linkages)
            gcn_spillover = A_norm @ (A_norm @ mom_t)

            # Step 4: Combine direct and multi-hop GCN features
            combined_score = 0.65 * direct_spillover.flatten() + 0.35 * gcn_spillover.flatten()

            # Step 5: Orthogonalize / Residualize against supplier's own 5-day momentum
            # Eliminates traditional momentum contamination
            own_mom = mom_t.flatten()
            if np.std(own_mom) > 1e-6:
                # OLS residualization
                cov = np.cov(combined_score, own_mom)[0, 1]
                var = np.var(own_mom)
                beta = cov / var if var > 1e-12 else 0.0
                residual_signal = combined_score - beta * own_mom
            else:
                residual_signal = combined_score

            # Step 6: Cross-Sectional Z-Score Standardization
            sig_std = np.std(residual_signal)
            if sig_std > 1e-6:
                sig_z = (residual_signal - np.mean(residual_signal)) / sig_std
            else:
                sig_z = residual_signal

            signals[t, :] = sig_z

        signals_df = pd.DataFrame(signals, index=p.index, columns=universe)
        return signals_df

    def backtest_strategy(
        self,
        prices_df: pd.DataFrame,
        signals_df: Optional[pd.DataFrame] = None,
        n_quantiles: int = 5,
        risk_free_rate: float = 0.02,
    ) -> GraphAlphaResult:
        """Backtests a dollar-neutral Long/Short Supply-Chain Momentum strategy."""
        if signals_df is None:
            signals_df = self.compute_lead_lag_signals(prices_df, customer_mom_window=self.lead_lag_window)

        universe = [c for c in signals_df.columns if c in prices_df.columns]
        p = prices_df[universe]
        returns_1d = p.pct_change().fillna(0.0)

        # Standalone 12-1M Momentum Benchmark
        mom_bench = p.pct_change(126).shift(1).fillna(0.0)

        n_dates = len(p)
        strat_returns = np.zeros(n_dates)
        standalone_returns = np.zeros(n_dates)
        bench_mkt_returns = returns_1d.mean(axis=1).values
        ic_list = []

        curr_weights = np.zeros(len(universe))
        curr_mom_weights = np.zeros(len(universe))

        for t in range(126, n_dates):
            sig_row = signals_df.iloc[t].values
            mom_row = mom_bench.iloc[t].values

            # Rebalance periodically
            if t % self.rebalance_freq == 0:

                # Long top quintile, short bottom quintile
                ranks = pd.Series(sig_row).rank(pct=True).values
                q_top = ranks >= (1.0 - 1.0 / n_quantiles)
                q_bot = ranks <= (1.0 / n_quantiles)

                target_w = np.zeros(len(universe))
                if np.sum(q_top) > 0:
                    target_w[q_top] = +0.5 / np.sum(q_top)
                if np.sum(q_bot) > 0:
                    target_w[q_bot] = -0.5 / np.sum(q_bot)

                # Standalone momentum weights
                mom_ranks = pd.Series(mom_row).rank(pct=True).values
                mom_q_top = mom_ranks >= (1.0 - 1.0 / n_quantiles)
                mom_q_bot = mom_ranks <= (1.0 / n_quantiles)

                target_mom_w = np.zeros(len(universe))
                if np.sum(mom_q_top) > 0:
                    target_mom_w[mom_q_top] = +0.5 / np.sum(mom_q_top)
                if np.sum(mom_q_bot) > 0:
                    target_mom_w[mom_q_bot] = -0.5 / np.sum(mom_q_bot)

                # Transaction cost deduction on rebalance
                turnover = np.sum(np.abs(target_w - curr_weights))
                cost = turnover * (self.transaction_cost_bps / 10000.0)

                curr_weights = target_w
                curr_mom_weights = target_mom_w
            else:
                cost = 0.0

            # Realized daily return
            ret_t = returns_1d.iloc[t].values
            daily_gross = np.sum(curr_weights * ret_t)
            daily_net = daily_gross - cost
            strat_returns[t] = daily_net

            daily_mom = np.sum(curr_mom_weights * ret_t)
            standalone_returns[t] = daily_mom

            # Daily Information Coefficient (IC)
            if t + 1 < n_dates:
                fwd_ret = returns_1d.iloc[t + 1].values
                if np.std(sig_row) > 1e-6 and np.std(fwd_ret) > 1e-6:
                    ic = np.corrcoef(sig_row, fwd_ret)[0, 1]
                    if not np.isnan(ic):
                        ic_list.append(ic)

        # Performance analytics
        valid_returns = strat_returns[126:]
        valid_mom = standalone_returns[126:]
        valid_dates = p.index[126:]

        cum_equity = np.cumprod(1.0 + valid_returns)
        cum_mom = np.cumprod(1.0 + valid_mom)
        cum_bench = np.cumprod(1.0 + bench_mkt_returns[126:])

        # Annualized metrics
        n_years = len(valid_returns) / 252.0
        cagr = (cum_equity[-1] ** (1.0 / max(0.1, n_years))) - 1.0 if cum_equity[-1] > 0 else -1.0
        ann_vol = np.std(valid_returns) * np.sqrt(252)
        sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 1e-6 else 0.0

        # Downside risk & Sortino
        downside = valid_returns[valid_returns < 0]
        downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (cagr - risk_free_rate) / downside_std if downside_std > 1e-6 else 0.0

        # Max Drawdown
        running_max = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        # IC statistics
        mean_ic = float(np.mean(ic_list)) if ic_list else 0.0
        std_ic = float(np.std(ic_list)) if ic_list else 1.0
        ic_ir = mean_ic / std_ic if std_ic > 1e-6 else 0.0

        # Standalone momentum Sharpe
        mom_cagr = (cum_mom[-1] ** (1.0 / max(0.1, n_years))) - 1.0 if cum_mom[-1] > 0 else -1.0
        mom_vol = np.std(valid_mom) * np.sqrt(252)
        standalone_sharpe = (mom_cagr - risk_free_rate) / mom_vol if mom_vol > 1e-6 else 0.0

        # Correlation between strategy and traditional momentum
        corr_matrix = np.corrcoef(valid_returns, valid_mom)
        corr_to_mom = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        win_rate = float(np.sum(valid_returns > 0) / len(valid_returns))

        metrics = {
            "cagr": cagr,
            "volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "mean_ic": mean_ic,
            "ic_ir": ic_ir,
            "standalone_sharpe": standalone_sharpe,
            "corr_to_momentum": corr_to_mom,
        }

        # Links summary table
        links_df = pd.DataFrame([link.to_dict() for link in self.network.links])

        return GraphAlphaResult(
            signals_df=signals_df,
            centrality_df=self.network.get_centrality_table(),
            equity_curve=pd.Series(cum_equity, index=valid_dates, name="SupplyChain_Alpha"),
            standalone_momentum_curve=pd.Series(cum_mom, index=valid_dates, name="Standalone_Momentum"),
            benchmark_curve=pd.Series(cum_bench, index=valid_dates, name="Equal_Weight_Market"),
            ic_series=pd.Series(ic_list, name="Information_Coefficient"),
            metrics=metrics,
            links_summary=links_df,
        )
