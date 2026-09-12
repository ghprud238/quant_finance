"""
Supply-Chain Knowledge Graph & Graph Neural Network (GNN) Spillover Momentum Alpha.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


class SupplyChainNetwork:
    """Represents a directed customer-supplier economic dependency network."""

    def __init__(self, tickers: List[str], adjacency_matrix: np.ndarray):
        self.tickers = tickers
        self.n_nodes = len(tickers)
        self.adj_matrix = np.array(adjacency_matrix, dtype=float)

    @classmethod
    def create_default_network(cls) -> "SupplyChainNetwork":
        tickers = ["AAPL", "NVDA", "TSMC", "ASML", "MSFT", "AMZN", "AMD", "QCOM", "AVGO", "CRUS"]
        n = len(tickers)
        A = np.zeros((n, n))
        idx = {t: i for i, t in enumerate(tickers)}

        A[idx["CRUS"], idx["AAPL"]] = 0.76
        A[idx["TSMC"], idx["AAPL"]] = 0.25
        A[idx["TSMC"], idx["NVDA"]] = 0.15
        A[idx["ASML"], idx["TSMC"]] = 0.40
        A[idx["NVDA"], idx["MSFT"]] = 0.18
        A[idx["NVDA"], idx["AMZN"]] = 0.14
        A[idx["AMD"], idx["MSFT"]] = 0.12
        A[idx["QCOM"], idx["AAPL"]] = 0.22
        A[idx["AVGO"], idx["AAPL"]] = 0.20
        return cls(tickers, A)

    def get_normalized_adjacency(self) -> np.ndarray:
        A_tilde = self.adj_matrix + np.eye(self.n_nodes)
        D_tilde = np.sum(A_tilde, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_tilde, 1e-8)))
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def get_centrality_table(self) -> pd.DataFrame:
        A = self.adj_matrix
        out_degree = np.sum(A > 0, axis=1)
        in_degree = np.sum(A > 0, axis=0)
        hhi = np.sum(A ** 2, axis=1)
        total_dep = np.sum(A, axis=1)

        # Standard PageRank with dangling node redistribution
        col_sums = np.sum(A, axis=0)
        M = np.zeros_like(A)
        for j in range(self.n_nodes):
            if col_sums[j] > 0:
                M[:, j] = A[:, j] / col_sums[j]
            else:
                M[:, j] = 1.0 / self.n_nodes

        d = 0.85
        p = np.ones(self.n_nodes) / self.n_nodes
        for _ in range(100):
            p = (1.0 - d) / self.n_nodes + d * (M @ p)
            p = p / np.sum(p)

        return pd.DataFrame({
            "Ticker": self.tickers,
            "PageRank_Centrality": p,
            "Supplier_Customer_Count": in_degree,
            "Supplier_Out_Count": out_degree,
            "Customer_Concentration_HHI": hhi,
            "Total_Customer_Revenue_Pct": total_dep,
        }).sort_values("PageRank_Centrality", ascending=False)


@dataclass
class SupplyChainAlphaResult:
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    information_coefficient: float
    ic_information_ratio: float
    equity_curve: pd.Series
    strategy_returns: pd.Series

    def summary(self) -> str:
        return f"""Supply-Chain GNN Alpha Performance:
  - Strategy CAGR:                   {self.cagr:+.2%}
  - Annualized Volatility:           {self.volatility:.2%}
  - Sharpe Ratio (Rf=2%):            {self.sharpe_ratio:.2f}
  - Maximum Drawdown:                {self.max_drawdown:.2%}
  - Mean Information Coefficient:    {self.information_coefficient:+.4f}
  - IC Information Ratio:            {self.ic_information_ratio:.2f}"""


class SupplyChainGraphAlpha:
    """GNN-driven Customer-to-Supplier Lead-Lag Momentum Extraction."""

    def __init__(self, network: Optional[SupplyChainNetwork] = None, lead_lag_window: int = 5):
        self.network = network or SupplyChainNetwork.create_default_network()
        self.lead_lag_window = lead_lag_window

    def compute_graph_momentum_signals(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        aligned_tickers = [t for t in self.network.tickers if t in prices_df.columns]
        P = prices_df[aligned_tickers]
        returns = P.pct_change(self.lead_lag_window)

        A_norm = self.network.get_normalized_adjacency()
        signals = pd.DataFrame(index=returns.index, columns=aligned_tickers, dtype=float)

        for dt in returns.index:
            r_t = returns.loc[dt].fillna(0.0).values
            h1 = A_norm @ r_t
            h2 = A_norm @ np.maximum(0, h1)
            signals.loc[dt] = h2

        mean_sig = signals.mean(axis=1)
        std_sig = signals.std(axis=1).replace(0, 1.0)
        z_signals = signals.sub(mean_sig, axis=0).div(std_sig, axis=0)
        return z_signals

    def backtest_strategy(self, prices_df: pd.DataFrame, n_quantiles: int = 3, tx_cost_bps: float = 5.0) -> SupplyChainAlphaResult:
        signals = self.compute_graph_momentum_signals(prices_df).shift(1)
        aligned_tickers = [t for t in self.network.tickers if t in prices_df.columns]
        daily_returns = prices_df[aligned_tickers].pct_change()

        strat_returns = []
        ics = []

        for dt in daily_returns.index:
            if dt not in signals.index:
                continue
            sig_row = signals.loc[dt].dropna()
            ret_row = daily_returns.loc[dt].dropna()
            common = sig_row.index.intersection(ret_row.index)
            if len(common) < 4:
                continue

            s = sig_row[common]
            r = ret_row[common]

            ic, _ = stats.spearmanr(s, r)
            if not np.isnan(ic):
                ics.append(ic)

            q_high = s.quantile(1.0 - 1.0 / n_quantiles)
            q_low = s.quantile(1.0 / n_quantiles)

            long_mask = s >= q_high
            short_mask = s <= q_low

            w = pd.Series(0.0, index=common)
            if long_mask.sum() > 0:
                w[long_mask] = 0.5 / long_mask.sum()
            if short_mask.sum() > 0:
                w[short_mask] = -0.5 / short_mask.sum()

            gross_ret = (w * r).sum()
            strat_returns.append(gross_ret - tx_cost_bps * 1e-4 * 0.1)

        ret_series = pd.Series(strat_returns)
        cagr = float(np.prod(1.0 + ret_series) ** (252.0 / max(len(ret_series), 1)) - 1.0) if len(ret_series) > 0 else 0.0
        vol = float(ret_series.std() * np.sqrt(252.0)) if len(ret_series) > 0 else 0.0
        sharpe = float((cagr - 0.02) / vol) if vol > 0 else 0.0

        cum = (1.0 + ret_series).cumprod()
        peaks = cum.cummax()
        dd = (cum - peaks) / peaks
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        win_rate = float(np.mean(ret_series > 0)) if len(ret_series) > 0 else 0.0

        mean_ic = float(np.mean(ics)) if ics else 0.0
        ic_ir = float(mean_ic / (np.std(ics) + 1e-8)) if ics else 0.0

        return SupplyChainAlphaResult(
            cagr=cagr,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            information_coefficient=mean_ic,
            ic_information_ratio=ic_ir,
            equity_curve=cum,
            strategy_returns=ret_series,
        )
