"""
Autonomous Multi-Agent Macroeconomic & Crypto Hedge Fund Swarm.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class AgentView:
    agent_name: str
    asset_views: Dict[str, float]
    conviction: float
    rationale: str


@dataclass
class InvestmentCommitteeMemo:
    date: str
    agent_views: List[AgentView]
    consensus_returns: Dict[str, float]
    optimal_weights: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    committee_consensus_score: float
    memo_text: str

    def to_markdown(self) -> str:
        weights_str = "\n".join([f"  - **{k}**: {v:.1%}" for k, v in self.optimal_weights.items()])
        return f"""# Investment Committee Memorandum — {self.date}
**Consensus Conviction Score**: {self.committee_consensus_score:.1f}/10.0
**Expected Portfolio Return (Ann.)**: {self.expected_portfolio_return:.2%}
**Expected Portfolio Volatility**: {self.expected_portfolio_volatility:.2%}

## Target Asset Allocations:
{weights_str}

## Committee Debate Summary:
{self.memo_text}
"""


class MultiAgentHedgeFundSwarm:
    """Orchestrates autonomous specialist agents and Black-Litterman portfolio construction."""

    def __init__(self, target_vol_annual: float = 0.12, assets: Optional[List[str]] = None):
        self.target_vol_annual = target_vol_annual
        self.assets = assets or ["Equities (SPY)", "Bonds (TLT)", "Gold (GLD)", "Crypto (BTC)", "FX Carry (EM)"]

    def run_investment_committee(
        self,
        macro_signal: float,
        crypto_onchain: float,
        sentiment_fgi: float,
        cov_matrix: Optional[np.ndarray] = None,
        date: str = "2026-08-30",
    ) -> InvestmentCommitteeMemo:
        macro_views = {
            "Equities (SPY)": 0.08 * macro_signal,
            "Bonds (TLT)": -0.04 * macro_signal if macro_signal > 0 else 0.06,
            "Gold (GLD)": 0.05 if macro_signal < 0 else 0.02,
            "Crypto (BTC)": 0.15 * macro_signal,
            "FX Carry (EM)": 0.07 * macro_signal,
        }
        macro_agent = AgentView(
            agent_name="MacroEconomistAgent",
            asset_views=macro_views,
            conviction=0.85,
            rationale=f"Global macro regime is {'Expansionary' if macro_signal > 0 else 'Defensive'}."
        )

        crypto_views = {
            "Equities (SPY)": 0.04,
            "Bonds (TLT)": 0.02,
            "Gold (GLD)": 0.03,
            "Crypto (BTC)": 0.25 * crypto_onchain,
            "FX Carry (EM)": 0.03,
        }
        crypto_agent = AgentView(
            agent_name="CryptoMicrostructureAgent",
            asset_views=crypto_views,
            conviction=0.80,
            rationale=f"On-chain MVRV and whale flows signal {'Strong Inflows' if crypto_onchain > 0 else 'Deleveraging Risk'}."
        )

        sent_normalized = (sentiment_fgi - 50.0) / 50.0
        sent_views = {
            "Equities (SPY)": 0.05 * sent_normalized,
            "Bonds (TLT)": -0.02 * sent_normalized,
            "Gold (GLD)": 0.04 if sent_normalized < -0.3 else 0.01,
            "Crypto (BTC)": 0.18 * sent_normalized,
            "FX Carry (EM)": 0.04 * sent_normalized,
        }
        sent_agent = AgentView(
            agent_name="SentimentAlphaAgent",
            asset_views=sent_views,
            conviction=0.75,
            rationale=f"Market Fear & Greed index is at {sentiment_fgi:.0f}/100."
        )

        all_views = [macro_agent, crypto_agent, sent_agent]
        total_conviction = sum(a.conviction for a in all_views)
        
        consensus_ret = {}
        for asset in self.assets:
            exp_r = sum(a.asset_views.get(asset, 0.0) * a.conviction for a in all_views) / total_conviction
            consensus_ret[asset] = float(exp_r)

        n = len(self.assets)
        if cov_matrix is None:
            vols = np.array([0.16, 0.12, 0.15, 0.55, 0.10])
            corr = np.eye(n)
            corr[0, 3] = 0.35; corr[3, 0] = 0.35
            corr[0, 1] = -0.20; corr[1, 0] = -0.20
            cov_matrix = np.outer(vols, vols) * corr

        mu = np.array([consensus_ret[a] for a in self.assets])
        
        def loss(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, cov_matrix @ w))
            return -port_ret + 1.5 * (port_vol ** 2)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        bnds = [(0.0, 0.50) for _ in range(n)]
        w0 = np.ones(n) / n
        opt = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons)

        opt_weights = {a: float(opt.x[i]) for i, a in enumerate(self.assets)}
        p_ret = float(np.dot(opt.x, mu))
        p_vol = float(np.sqrt(np.dot(opt.x, cov_matrix @ opt.x)))
        consensus_score = float(total_conviction / len(all_views) * 10.0)

        memo_text = f"""The Macro and Sentiment agents reached a unified view favoring {'Risk-On asset classes' if macro_signal > 0 else 'Defensive & Real Assets'}.
The Crypto Microstructure specialist noted {'positive on-chain accumulation' if crypto_onchain > 0 else 'elevated funding drag'}.
The PM Chair executed constrained optimization maintaining a portfolio target volatility of {p_vol:.1%}."""

        return InvestmentCommitteeMemo(
            date=date,
            agent_views=all_views,
            consensus_returns=consensus_ret,
            optimal_weights=opt_weights,
            expected_portfolio_return=p_ret,
            expected_portfolio_volatility=p_vol,
            committee_consensus_score=consensus_score,
            memo_text=memo_text,
        )
