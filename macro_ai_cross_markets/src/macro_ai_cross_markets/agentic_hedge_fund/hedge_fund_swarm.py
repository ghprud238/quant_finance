"""Autonomous Multi-Agent Macroeconomic & Crypto Hedge Fund Swarm (Project 50).

Implements a multi-agent quantitative hedge fund committee consisting of:
1. MacroEconomistAgent: Analyzes macro cycles, inflation, yield curve & central bank policies.
2. CryptoMicrostructureAgent: Analyzes on-chain flows, MVRV, funding rates & AMM liquidity.
3. SentimentAlphaAgent: Synthesizes financial news, social conviction & Fear/Greed sentiment.
4. RiskAndExecutionPMAgent: Committee Chair reconciling views via Black-Litterman optimization,
   risk budgeting, VaR constraints, and Investment Committee Memorandum synthesis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


# =========================================================================
# 1. AGENT INPUT DATA STRUCTURES
# =========================================================================

@dataclass
class MacroData:
    """Macroeconomic indicators observed by the MacroEconomistAgent."""
    gdp_growth_pct: float = 2.4              # Real GDP Growth YoY (%)
    cpi_inflation_pct: float = 2.8           # CPI Inflation YoY (%)
    central_bank_rate_pct: float = 4.75      # Policy Rate (%)
    yield_curve_slope_bps: float = 25.0      # 10Y - 2Y Sovereign Spread (bps)
    dxy_index: float = 103.5                 # US Dollar Index
    vix_index: float = 16.5                  # CBOE Volatility Index


@dataclass
class CryptoData:
    """On-chain and microstructure metrics observed by CryptoMicrostructureAgent."""
    btc_price: float = 65000.0               # BTC/USD Spot Price ($)
    eth_price: float = 3200.0                # ETH/USD Spot Price ($)
    mvrv_z_score: float = 1.85               # Bitcoin MVRV Z-Score
    funding_rate_8h_pct: float = 0.015       # 8-Hour Perpetual Funding Rate (%)
    exchange_reserve_flow_usd: float = -1.2e8# 24h Net Exchange Flow ($: - is outflow)
    defi_tvl_change_pct: float = 4.2         # 30d DeFi Total Value Locked Growth (%)


@dataclass
class SentimentData:
    """Cross-asset market sentiment observed by SentimentAlphaAgent."""
    fear_and_greed_index: float = 58.0       # 0 (Extreme Fear) to 100 (Extreme Greed)
    news_sentiment_score: float = 0.35       # -1.0 (Very Negative) to +1.0 (Very Positive)
    social_media_bull_bear_ratio: float = 1.45 # Bulls / Bears ratio
    retail_put_call_ratio: float = 0.78      # CBOE Equity Put/Call Ratio


@dataclass
class AgentView:
    """Subjective asset allocation view formulated by an individual agent."""
    asset: str
    direction: str                           # 'BULLISH', 'BEARISH', 'NEUTRAL'
    expected_return_annual: float            # Annualized expected return (%)
    conviction: float                        # Confidence score in [0.0, 1.0]
    thesis: str                              # Qualitative reasoning
    agent_name: str                          # Name of the authoring agent


# =========================================================================
# 2. SPECIALIZED AGENT PERSONAS
# =========================================================================

class MacroEconomistAgent:
    """Macroeconomic Specialist Agent.
    
    Classifies the macro regime into:
    - DISINFLATIONARY_GROWTH (High Growth, Low Inflation) -> Overweight Equities, Commodities.
    - STAGFLATION (Low Growth, High Inflation) -> Overweight Commodities/Gold, Underweight Equities/Bonds.
    - RECESSION (Negative Growth, Falling Inflation) -> Overweight Long-Duration Sovereign Bonds, Cash.
    - REFLATION (Accelerating Growth, Bottoming Rates) -> Overweight Cyclicals, Equities.
    """
    
    def __init__(self, name: str = "Chief Macroeconomist"):
        self.name = name

    def evaluate(self, data: MacroData) -> Tuple[str, List[AgentView]]:
        views: List[AgentView] = []
        
        # 1. Macro Regime Classification
        is_high_inflation = data.cpi_inflation_pct > 3.2
        is_strong_growth = data.gdp_growth_pct > 2.0
        is_yield_curve_inverted = data.yield_curve_slope_bps < 0.0
        
        if is_strong_growth and not is_high_inflation:
            regime = "DISINFLATIONARY_GROWTH"
            views.append(AgentView(
                asset="Global_Equities",
                direction="BULLISH",
                expected_return_annual=0.14,
                conviction=0.85,
                thesis=f"Solid GDP growth ({data.gdp_growth_pct:.1f}%) with tame inflation ({data.cpi_inflation_pct:.1f}%) provides a supportive backdrop for corporate earnings.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Sovereign_Bonds",
                direction="NEUTRAL",
                expected_return_annual=0.045,
                conviction=0.60,
                thesis="Neutral duration stance as policy rates remain stable with modest term premium.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Commodities",
                direction="BULLISH",
                expected_return_annual=0.10,
                conviction=0.70,
                thesis="Cyclical demand across industrials and energy remains strong under steady economic activity.",
                agent_name=self.name
            ))
        elif not is_strong_growth and is_high_inflation:
            regime = "STAGFLATION"
            views.append(AgentView(
                asset="Global_Equities",
                direction="BEARISH",
                expected_return_annual=-0.06,
                conviction=0.80,
                thesis=f"Input cost inflation ({data.cpi_inflation_pct:.1f}%) alongside decelerating GDP ({data.gdp_growth_pct:.1f}%) pressures gross margins.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Commodities",
                direction="BULLISH",
                expected_return_annual=0.16,
                conviction=0.90,
                thesis="Real assets and energy serve as the primary inflation hedge during supply-driven stagflationary regimes.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Sovereign_Bonds",
                direction="BEARISH",
                expected_return_annual=0.01,
                conviction=0.75,
                thesis="Persistent inflation requires elevated policy rates, suppressing fixed-income capital appreciation.",
                agent_name=self.name
            ))
        elif not is_strong_growth and not is_high_inflation:
            regime = "RECESSION" if is_yield_curve_inverted else "SLOWDOWN"
            views.append(AgentView(
                asset="Sovereign_Bonds",
                direction="BULLISH",
                expected_return_annual=0.09,
                conviction=0.85,
                thesis="Central banks are poised to ease policy rates to stimulate growth, driving long-duration bond capital gains.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Global_Equities",
                direction="BEARISH",
                expected_return_annual=-0.04,
                conviction=0.70,
                thesis="Earnings revisions face headwinds as consumer spending and business capex contract.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Commodities",
                direction="BEARISH",
                expected_return_annual=-0.05,
                conviction=0.75,
                thesis="Aggregate industrial demand slows during macro deceleration.",
                agent_name=self.name
            ))
        else:
            regime = "REFLATION"
            views.append(AgentView(
                asset="Global_Equities",
                direction="BULLISH",
                expected_return_annual=0.12,
                conviction=0.75,
                thesis="Nominal revenue growth benefits from pricing power and high economic velocity.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Commodities",
                direction="BULLISH",
                expected_return_annual=0.14,
                conviction=0.80,
                thesis="Broad-based commodity complex rallies alongside economic expansion.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Sovereign_Bonds",
                direction="BEARISH",
                expected_return_annual=0.02,
                conviction=0.70,
                thesis="Higher terminal rate expectations steepen the sovereign yield curve.",
                agent_name=self.name
            ))
            
        # Cash / FX positioning
        dxy_view = "BULLISH" if data.dxy_index > 105.0 or data.vix_index > 22.0 else "NEUTRAL"
        views.append(AgentView(
            asset="Cash_and_FX",
            direction=dxy_view,
            expected_return_annual=data.central_bank_rate_pct / 100.0,
            conviction=0.80,
            thesis=f"Cash yields {data.central_bank_rate_pct:.2f}% risk-free; provides dry powder for market dislocations.",
            agent_name=self.name
        ))
        
        return regime, views


class CryptoMicrostructureAgent:
    """Crypto & Digital Assets Microstructure Specialist Agent.
    
    Analyzes on-chain flows, valuation metrics (MVRV), perpetual funding rates,
    and exchange inventory dynamics to evaluate the digital asset market state.
    """
    
    def __init__(self, name: str = "Crypto Alpha Specialist"):
        self.name = name

    def evaluate(self, data: CryptoData) -> Tuple[str, List[AgentView]]:
        views: List[AgentView] = []
        
        # 1. Regime Identification
        is_overheated_funding = data.funding_rate_8h_pct > 0.03  # > 30% APR
        is_heavy_exchange_inflow = data.exchange_reserve_flow_usd > 1.0e8
        is_mvrv_undervalued = data.mvrv_z_score < 0.5
        is_mvrv_overheated = data.mvrv_z_score > 3.5
        
        if is_mvrv_overheated or is_overheated_funding:
            regime = "EUPHORIC_OVERHEATED"
            btc_ret = -0.15
            btc_conv = 0.85
            btc_dir = "BEARISH"
            thesis = f"Perpetual funding rate ({data.funding_rate_8h_pct:.3f}% 8h) and MVRV Z-Score ({data.mvrv_z_score:.2f}) signal extreme leverage and high distribution risk."
        elif is_mvrv_undervalued and data.exchange_reserve_flow_usd < 0:
            regime = "ACCUMULATION_BOTTOM"
            btc_ret = 0.45
            btc_conv = 0.90
            btc_dir = "BULLISH"
            thesis = f"MVRV Z-Score ({data.mvrv_z_score:.2f}) in generational value zone with sustained cold-storage exchange outflows (${abs(data.exchange_reserve_flow_usd)/1e6:.1f}M)."
        elif data.mvrv_z_score >= 1.0 and data.defi_tvl_change_pct > 0:
            regime = "HEALTHY_BULL_EXPANSION"
            btc_ret = 0.28
            btc_conv = 0.75
            btc_dir = "BULLISH"
            thesis = f"Moderate MVRV ({data.mvrv_z_score:.2f}) and positive DeFi TVL growth (+{data.defi_tvl_change_pct:.1f}%) indicate organic liquidity expansion without speculative excess."
        else:
            regime = "NEUTRAL_CONSOLIDATION"
            btc_ret = 0.08
            btc_conv = 0.55
            btc_dir = "NEUTRAL"
            thesis = "Range-bound on-chain metrics suggest sideways price consolidation."
            
        views.append(AgentView(
            asset="Crypto_Assets",
            direction=btc_dir,
            expected_return_annual=btc_ret,
            conviction=btc_conv,
            thesis=thesis,
            agent_name=self.name
        ))
        
        return regime, views


class SentimentAlphaAgent:
    """Cross-Asset Sentiment & Behavioral Specialist Agent.
    
    Synthesizes Fear & Greed indices, news sentiment NLP, and retail options skew
    to detect contrarian inflection points vs momentum trends.
    """
    
    def __init__(self, name: str = "Sentiment & Behavioral Strategist"):
        self.name = name

    def evaluate(self, data: SentimentData) -> Tuple[str, List[AgentView]]:
        views: List[AgentView] = []
        
        # Contrarian extremes vs momentum
        is_extreme_fear = data.fear_and_greed_index < 25.0
        is_extreme_greed = data.fear_and_greed_index > 75.0
        is_put_heavy = data.retail_put_call_ratio > 1.10
        is_call_heavy = data.retail_put_call_ratio < 0.60
        
        if is_extreme_fear or is_put_heavy:
            regime = "EXTREME_FEAR_CONTRARIAN_BULLISH"
            views.append(AgentView(
                asset="Global_Equities",
                direction="BULLISH",
                expected_return_annual=0.15,
                conviction=0.80,
                thesis=f"Market sentiment is washed out (Fear & Greed: {data.fear_and_greed_index:.0f}, Put/Call: {data.retail_put_call_ratio:.2f}). Asymmetric upside from mean reversion.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Crypto_Assets",
                direction="BULLISH",
                expected_return_annual=0.35,
                conviction=0.75,
                thesis="Capitulation sentiment historically marks durable local bottoms across high-beta digital assets.",
                agent_name=self.name
            ))
        elif is_extreme_greed or is_call_heavy:
            regime = "EXTREME_GREED_CONTRARIAN_BEARISH"
            views.append(AgentView(
                asset="Global_Equities",
                direction="BEARISH",
                expected_return_annual=-0.05,
                conviction=0.75,
                thesis=f"Complacent sentiment (Fear & Greed: {data.fear_and_greed_index:.0f}, Put/Call: {data.retail_put_call_ratio:.2f}) leaves equity markets vulnerable to negative catalysts.",
                agent_name=self.name
            ))
            views.append(AgentView(
                asset="Crypto_Assets",
                direction="BEARISH",
                expected_return_annual=-0.12,
                conviction=0.80,
                thesis="Frothy retail sentiment and high call volume indicate an impending leverage flush.",
                agent_name=self.name
            ))
        else:
            regime = "NEUTRAL_TREND_FOLLOWING"
            trend_dir = "BULLISH" if data.news_sentiment_score > 0.15 else ("BEARISH" if data.news_sentiment_score < -0.15 else "NEUTRAL")
            views.append(AgentView(
                asset="Global_Equities",
                direction=trend_dir,
                expected_return_annual=0.08 * (1.0 if trend_dir == "BULLISH" else (-1.0 if trend_dir == "BEARISH" else 0.5)),
                conviction=0.60,
                thesis=f"Neutral sentiment (News score: {data.news_sentiment_score:+.2f}). Positioning in accordance with baseline fundamental momentum.",
                agent_name=self.name
            ))
            
        return regime, views


# =========================================================================
# 3. INVESTMENT COMMITTEE MEMORANDUM DATA STRUCTURE
# =========================================================================

@dataclass
class InvestmentCommitteeMemo:
    """Structured Investment Committee Memorandum produced by the Swarm."""
    date: str
    macro_regime: str
    crypto_regime: str
    sentiment_regime: str
    executive_summary: str
    agent_views: List[AgentView]
    recommended_weights: Dict[str, float]
    posterior_expected_returns: Dict[str, float]
    risk_metrics: Dict[str, float]
    dissenting_debates: List[str]
    committee_approval: str = "UNANIMOUS"

    def to_dataframe(self) -> pd.DataFrame:
        """Formats allocation and views into a clean tabular DataFrame."""
        records = []
        for asset, weight in self.recommended_weights.items():
            exp_ret = self.posterior_expected_returns.get(asset, 0.0)
            records.append({
                "Asset_Class": asset,
                "Target_Weight": f"{weight:.2%}",
                "Weight_Numeric": weight,
                "Posterior_Return": f"{exp_ret:+.2%}",
                "Return_Numeric": exp_ret,
            })
        return pd.DataFrame(records)

    def to_markdown(self) -> str:
        """Generates a comprehensive executive memo in Markdown format."""
        lines = [
            f"# 🏛️ INVESTMENT COMMITTEE MEMORANDUM — {self.date}",
            f"**Status**: {self.committee_approval} | **Macro**: `{self.macro_regime}` | **Crypto**: `{self.crypto_regime}` | **Sentiment**: `{self.sentiment_regime}`",
            "",
            "## 1. Executive Summary",
            self.executive_summary,
            "",
            "## 2. Specialized Agent Deliberations & Convictions",
        ]
        
        for view in self.agent_views:
            icon = "🟢" if view.direction == "BULLISH" else ("🔴" if view.direction == "BEARISH" else "⚪")
            lines.append(f"- **{view.agent_name}** on `{view.asset}`: {icon} **{view.direction}** (Conviction: {view.conviction:.0%}, Exp Return: {view.expected_return_annual:+.1%})")
            lines.append(f"  *Thesis*: {view.thesis}")
            
        lines.extend([
            "",
            "## 3. Black-Litterman Reconciled Asset Allocation",
            "| Asset Class | Target Weight | Posterior Expected Return |",
            "| :--- | :---: | :---: |",
        ])
        for asset, w in self.recommended_weights.items():
            r = self.posterior_expected_returns.get(asset, 0.0)
            lines.append(f"| **{asset}** | `{w:.2%}` | `{r:+.2%}` |")
            
        lines.extend([
            "",
            "## 4. Portfolio Risk Diagnostics & Limits",
            f"- **Target Annualized Volatility**: `{self.risk_metrics.get('portfolio_volatility_ann', 0.0):.2%}`",
            f"- **95% Daily Parametric VaR**: `{self.risk_metrics.get('var_95_daily', 0.0):.2%}`",
            f"- **99% Daily Parametric CVaR (ES)**: `{self.risk_metrics.get('cvar_99_daily', 0.0):.2%}`",
            f"- **Effective Number of Assets (1/HHI)**: `{self.risk_metrics.get('effective_n_assets', 0.0):.2f}`",
            "",
            "## 5. Committee Debates & Dissenting Opinions",
        ])
        for debate in self.dissenting_debates:
            lines.append(f"- ⚠️ {debate}")
            
        return chr(10).join(lines)


# =========================================================================
# 4. RISK & EXECUTION PM AGENT (COMMITTEE CHAIR)
# =========================================================================

class RiskAndExecutionPMAgent:
    """Portfolio Manager & Investment Committee Chair.
    
    Reconciles multi-agent views using the Black-Litterman (1992) model,
    enforces institutional risk bounds, and solves constrained optimization.
    """
    
    DEFAULT_ASSETS = [
        "Global_Equities",
        "Sovereign_Bonds",
        "Commodities",
        "Crypto_Assets",
        "Cash_and_FX"
    ]
    
    DEFAULT_BENCHMARK_WEIGHTS = {
        "Global_Equities": 0.40,
        "Sovereign_Bonds": 0.30,
        "Commodities": 0.15,
        "Crypto_Assets": 0.05,
        "Cash_and_FX": 0.10,
    }
    
    DEFAULT_MAX_WEIGHTS = {
        "Global_Equities": 0.50,
        "Sovereign_Bonds": 0.45,
        "Commodities": 0.30,
        "Crypto_Assets": 0.20,
        "Cash_and_FX": 0.40,
    }

    def __init__(
        self,
        name: str = "Portfolio Manager (Committee Chair)",
        risk_aversion_delta: float = 2.5,
        tau: float = 0.05,
        target_vol_annual: float = 0.10,
    ):
        self.name = name
        self.delta = risk_aversion_delta
        self.tau = tau
        self.target_vol = target_vol_annual

    def reconcile_and_optimize(
        self,
        agent_views: List[AgentView],
        cov_matrix: Optional[pd.DataFrame] = None,
        date: str = "2026-08-25",
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[str]]:
        """Blends agent views via Black-Litterman and executes constrained optimization."""
        assets = self.DEFAULT_ASSETS
        n_assets = len(assets)
        
        # 1. Base Covariance Matrix (if not provided)
        if cov_matrix is None or cov_matrix.empty:
            vols = np.array([0.16, 0.08, 0.20, 0.60, 0.03])
            corr = np.array([
                [ 1.00, -0.20,  0.30,  0.40, -0.10],
                [-0.20,  1.00, -0.10, -0.15,  0.20],
                [ 0.30, -0.10,  1.00,  0.25, -0.20],
                [ 0.40, -0.15,  0.25,  1.00, -0.15],
                [-0.10,  0.20, -0.20, -0.15,  1.00],
            ])
            sigma = np.outer(vols, vols) * corr
            cov_df = pd.DataFrame(sigma, index=assets, columns=assets)
        else:
            cov_df = cov_matrix.loc[assets, assets]
            sigma = cov_df.values
            
        w_mkt = np.array([self.DEFAULT_BENCHMARK_WEIGHTS[a] for a in assets])
        
        # 2. Black-Litterman Prior Equilibrium Returns: Pi = delta * Sigma * w_mkt
        pi = self.delta * (sigma @ w_mkt)
        
        # 3. Aggregate Agent Views into Pick Matrix P and View Vector q
        views_dict: Dict[str, List[AgentView]] = {a: [] for a in assets}
        for v in agent_views:
            if v.asset in views_dict:
                views_dict[v.asset].append(v)
                
        active_views = []
        dissenting_debates = []
        
        for asset, v_list in views_dict.items():
            if not v_list:
                continue
            directions = [v.direction for v in v_list]
            if "BULLISH" in directions and "BEARISH" in directions:
                dissenting_debates.append(
                    f"Conflict on {asset}: {v_list[0].agent_name} ({v_list[0].direction}) vs {v_list[1].agent_name} ({v_list[1].direction}). Reconciled via conviction-weighted blending."
                )
                
            total_conv = sum(v.conviction for v in v_list)
            avg_return = sum(v.expected_return_annual * v.conviction for v in v_list) / max(1e-5, total_conv)
            avg_conv = max(0.10, min(0.99, total_conv / len(v_list)))
            active_views.append((asset, avg_return, avg_conv))
            
        k_views = len(active_views)
        if k_views > 0:
            P = np.zeros((k_views, n_assets))
            q = np.zeros(k_views)
            omega_diag = np.zeros(k_views)
            
            for i, (asset, ret_val, conv_val) in enumerate(active_views):
                idx = assets.index(asset)
                P[i, idx] = 1.0
                q[i] = ret_val
                
                # Proportional uncertainty scaled by inverse conviction
                view_var = self.tau * (P[i] @ sigma @ P[i].T)
                omega_diag[i] = max(1e-6, view_var / (conv_val ** 2))
                
            Omega = np.diag(omega_diag)
            
            sigma_reg = sigma + 1e-5 * np.eye(len(sigma))
            inv_tau_sigma = np.linalg.pinv(self.tau * sigma_reg)
            inv_omega = np.linalg.pinv(Omega + 1e-5 * np.eye(len(Omega)))
            
            M = np.linalg.pinv(inv_tau_sigma + P.T @ inv_omega @ P)
            mu_bl = M @ (inv_tau_sigma @ pi + P.T @ inv_omega @ q)
            sigma_bl = sigma + M
        else:
            mu_bl = pi
            sigma_bl = sigma
            
        # 4. Constrained Portfolio Optimization: max w^T mu_BL - gamma/2 w^T sigma_BL w
        def objective(w):
            return -(w @ mu_bl - 0.5 * self.delta * (w @ sigma_bl @ w))
            
        bounds = [(0.0, self.DEFAULT_MAX_WEIGHTS[a]) for a in assets]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        
        w_init = w_mkt.copy()
        res = minimize(objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if res.success:
            opt_w = res.x
        else:
            opt_w = w_mkt
            
        opt_w = opt_w / np.sum(opt_w)
        
        # 5. Risk Diagnostics
        port_var = float(opt_w @ sigma_bl @ opt_w)
        port_vol_ann = float(np.sqrt(port_var))
        var_95_daily = float(norm.ppf(0.95) * (port_vol_ann / np.sqrt(252)))
        cvar_99_daily = float(norm.pdf(norm.ppf(0.99)) / 0.01 * (port_vol_ann / np.sqrt(252)))
        hhi = float(np.sum(opt_w ** 2))
        eff_n = float(1.0 / hhi) if hhi > 0 else float(n_assets)
        
        weights_dict = {assets[i]: float(opt_w[i]) for i in range(n_assets)}
        posterior_returns_dict = {assets[i]: float(mu_bl[i]) for i in range(n_assets)}
        risk_metrics = {
            "portfolio_volatility_ann": port_vol_ann,
            "var_95_daily": var_95_daily,
            "cvar_99_daily": cvar_99_daily,
            "herfindahl_index": hhi,
            "effective_n_assets": eff_n,
        }
        
        if not dissenting_debates:
            dissenting_debates.append("Unanimous consensus across all agent sub-committees; no conflicting allocations detected.")
            
        return weights_dict, posterior_returns_dict, risk_metrics, dissenting_debates


# =========================================================================
# 5. MULTI-AGENT HEDGE FUND SWARM & BACKTEST ENGINE
# =========================================================================

@dataclass
class SwarmBacktestResult:
    """Multi-asset backtesting result produced by the Autonomous Swarm."""
    dates: pd.DatetimeIndex
    equity_curve: pd.Series
    weights_df: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    metrics: Dict[str, Any]
    regimes_df: pd.DataFrame

    def summary_table(self) -> pd.DataFrame:
        """Returns key institutional performance metrics."""
        records = []
        for k, v in self.metrics.items():
            if isinstance(v, float):
                if "CAGR" in k or "Return" in k or "Volatility" in k or "Drawdown" in k or "Rate" in k:
                    val_str = f"{v:+.2%}" if "Return" in k or "CAGR" in k else f"{v:.2%}"
                else:
                    val_str = f"{v:.2f}"
            else:
                val_str = str(v)
            records.append({"Metric": k, "Value": val_str})
        return pd.DataFrame(records)


class MultiAgentHedgeFundSwarm:
    """Autonomous Multi-Agent Macroeconomic & Crypto Hedge Fund Swarm (Project 50).
    
    Coordinates the continuous interaction and portfolio construction across:
    - MacroEconomistAgent
    - CryptoMicrostructureAgent
    - SentimentAlphaAgent
    - RiskAndExecutionPMAgent
    """
    
    def __init__(
        self,
        risk_aversion_delta: float = 2.5,
        target_vol_annual: float = 0.10,
        rebalance_cost_bps: float = 5.0,
    ):
        self.macro_agent = MacroEconomistAgent()
        self.crypto_agent = CryptoMicrostructureAgent()
        self.sentiment_agent = SentimentAlphaAgent()
        self.pm_agent = RiskAndExecutionPMAgent(
            risk_aversion_delta=risk_aversion_delta,
            target_vol_annual=target_vol_annual,
        )
        self.cost_bps = rebalance_cost_bps

    def conduct_investment_committee(
        self,
        macro_data: MacroData,
        crypto_data: CryptoData,
        sentiment_data: SentimentData,
        cov_matrix: Optional[pd.DataFrame] = None,
        date: str = "2026-08-25",
    ) -> InvestmentCommitteeMemo:
        """Simulates an autonomous Investment Committee session and produces a memorandum."""
        # 1. Solicit views from specialized agents
        macro_regime, macro_views = self.macro_agent.evaluate(macro_data)
        crypto_regime, crypto_views = self.crypto_agent.evaluate(crypto_data)
        sentiment_regime, sentiment_views = self.sentiment_agent.evaluate(sentiment_data)
        
        all_views = macro_views + crypto_views + sentiment_views
        
        # 2. PM Agent reconciliation & Black-Litterman optimization
        weights, post_returns, risk_metrics, debates = self.pm_agent.reconcile_and_optimize(
            agent_views=all_views,
            cov_matrix=cov_matrix,
            date=date,
        )
        
        # 3. Executive Summary Synthesis
        exec_summary = (
            f"The Investment Committee convened on {date} under a `{macro_regime}` macro environment, "
            f"`{crypto_regime}` digital asset conditions, and `{sentiment_regime}` sentiment. "
            f"The Committee approved a target allocation of {weights.get('Global_Equities', 0):.1%} Global Equities, "
            f"{weights.get('Sovereign_Bonds', 0):.1%} Sovereign Bonds, {weights.get('Commodities', 0):.1%} Commodities, "
            f"{weights.get('Crypto_Assets', 0):.1%} Digital Assets, and {weights.get('Cash_and_FX', 0):.1%} Cash. "
            f"The estimated portfolio volatility is {risk_metrics['portfolio_volatility_ann']:.2%} with a 95% Daily VaR of {risk_metrics['var_95_daily']:.2%}."
        )
        
        approval_status = "MAJORITY_CONSENSUS" if len(debates) > 1 and "Conflict" in debates[0] else "UNANIMOUS_CONSENSUS"
        
        return InvestmentCommitteeMemo(
            date=date,
            macro_regime=macro_regime,
            crypto_regime=crypto_regime,
            sentiment_regime=sentiment_regime,
            executive_summary=exec_summary,
            agent_views=all_views,
            recommended_weights=weights,
            posterior_expected_returns=post_returns,
            risk_metrics=risk_metrics,
            dissenting_debates=debates,
            committee_approval=approval_status,
        )

    def backtest(
        self,
        multi_asset_returns_df: pd.DataFrame,
        macro_series_df: Optional[pd.DataFrame] = None,
        crypto_series_df: Optional[pd.DataFrame] = None,
        sentiment_series_df: Optional[pd.DataFrame] = None,
        rebalance_freq_days: int = 21,
        initial_capital: float = 10_000_000.0,
    ) -> SwarmBacktestResult:
        """Runs multi-year backtest of the autonomous swarm across historical market cycles."""
        dates = multi_asset_returns_df.index
        n_days = len(dates)
        assets = self.pm_agent.DEFAULT_ASSETS
        
        # Ensure returns dataframe has standard asset columns
        returns_clean = multi_asset_returns_df.copy()
        for a in assets:
            if a not in returns_clean.columns:
                returns_clean[a] = 0.0
        returns_clean = returns_clean[assets]
        
        # Generate default synthetic macro/crypto/sentiment series if not passed
        if macro_series_df is None:
            macro_series_df = pd.DataFrame({
                "gdp_growth_pct": np.linspace(2.8, 1.5, n_days) + np.random.normal(0, 0.3, n_days),
                "cpi_inflation_pct": np.linspace(3.5, 2.2, n_days) + np.random.normal(0, 0.2, n_days),
                "central_bank_rate_pct": np.full(n_days, 4.5),
                "yield_curve_slope_bps": np.linspace(-30, 40, n_days),
                "dxy_index": np.full(n_days, 103.0),
                "vix_index": np.full(n_days, 16.0),
            }, index=dates)
            
        if crypto_series_df is None:
            crypto_series_df = pd.DataFrame({
                "btc_price": 60000.0 * np.cumprod(1.0 + returns_clean["Crypto_Assets"]),
                "eth_price": 3000.0 * np.cumprod(1.0 + returns_clean["Crypto_Assets"]),
                "mvrv_z_score": np.clip(1.5 + np.cumsum(returns_clean["Crypto_Assets"]), 0.2, 4.5),
                "funding_rate_8h_pct": np.full(n_days, 0.01),
                "exchange_reserve_flow_usd": np.full(n_days, -5e7),
                "defi_tvl_change_pct": np.full(n_days, 2.5),
            }, index=dates)
            
        if sentiment_series_df is None:
            sentiment_series_df = pd.DataFrame({
                "fear_and_greed_index": np.full(n_days, 55.0),
                "news_sentiment_score": np.full(n_days, 0.20),
                "social_media_bull_bear_ratio": np.full(n_days, 1.3),
                "retail_put_call_ratio": np.full(n_days, 0.85),
            }, index=dates)
            
        daily_weights = np.zeros((n_days, len(assets)))
        gross_pnl = np.zeros(n_days)
        net_pnl = np.zeros(n_days)
        regimes_list = []
        
        current_weights = np.array([self.pm_agent.DEFAULT_BENCHMARK_WEIGHTS[a] for a in assets])
        
        for t in range(n_days):
            current_date = dates[t]
            current_date_str = str(current_date)[:10]
            
            # Rebalance on schedule
            if t % rebalance_freq_days == 0:
                lookback_ret = returns_clean.iloc[max(0, t-60):t] if t >= 20 else returns_clean.iloc[:60]
                cov_roll = lookback_ret.cov() * 252.0 if len(lookback_ret) > 10 else None
                
                m_row = macro_series_df.iloc[t]
                c_row = crypto_series_df.iloc[t]
                s_row = sentiment_series_df.iloc[t]
                
                m_data = MacroData(
                    gdp_growth_pct=float(m_row["gdp_growth_pct"]),
                    cpi_inflation_pct=float(m_row["cpi_inflation_pct"]),
                    central_bank_rate_pct=float(m_row["central_bank_rate_pct"]),
                    yield_curve_slope_bps=float(m_row["yield_curve_slope_bps"]),
                    dxy_index=float(m_row["dxy_index"]),
                    vix_index=float(m_row["vix_index"]),
                )
                c_data = CryptoData(
                    btc_price=float(c_row["btc_price"]),
                    eth_price=float(c_row["eth_price"]),
                    mvrv_z_score=float(c_row["mvrv_z_score"]),
                    funding_rate_8h_pct=float(c_row["funding_rate_8h_pct"]),
                    exchange_reserve_flow_usd=float(c_row["exchange_reserve_flow_usd"]),
                    defi_tvl_change_pct=float(c_row["defi_tvl_change_pct"]),
                )
                s_data = SentimentData(
                    fear_and_greed_index=float(s_row["fear_and_greed_index"]),
                    news_sentiment_score=float(s_row["news_sentiment_score"]),
                    social_media_bull_bear_ratio=float(s_row["social_media_bull_bear_ratio"]),
                    retail_put_call_ratio=float(s_row["retail_put_call_ratio"]),
                )
                
                memo = self.conduct_investment_committee(
                    macro_data=m_data,
                    crypto_data=c_data,
                    sentiment_data=s_data,
                    cov_matrix=cov_roll,
                    date=current_date_str,
                )
                
                new_weights = np.array([memo.recommended_weights[a] for a in assets])
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost_drag = turnover * (self.cost_bps / 10000.0)
                
                current_weights = new_weights
                regimes_list.append({
                    "Date": current_date,
                    "Macro": memo.macro_regime,
                    "Crypto": memo.crypto_regime,
                    "Sentiment": memo.sentiment_regime,
                    "Approval": memo.committee_approval,
                })
            else:
                cost_drag = 0.0
                
            daily_weights[t] = current_weights
            ret_t = returns_clean.iloc[t].values
            
            day_gross = float(np.dot(current_weights, ret_t))
            gross_pnl[t] = day_gross
            net_pnl[t] = day_gross - cost_drag
            
        equity_curve = initial_capital * np.cumprod(1.0 + net_pnl)
        equity_series = pd.Series(equity_curve, index=dates)
        gross_series = pd.Series(gross_pnl, index=dates)
        net_series = pd.Series(net_pnl, index=dates)
        weights_df = pd.DataFrame(daily_weights, index=dates, columns=assets)
        regimes_df = pd.DataFrame(regimes_list)
        
        total_return = float(equity_curve[-1] / initial_capital - 1.0)
        years = n_days / 252.0
        cagr = float((equity_curve[-1] / initial_capital) ** (1.0 / max(0.1, years)) - 1.0)
        vol_ann = float(net_series.std() * np.sqrt(252.0))
        sharpe = float((cagr - 0.02) / max(1e-4, vol_ann))
        
        downside_std = float(net_series[net_series < 0].std() * np.sqrt(252.0))
        sortino = float((cagr - 0.02) / max(1e-4, downside_std))
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 10.0
        win_rate = float(np.mean(net_pnl > 0))
        
        metrics = {
            "Initial_Capital": initial_capital,
            "Final_Portfolio_Equity": float(equity_curve[-1]),
            "Total_Return": total_return,
            "CAGR": cagr,
            "Annualized_Volatility": vol_ann,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino,
            "Calmar_Ratio": calmar,
            "Max_Drawdown": max_dd,
            "Daily_Win_Rate": win_rate,
            "Total_Trading_Days": n_days,
        }
        
        return SwarmBacktestResult(
            dates=dates,
            equity_curve=equity_series,
            weights_df=weights_df,
            gross_returns=gross_series,
            net_returns=net_series,
            metrics=metrics,
            regimes_df=regimes_df,
        )
