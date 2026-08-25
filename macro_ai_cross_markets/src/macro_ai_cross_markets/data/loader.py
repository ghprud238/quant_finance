"""Comprehensive Data Loader & Simulation Suite for Global Macro AI & Sovereign Risk (46-47).

Provides generators and datasets for:
1. Multilingual Central Bank Policy Statements (Fed, ECB, BOJ, RBI, BCB, Banxico, PBOC).
2. Global Macro Market Data (10Y Sovereign Yields, 5Y CDS Spreads, CPI, Policy Rates, GDP).
3. Financial News and Market Headlines Stream with Sentiment.
4. G10 and EM FX Rates, Yield Differentials, Risk Reversals, and Volatility Surfaces.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# =========================================================================
# 1. CENTRAL BANK MONETARY POLICY STATEMENTS (Module 46)
# =========================================================================

@dataclass
class PolicyStatement:
    """Represents a central bank rate decision statement and press release."""
    central_bank: str
    date: str
    policy_rate: float
    rate_decision: str  # 'HIKE', 'CUT', 'HOLD'
    statement_text: str
    official_language: str
    english_translation: str


def generate_central_bank_statements(seed: int = 42) -> List[PolicyStatement]:
    """Generates historical and synthetic monetary policy communications across 7 major central banks."""
    statements = [
        # Federal Reserve (FOMC)
        PolicyStatement(
            central_bank="FED",
            date="2022-06-15",
            policy_rate=1.75,
            rate_decision="HIKE",
            statement_text=(
                "The Federal Open Market Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. "
                "In support of these goals, the Committee decided to raise the target range for the federal funds rate to 1.50 to 1.75 percent and anticipates that ongoing increases in the target range will be appropriate. "
                "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher energy prices, and broader price pressures. "
                "The Committee is strongly committed to returning inflation to its 2 percent objective. "
                "In addition, the Committee will continue reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities."
            ),
            official_language="en",
            english_translation=(
                "The Federal Open Market Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. "
                "In support of these goals, the Committee decided to raise the target range for the federal funds rate to 1.50 to 1.75 percent and anticipates that ongoing increases in the target range will be appropriate. "
                "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher energy prices, and broader price pressures. "
                "The Committee is strongly committed to returning inflation to its 2 percent objective."
            ),
        ),
        PolicyStatement(
            central_bank="FED",
            date="2023-07-26",
            policy_rate=5.50,
            rate_decision="HIKE",
            statement_text=(
                "Recent indicators suggest that economic activity has been expanding at a moderate pace. Job gains have been robust in recent months, and the unemployment rate has remained low. "
                "Inflation remains elevated. The Committee decided to raise the target range for the federal funds rate to 5.25 to 5.50 percent. "
                "In determining the extent of additional policy firming that may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening of monetary policy, "
                "the lags with which monetary policy affects economic activity and inflation, and economic and financial developments."
            ),
            official_language="en",
            english_translation=(
                "Recent indicators suggest that economic activity has been expanding at a moderate pace. Job gains have been robust, and the unemployment rate has remained low. "
                "Inflation remains elevated. The Committee decided to raise the target range for the federal funds rate to 5.25 to 5.50 percent. "
                "The Committee will take into account cumulative tightening and monetary policy transmission lags."
            ),
        ),
        PolicyStatement(
            central_bank="FED",
            date="2024-09-18",
            policy_rate=5.00,
            rate_decision="CUT",
            statement_text=(
                "Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have slowed, and the unemployment rate has moved up but remains low. "
                "Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat elevated. "
                "In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 50 basis points to 4.75 to 5.00 percent. "
                "The Committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective."
            ),
            official_language="en",
            english_translation=(
                "Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have slowed, and the unemployment rate has moved up. "
                "Inflation has made further progress toward the 2 percent objective. The Committee decided to lower the target range by 50 basis points. "
                "The Committee is strongly committed to supporting maximum employment and price stability."
            ),
        ),

        # European Central Bank (ECB)
        PolicyStatement(
            central_bank="ECB",
            date="2022-09-08",
            policy_rate=1.25,
            rate_decision="HIKE",
            statement_text=(
                "The Governing Council today decided to raise the three key ECB interest rates by 75 basis points. "
                "This major step frontloads the transition from the prevailing highly accommodative level of policy rates towards levels that will ensure the timely return of inflation to the ECB's 2% medium-term target. "
                "Inflation remains far too high and is likely to stay above target for an extended period. Price pressures have continued to strengthen and broaden across the economy."
            ),
            official_language="en",
            english_translation=(
                "The Governing Council decided to raise key interest rates by 75 basis points. "
                "This major step frontloads the transition towards levels ensuring a timely return of inflation to 2%. "
                "Inflation remains far too high and price pressures have broadened across the economy."
            ),
        ),
        PolicyStatement(
            central_bank="ECB",
            date="2024-06-06",
            policy_rate=3.75,
            rate_decision="CUT",
            statement_text=(
                "Based on an updated assessment of the inflation outlook, the dynamics of underlying inflation and the strength of monetary policy transmission, "
                "it is now appropriate to moderate the degree of monetary policy restriction after nine months of holding rates steady. "
                "The Governing Council decided to lower the three key ECB interest rates by 25 basis points."
            ),
            official_language="en",
            english_translation=(
                "Based on an updated assessment of the inflation outlook and transmission strength, it is appropriate to moderate monetary policy restriction. "
                "The Governing Council decided to lower key interest rates by 25 basis points."
            ),
        ),

        # Bank of Japan (BOJ)
        PolicyStatement(
            central_bank="BOJ",
            date="2023-01-18",
            policy_rate=-0.10,
            rate_decision="HOLD",
            statement_text=(
                "当委員会は、2％の「物価安定の目標」の持続的・安定的な実現に向けて、長短金利操作付き量的・質的金融緩和（YCC）を継続することを全員一致で決定した。"
                "イールドカーブ・コントロールのもとで、短期政策金利をマイナス0.1％とし、10年物国債金利がゼロ％程度で推移するよう上限を設けず長期国債の買入れを行う。"
                "賃金の上昇を伴う形で物価安定の目標が達成されるまで、強力な金融緩和を粘り強く継続する。"
            ),
            official_language="ja",
            english_translation=(
                "The Policy Board decided unanimously to maintain Quantitative and Qualitative Monetary Easing with Yield Curve Control (YCC). "
                "The Bank will apply a negative interest rate of -0.1 percent to policy balances and purchase JGBs to maintain 10-year yields around zero percent. "
                "The Bank will patiently continue with monetary easing accompanied by wage increases to sustainably achieve the 2 percent price stability target."
            ),
        ),
        PolicyStatement(
            central_bank="BOJ",
            date="2024-03-19",
            policy_rate=0.10,
            rate_decision="HIKE",
            statement_text=(
                "日本銀行は、マイナス金利政策およびイールドカーブ・コントロール（YCC）の枠組みを終了し、無担保コールレート（オーバーナイト物）を0〜0.1％程度で推移するよう促すことを決定した。"
                "賃金と物価の好循環の強まりが確認され、2％の物価安定目標が持続的・安定的に達成されていくことが見通せる状況に至ったと判断した。"
            ),
            official_language="ja",
            english_translation=(
                "The Bank of Japan decided to end the negative interest rate policy and Yield Curve Control (YCC) framework, guiding the uncollateralized overnight call rate to around 0 to 0.1 percent. "
                "The Bank judged that the virtuous cycle between wages and prices has strengthened and the sustainable achievement of the 2 percent target is in sight."
            ),
        ),

        # Reserve Bank of India (RBI)
        PolicyStatement(
            central_bank="RBI",
            date="2022-08-05",
            policy_rate=5.40,
            rate_decision="HIKE",
            statement_text=(
                "The Monetary Policy Committee (MPC) decided by a majority of 5 out of 6 members to increase the policy repo rate under the liquidity adjustment facility (LAF) by 50 basis points to 5.40 per cent with immediate effect. "
                "The MPC also decided to remain focused on withdrawal of accommodation to ensure that inflation remains within the target going forward, while supporting growth. "
                "Inflation is projected to remain above the upper tolerance band of 6 per cent for the first three quarters of 2022-23."
            ),
            official_language="en",
            english_translation=(
                "The Monetary Policy Committee decided to increase the repo rate by 50 basis points to 5.40 per cent. "
                "The MPC remains focused on withdrawal of accommodation to ensure inflation aligns with the target while supporting growth. "
                "Headline inflation remains elevated above the upper tolerance band."
            ),
        ),
        PolicyStatement(
            central_bank="RBI",
            date="2023-10-06",
            policy_rate=6.50,
            rate_decision="HOLD",
            statement_text=(
                "The Monetary Policy Committee decided unanimously to keep the policy repo rate unchanged at 6.50 per cent. "
                "The MPC also decided by a majority of 5 out of 6 members to remain focused on withdrawal of accommodation to ensure that inflation progressively aligns to the target, while supporting growth. "
                "Headline inflation has moderated from its July peak, but food price shocks require ongoing vigilance."
            ),
            official_language="en",
            english_translation=(
                "The Monetary Policy Committee decided unanimously to keep the repo rate unchanged at 6.50 per cent. "
                "The MPC remains vigilant against recurring food price shocks and committed to aligning inflation with the 4 percent target."
            ),
        ),

        # Banco Central do Brasil (BCB - Copom)
        PolicyStatement(
            central_bank="BCB",
            date="2022-05-04",
            policy_rate=12.75,
            rate_decision="HIKE",
            statement_text=(
                "O Comitê de Política Monetária (Copom) decidiu, por unanimidade, elevar a taxa Selic em 1,00 ponto percentual, para 12,75% a.a. "
                "O ambiente externo continuou se deteriorando, com pressões inflacionárias decorrentes de gargalos nas cadeias de suprimento globais e choque nos preços de commodities. "
                "O Copom considera que o momento exige vigilância redobrada e perseverança na estratégia contracionista para ancorar as expectativas de inflação em torno das metas."
            ),
            official_language="pt",
            english_translation=(
                "The Monetary Policy Committee (Copom) unanimously decided to raise the Selic rate by 1.00 percentage point to 12.75 percent. "
                "The external environment deteriorated with global supply chain bottlenecks and commodity shocks. "
                "The Committee emphasizes heightened vigilance and perseverance in contractionary policy to anchor inflation expectations."
            ),
        ),
        PolicyStatement(
            central_bank="BCB",
            date="2023-08-02",
            policy_rate=13.25,
            rate_decision="CUT",
            statement_text=(
                "O Copom decidiu reduzir a taxa Selic em 0,50 ponto percentual, para 13,25% a.a. "
                "A melhora do quadro inflacionário, refletindo em parte os impactos acumulados da política monetária, permitiu o início de um ciclo gradual de flexibilização monetária. "
                "O Comitê antevê novas reduções de mesma magnitude nas próximas reuniões caso o cenário evolua conforme o esperado."
            ),
            official_language="pt",
            english_translation=(
                "The Copom decided to lower the Selic rate by 0.50 percentage points to 13.25 percent. "
                "The improvement in inflation dynamics, reflecting cumulative monetary tightening, allows the start of a gradual easing cycle. "
                "The Committee anticipates further rate reductions of the same magnitude."
            ),
        ),

        # Banco de México (Banxico)
        PolicyStatement(
            central_bank="BANXICO",
            date="2022-12-15",
            policy_rate=10.50,
            rate_decision="HIKE",
            statement_text=(
                "La Junta de Gobierno del Banco de México decidió incrementar en 50 puntos base el objetivo para la Tasa de Interés Interbancaria a un día a un nivel de 10.50%. "
                "La inflación global se mantiene en niveles elevados y las presiones sobre la inflación subyacente continúan reflejando choques acumulados. "
                "La Junta considera necesario mantener una postura restrictiva para conducir la inflación a su meta de 3.00% en el horizonte de pronóstico."
            ),
            official_language="es",
            english_translation=(
                "The Governing Board of the Bank of Mexico decided to raise the target overnight interbank rate by 50 basis points to 10.50 percent. "
                "Global inflation remains elevated and core inflation pressures persist. "
                "The Board considers a restrictive stance necessary to guide inflation toward the 3 percent target."
            ),
        ),

        # People's Bank of China (PBOC)
        PolicyStatement(
            central_bank="PBOC",
            date="2023-08-15",
            policy_rate=2.50,
            rate_decision="CUT",
            statement_text=(
                "中国人民银行开展4010亿元1年期中期借贷便利（MLF）操作，中标利率下调15个基点至2.50%，以维护银行体系流动性合理充裕，加强逆周期调节，支持实体经济恢复发展。"
                "当前国内经济面临需求不足、房地产风险和外需放缓压力，稳健的货币政策将精准有力，加大信贷支持实体经济力度。"
            ),
            official_language="zh",
            english_translation=(
                "The People's Bank of China lowered the 1-year Medium-term Lending Facility (MLF) rate by 15 basis points to 2.50 percent to maintain ample liquidity and strengthen counter-cyclical adjustment. "
                "Facing soft domestic demand, real estate headwinds, and slowing exports, prudent monetary policy will provide targeted and robust credit support to the real economy."
            ),
        ),
    ]

    return statements


def load_central_bank_statements() -> List[PolicyStatement]:
    """Loads central bank statements."""
    return generate_central_bank_statements()


# =========================================================================
# 2. GLOBAL MACRO SOVEREIGN MARKET DATA (Module 47)
# =========================================================================

def generate_macro_market_data(
    n_days: int = 1500,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generates multi-year daily macro time series for DM and EM sovereign entities."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")

    countries = [
        "US", "Germany", "Japan", "UK", "Italy", "Greece",
        "Brazil", "Mexico", "India", "South_Africa", "Turkey"
    ]

    base_yields = {
        "US": 2.20, "Germany": -0.20, "Japan": 0.05, "UK": 1.20, "Italy": 2.50, "Greece": 3.80,
        "Brazil": 8.50, "Mexico": 7.20, "India": 6.80, "South_Africa": 8.90, "Turkey": 14.50
    }

    base_cds = {
        "US": 15.0, "Germany": 10.0, "Japan": 18.0, "UK": 20.0, "Italy": 110.0, "Greece": 160.0,
        "Brazil": 220.0, "Mexico": 140.0, "India": 110.0, "South_Africa": 260.0, "Turkey": 420.0
    }

    global_factor = np.cumsum(rng.normal(0, 0.03, n_days))
    em_shock_factor = np.cumsum(rng.normal(0, 0.05, n_days))

    em_crisis_event = np.zeros(n_days)
    if n_days >= 300:
        c_start = int(n_days * 0.45)
        c_end = min(n_days, c_start + int(n_days * 0.15))
        c_len = c_end - c_start
        if c_len > 0:
            em_crisis_event[c_start:c_end] = np.sin(np.linspace(0, np.pi, c_len)) * 250.0

    yields_dict = {}
    cds_dict = {}

    for c in countries:
        is_em = c in ["Brazil", "Mexico", "India", "South_Africa", "Turkey"]
        is_periphery = c in ["Italy", "Greece"]

        beta_global = 1.0 if not is_em else 1.6
        beta_em = 1.8 if is_em else (0.6 if is_periphery else 0.1)

        noise = rng.normal(0, 0.04 if is_em else 0.02, n_days)
        y_series = base_yields[c] + beta_global * global_factor + beta_em * (em_shock_factor * 0.4) + noise
        if c == "Turkey":
            y_series += em_crisis_event * 0.05
        yields_dict[c] = np.maximum(y_series, -0.80)

        cds_noise = rng.normal(0, 3.0 if is_em else 0.8, n_days)
        cds_s = base_cds[c] + beta_global * (global_factor * 40.0) + beta_em * (em_shock_factor * 60.0) + cds_noise
        if c == "Turkey":
            cds_s += em_crisis_event * 1.5
        elif c == "Brazil":
            cds_s += em_crisis_event * 0.8
        elif c == "South_Africa":
            cds_s += em_crisis_event * 0.5
        cds_dict[c] = np.maximum(cds_s, 5.0)

    yields_df = pd.DataFrame(yields_dict, index=dates)
    cds_df = pd.DataFrame(cds_dict, index=dates)

    macro_fundamentals = {
        "US": {"CPI": 3.2, "GDP": 2.4, "Policy_Rate": 5.25, "Debt_to_GDP": 122.0},
        "Germany": {"CPI": 2.4, "GDP": 0.3, "Policy_Rate": 3.75, "Debt_to_GDP": 66.0},
        "Japan": {"CPI": 2.6, "GDP": 0.8, "Policy_Rate": 0.10, "Debt_to_GDP": 260.0},
        "UK": {"CPI": 3.4, "GDP": 0.6, "Policy_Rate": 5.00, "Debt_to_GDP": 100.0},
        "Italy": {"CPI": 1.8, "GDP": 0.7, "Policy_Rate": 3.75, "Debt_to_GDP": 140.0},
        "Greece": {"CPI": 2.8, "GDP": 2.1, "Policy_Rate": 3.75, "Debt_to_GDP": 160.0},
        "Brazil": {"CPI": 4.2, "GDP": 2.5, "Policy_Rate": 10.50, "Debt_to_GDP": 74.0},
        "Mexico": {"CPI": 4.8, "GDP": 2.2, "Policy_Rate": 11.00, "Debt_to_GDP": 50.0},
        "India": {"CPI": 5.1, "GDP": 7.2, "Policy_Rate": 6.50, "Debt_to_GDP": 82.0},
        "South_Africa": {"CPI": 5.2, "GDP": 1.1, "Policy_Rate": 8.25, "Debt_to_GDP": 73.0},
        "Turkey": {"CPI": 65.0, "GDP": 4.1, "Policy_Rate": 50.00, "Debt_to_GDP": 34.0},
    }
    fundamentals_df = pd.DataFrame(macro_fundamentals).T

    return {
        "yields": yields_df,
        "cds_spreads": cds_df,
        "fundamentals": fundamentals_df,
    }


def load_macro_market_data() -> Dict[str, pd.DataFrame]:
    """Loads sovereign market data."""
    return generate_macro_market_data()


# =========================================================================
# 3. FINANCIAL NEWS & MACRO HEADLINES STREAM
# =========================================================================

def generate_news_and_social_stream() -> pd.DataFrame:
    """Generates synthetic multi-lingual news flow and macro event headlines."""
    news_items = [
        {"Timestamp": "2024-05-10 08:30:00", "Source": "Reuters", "Language": "en", "Ticker": "USD", "Headline": "US nonfarm payrolls beat expectations as wage growth accelerates to 4.2% YoY"},
        {"Timestamp": "2024-05-10 09:15:00", "Source": "Bloomberg", "Language": "en", "Ticker": "EUR", "Headline": "ECB officials signal openness to rate cuts if wage pressures ease across Eurozone"},
        {"Timestamp": "2024-05-11 02:00:00", "Source": "Nikkei", "Language": "ja", "Ticker": "JPY", "Headline": "日銀総裁、円安進行に伴う輸入物価上昇と追加利上げの可能性に言及"},
        {"Timestamp": "2024-05-12 11:30:00", "Source": "Valor Economico", "Language": "pt", "Ticker": "BRL", "Headline": "Banco Central alerta para risco fiscal crescente e desacelera ritmo de corte de juros"},
        {"Timestamp": "2024-05-13 14:00:00", "Source": "El Economista", "Language": "es", "Ticker": "MXN", "Headline": "Banxico mantiene cautela ante persistencia de inflación en servicios y volatilidad electoral"},
        {"Timestamp": "2024-05-14 06:30:00", "Source": "Economic Times", "Language": "en", "Ticker": "INR", "Headline": "RBI Governor emphasizes last mile of disinflation is challenging amid volatile food prices"},
    ]
    return pd.DataFrame(news_items)


# =========================================================================
# 4. FX RATES & VOLATILITY SURFACES
# =========================================================================

def generate_fx_rates_and_vol_surface(
    n_days: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates G10 and EM FX spot rates, interest rate differentials, 25-delta Risk Reversals & Butterfly spreads."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")

    pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDINR", "USDBRL", "USDMXN"]
    base_spots = {"EURUSD": 1.10, "USDJPY": 130.0, "GBPUSD": 1.30, "USDINR": 78.0, "USDBRL": 5.20, "USDMXN": 19.50}

    spots = {}
    for p in pairs:
        drift = 0.0001 if "INR" in p or "BRL" in p else 0.0
        vol = 0.007 if "USDJPY" in p or "BRL" in p else 0.004
        ret = rng.normal(drift, vol, n_days)
        spots[p] = base_spots[p] * np.cumprod(1.0 + ret)

    spots_df = pd.DataFrame(spots, index=dates)

    fx_vol_skew = {
        "EURUSD": {"ATM_Vol": 6.8, "25D_Risk_Reversal": -0.85, "25D_Butterfly": 0.22},
        "USDJPY": {"ATM_Vol": 9.4, "25D_Risk_Reversal": +1.40, "25D_Butterfly": 0.35},
        "GBPUSD": {"ATM_Vol": 7.5, "25D_Risk_Reversal": -0.65, "25D_Butterfly": 0.20},
        "USDBRL": {"ATM_Vol": 14.2, "25D_Risk_Reversal": +2.80, "25D_Butterfly": 0.65},
        "USDMXN": {"ATM_Vol": 11.5, "25D_Risk_Reversal": +2.10, "25D_Butterfly": 0.48},
        "USDINR": {"ATM_Vol": 4.5, "25D_Risk_Reversal": +0.90, "25D_Butterfly": 0.15},
    }
    vol_skew_df = pd.DataFrame(fx_vol_skew).T

    return {
        "spot_rates": spots_df,
        "vol_skew": vol_skew_df,
    }


def load_fx_market_data() -> Dict[str, Any]:
    """Loads FX spot and volatility skew data."""
    return generate_fx_rates_and_vol_surface()
