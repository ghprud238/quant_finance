"""
Example 04: Global Macro, Alternative Data & Central Bank NLP Workflow
=====================================================================
"""
import numpy as np
import nexus_quant as nq

# 1. Central Bank Monetary Policy NLP
cb_indexer = nq.macro.central_bank_nlp.CentralBankStanceIndexer()
stmt = "Inflation pressures remain elevated above our 2.0% target. Further policy tightening is warranted."
stance = cb_indexer.score_statement(stmt, central_bank="FED", actual_policy_rate=5.25)
print("Central Bank Hawk/Dove Tone Index:")
print(f"  - Central Bank:   {stance.central_bank}")
print(f"  - Stance:         {stance.stance_category}")
print(f"  - Score:          {stance.hawk_dove_score:+.2f}")
print(f"  - Expected Delta: {stance.expected_yield_impact_bps:+.1f} bps")

# 2. SEC 10-K Semantic Drift & Lazy Prices Anomaly
drift_engine = nq.ai.sec_semantic_drift.SemanticDriftEngine()
prev_10k = "The company faces standard competitive pressures and operational risks."
curr_10k_lazy = "The company faces standard competitive pressures and operational risks."
curr_10k_drift = "The company faces severe export embargoes, antitrust litigation, and supply-chain collapse."

d_lazy = drift_engine.compute_cosine_drift(prev_10k, curr_10k_lazy)
d_drift = drift_engine.compute_cosine_drift(prev_10k, curr_10k_drift)

print("\nSEC 10-K Semantic Drift & Lazy Prices:")
print(f"  - Lazy Filing Drift Score:    {d_lazy:.4f} (Recommendation: LONG)")
print(f"  - Changed Filing Drift Score: {d_drift:.4f} (Recommendation: SHORT)")

# 3. Nelson-Siegel Yield Curve Calibration
maturities = np.array([0.25, 0.50, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
par_yields = np.array([5.20, 5.00, 4.70, 4.40, 4.20, 4.10, 4.15, 4.25, 4.50, 4.60])
ns = nq.macro.yield_curve.NelsonSiegelCalibrator.fit_nelson_siegel(maturities, par_yields)
print("\nNelson-Siegel Yield Curve Calibration:")
print(f"  - 10Y Model Spot Rate:       {ns.spot_rate(10.0):.2f}%")
print(f"  - 10Y Instantaneous Forward: {ns.forward_rate(10.0):.2f}%")
