#include "../include/nexus_quant/nexus_quant.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_track01_foundations() {
    std::vector<double> prices = {100.0, 102.0, 101.0, 103.5, 105.0};
    auto rets = nq::foundations::simple_returns(prices);
    assert(rets.size() == 4);
    assert(std::abs(rets[0] - 0.02) < 1e-6);

    std::vector<double> opens = {100.0, 102.0, 101.0, 103.5};
    std::vector<double> highs = {103.0, 104.0, 103.0, 106.0};
    std::vector<double> lows = {99.0, 100.5, 100.0, 102.0};
    std::vector<double> closes = {102.0, 101.0, 103.5, 105.0};

    double p_vol = nq::foundations::parkinson_volatility(highs, lows);
    double gk_vol = nq::foundations::garman_klass_volatility(opens, highs, lows, closes);
    double yz_vol = nq::foundations::yang_zhang_volatility(opens, highs, lows, closes);

    assert(p_vol > 0.0);
    assert(gk_vol > 0.0);
    assert(yz_vol > 0.0);

    nq::foundations::GaussianHMM3State hmm;
    auto regime = hmm.predict(0.005);
    assert(regime.current_regime >= 0 && regime.current_regime <= 2);
    std::cout << "[+] Track 01 (Foundations & Volatility): PASSED\n";
}

void test_track02_risk() {
    std::vector<double> rets = {-0.03, -0.02, -0.01, 0.005, 0.01, 0.015, 0.02, -0.04, 0.005, 0.01};
    double h_var = nq::risk::historical_var(rets, 0.95);
    double c_var = nq::risk::historical_cvar(rets, 0.95);
    double cf_var = nq::risk::cornish_fisher_var(rets, 0.95);
    assert(h_var > 0.0);
    assert(c_var >= h_var); // Coherence property
    assert(cf_var > 0.0);

    double lr_pof = nq::risk::kupiec_pof_test(5, 252, 0.05);
    assert(lr_pof >= 0.0);

    auto c_res = nq::risk::evaluate_ngfs_climate_var(1e10, 2e9, 5e9, 1e7, 140.0);
    assert(c_res.ebitda_impairment_pct > 0.0);
    std::cout << "[+] Track 02 (Risk & Climate VaR): PASSED\n";
}

void test_track03_strategies() {
    std::vector<double> prices = {100, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 107};
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    nq::strategies::SystematicBacktester bt(5.0, 2.5, 0.005);
    auto res = bt.run(prices, weights);
    assert(res.cagr > 0.0);
    assert(res.sharpe_ratio > 0.0);

    nq::strategies::OnlineKalmanFilterPairs kf;
    double z = kf.update(102.0, 100.0);
    assert(!std::isnan(z));
    std::cout << "[+] Track 03 (Systematic Strategies & Kalman Filter): PASSED\n";
}

void test_track04_derivatives() {
    double call = nq::derivatives::BlackScholesModel::call_price(100.0, 100.0, 1.0, 0.05, 0.20);
    double put = nq::derivatives::BlackScholesModel::put_price(100.0, 100.0, 1.0, 0.05, 0.20);
    assert(call > 0.0 && put > 0.0);
    // Put-Call Parity: C - P = S - K * exp(-rT)
    double parity_diff = (call - put) - (100.0 - 100.0 * std::exp(-0.05));
    assert(std::abs(parity_diff) < 1e-6);

    auto greeks = nq::derivatives::BlackScholesModel::calculate_greeks(100.0, 100.0, 1.0, 0.05, 0.20);
    assert(greeks.delta > 0.5 && greeks.delta < 0.7);
    assert(greeks.gamma > 0.0);
    assert(greeks.vega > 0.0);

    double cos_call = nq::derivatives::HestonCOSPricer::price_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.20, -0.5);
    assert(cos_call > 0.0);

    std::vector<double> strikes = {80, 90, 100, 110, 120};
    auto rnd = nq::derivatives::extract_breeden_litzenberger_density(100.0, strikes, 1.0, 0.05, 0.0, 0.20);
    assert(rnd.size() == 5);
    std::cout << "[+] Track 04 (Derivatives, Heston & Breeden-Litzenberger): PASSED\n";
}

void test_track05_macro() {
    nq::macro::NelsonSiegelModel ns(0.045, -0.015, 0.020, 1.35);
    double y10 = ns.zero_rate(10.0);
    double f10 = ns.forward_rate(10.0);
    assert(y10 > 0.0 && f10 > 0.0);

    nq::macro::CarbonETSModel carbon;
    double p_switch = carbon.fuel_switching_parity_price(26.0, 13.0);
    assert(p_switch > 0.0);
    double greenium = carbon.greenium_bps(0.0425, 0.0420);
    assert(std::abs(greenium - 5.0) < 1e-6);
    std::cout << "[+] Track 05 (Macro, Yield Curves & Carbon Markets): PASSED\n";
}

void test_track06_microstructure() {
    nq::microstructure::LimitOrderBook lob;
    lob.add_limit_order({"B1", "buy", 99.95, 1000});
    lob.add_limit_order({"A1", "sell", 100.05, 500});
    assert(std::abs(lob.mid_price() - 100.0) < 1e-6);
    assert(std::abs(lob.spread() - 0.10) < 1e-6);
    assert(lob.order_book_imbalance() > 0.0); // More bids than asks

    nq::microstructure::AlmgrenChrissModel ac(1e6, 1.0, 20, 0.30);
    auto traj = ac.solve_trajectory();
    assert(traj.size() == 21);
    assert(std::abs(traj[0] - 1e6) < 1e-4);
    assert(std::abs(traj.back()) < 1e-6);

    nq::microstructure::AvellanedaStoikovMarketMaker mm;
    auto [bid, ask] = mm.optimal_quotes(100.0, 5, 0.5);
    assert(bid < ask);
    std::cout << "[+] Track 06 (Microstructure, LOB & Almgren-Chriss): PASSED\n";
}

void test_track07_ai_alt_data() {
    std::vector<double> s = {100, 101, 102, 101.5, 103, 104, 105, 106};
    auto ffd = nq::ai::fractional_differentiation(s, 0.40);
    assert(!ffd.empty());

    nq::core::Matrix A(3, 3, 0.0);
    A(0, 1) = 0.40; A(1, 2) = 0.60;
    std::vector<double> cust_ret = {0.02, -0.01, 0.03};
    auto supp_sig = nq::ai::GNNSupplyChainMessagePassing::forward(A, cust_ret);
    assert(supp_sig.size() == 3);
    std::cout << "[+] Track 07 (AI, Fractional Diff & GNN Supply Chain): PASSED\n";
}

void test_track08_defi() {
    nq::defi::Uniswapv3ConcentratedPool pool(3000.0, 0.0030, 1e6);
    double dy = pool.swap_x_for_y(1.0);
    assert(dy > 0.0);
    assert(pool.current_price < 3000.0); // Selling X drops price

    double lvr = nq::defi::LossVersusRebalancing::continuous_lvr(0.60, 1e7);
    assert(lvr > 0.0);

    double edge = nq::defi::PredictionMarketKellyArbitrage::intra_venue_edge(0.48, 0.49);
    assert(std::abs(edge - 0.03) < 1e-6);
    double kelly = nq::defi::PredictionMarketKellyArbitrage::kelly_fraction(0.999, 0.97);
    assert(kelly > 0.0);
    std::cout << "[+] Track 08 (DeFi, Uniswap v3, LVR & Prediction Markets): PASSED\n";
}

void test_track09_equities_etf() {
    nq::equities_etf::ETFBasket basket{"TECH_ETF", {{"AAPL", 1000}, {"MSFT", 800}}, 10000.0, 50000};
    nq::equities_etf::ETFiNAVCalculator calc(basket);
    double inav = calc.calculate_inav({{"AAPL", 180.0}, {"MSFT", 350.0}});
    assert(inav > 0.0);

    auto [b_usd, b_bps] = calc.compute_basis(inav + 0.50, {{"AAPL", 180.0}, {"MSFT", 350.0}});
    assert(b_usd > 0.0 && b_bps > 0.0);

    nq::equities_etf::CrossAssetDeltaHedger hedger;
    double hedge_sh = hedger.calculate_hedge_shares(450000.0, 450.0);
    assert(std::abs(hedge_sh - (-1000.0)) < 1e-6);
    std::cout << "[+] Track 09 (Equities & ETF iNAV Arbitrage, Hedging): PASSED\n";
}

void test_track10_rigor_orchestrator() {
    double exp_max_sr = nq::rigor::DeflatedSharpeCalculator::expected_max_sharpe(1000);
    assert(exp_max_sr > 0.0);
    double dsr = nq::rigor::DeflatedSharpeCalculator::deflated_sharpe_ratio(2.33, 1000, 1260);
    assert(dsr > 0.95); // High confidence

    auto audit = nq::orchestrator::NexusMasterOrchestrator::audit_cross_asset_platform();
    assert(audit.approved_for_production);
    assert(audit.sharpe_ratio > 2.0);
    std::cout << "[+] Track 10 (Rigor, Deflated Sharpe & Master Orchestrator): PASSED\n";
}

int main() {
    std::cout << "===============================================================\n";
    std::cout << "  NEXUS QUANT PLATFORM C++20 COMPREHENSIVE TEST RUNNER (ALL TRACKS)\n";
    std::cout << "===============================================================\n";
    test_track01_foundations();
    test_track02_risk();
    test_track03_strategies();
    test_track04_derivatives();
    test_track05_macro();
    test_track06_microstructure();
    test_track07_ai_alt_data();
    test_track08_defi();
    test_track09_equities_etf();
    test_track10_rigor_orchestrator();
    std::cout << "===============================================================\n";
    std::cout << "  ALL 10 VERIFICATION TEST SUITES PASSED (100% SUCCESS RATE)\n";
    std::cout << "===============================================================\n";
    return 0;
}
