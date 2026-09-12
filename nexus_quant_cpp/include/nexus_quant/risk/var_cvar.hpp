#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace nexus_quant::risk {

inline double historical_var(std::vector<double> returns, double confidence = 0.95) {
    if (returns.empty()) return 0.0;
    std::sort(returns.begin(), returns.end());
    size_t idx = static_cast<size_t>(std::floor((1.0 - confidence) * returns.size()));
    idx = std::min(idx, returns.size() - 1);
    return -returns[idx];
}

inline double historical_cvar(std::vector<double> returns, double confidence = 0.95) {
    if (returns.empty()) return 0.0;
    std::sort(returns.begin(), returns.end());
    size_t cutoff_idx = static_cast<size_t>(std::floor((1.0 - confidence) * returns.size()));
    cutoff_idx = std::max(cutoff_idx, size_t(1));
    double sum = 0.0;
    for (size_t i = 0; i < cutoff_idx; ++i) sum += returns[i];
    return -(sum / cutoff_idx);
}

inline double parametric_gaussian_var(const std::vector<double>& returns, double confidence = 0.95) {
    double m = core::mean(returns);
    double s = core::stdev(returns, 1);
    double z = core::inv_normal_cdf(confidence);
    return -(m - z * s);
}

inline double cornish_fisher_var(const std::vector<double>& returns, double confidence = 0.95) {
    double m = core::mean(returns);
    double s = core::stdev(returns, 1);
    double skew = core::skewness(returns);
    double kurt = core::kurtosis(returns, true);
    double z = core::inv_normal_cdf(confidence);
    double z_cf = z + (1.0 / 6.0) * (z * z - 1.0) * skew
                    + (1.0 / 24.0) * (z * z * z - 3.0 * z) * kurt
                    - (1.0 / 36.0) * (2.0 * z * z * z - 5.0 * z) * skew * skew;
    return -(m - z_cf * s);
}

inline double kupiec_pof_test(size_t violations, size_t total_obs, double expected_prob) {
    if (violations == 0 || violations >= total_obs) return 0.0;
    double p_hat = static_cast<double>(violations) / total_obs;
    double log_num = (total_obs - violations) * std::log(1.0 - expected_prob) + violations * std::log(expected_prob);
    double log_den = (total_obs - violations) * std::log(1.0 - p_hat) + violations * std::log(p_hat);
    return -2.0 * (log_num - log_den);
}

struct ClimateVaRResult {
    double ebitda_impairment_pct;
    double equity_climate_var_pct;
    double stressed_pd_pct;
    double credit_spread_widening_bps;
};

inline ClimateVaRResult evaluate_ngfs_climate_var(double market_cap, double ebitda, double debt,
                                                  double scope1_2_tco2, double carbon_tax_per_ton,
                                                  double pass_through_pct = 0.30) {
    double unhedged_tax = carbon_tax_per_ton * scope1_2_tco2 * (1.0 - pass_through_pct);
    double ebitda_impairment = std::min(unhedged_tax / std::max(ebitda, 1.0), 1.0);
    double dcf_equity_impact = -std::min((unhedged_tax * 7.5) / std::max(market_cap, 1.0), 0.95);
    double stressed_firm_value = (market_cap * (1.0 + dcf_equity_impact)) + debt;
    double leverage = debt / std::max(stressed_firm_value, 1.0);
    double stressed_pd = core::normal_cdf((std::log(leverage) + 0.02) / 0.25);
    double spread_widening = -std::log(1.0 - 0.60 * stressed_pd) * 10000.0;
    return {ebitda_impairment * 100.0, dcf_equity_impact * 100.0, stressed_pd * 100.0, spread_widening};
}

} // namespace nexus_quant::risk

#pragma once
