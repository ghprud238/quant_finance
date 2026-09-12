#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace nexus_quant::foundations {

inline std::vector<double> simple_returns(const std::vector<double>& prices) {
    if (prices.size() < 2) return {};
    std::vector<double> rets(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        rets[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
    }
    return rets;
}

inline std::vector<double> log_returns(const std::vector<double>& prices) {
    if (prices.size() < 2) return {};
    std::vector<double> rets(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        rets[i - 1] = std::log(prices[i] / prices[i - 1]);
    }
    return rets;
}

inline double annualized_cagr(const std::vector<double>& returns, double periods_per_year = 252.0) {
    if (returns.empty()) return 0.0;
    double cum = 1.0;
    for (double r : returns) cum *= (1.0 + r);
    double years = static_cast<double>(returns.size()) / periods_per_year;
    return std::pow(std::max(cum, 1e-8), 1.0 / years) - 1.0;
}

inline double parkinson_volatility(const std::vector<double>& highs,
                                   const std::vector<double>& lows,
                                   double periods_per_year = 252.0) {
    size_t n = std::min(highs.size(), lows.size());
    if (n < 2) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double hl = std::log(std::max(highs[i], 1e-8) / std::max(lows[i], 1e-8));
        sum_sq += hl * hl;
    }
    return std::sqrt((periods_per_year / (4.0 * std::log(2.0) * n)) * sum_sq);
}

inline double garman_klass_volatility(const std::vector<double>& opens,
                                      const std::vector<double>& highs,
                                      const std::vector<double>& lows,
                                      const std::vector<double>& closes,
                                      double periods_per_year = 252.0) {
    size_t n = std::min({opens.size(), highs.size(), lows.size(), closes.size()});
    if (n < 2) return 0.0;
    double sum = 0.0;
    double coeff = 2.0 * std::log(2.0) - 1.0;
    for (size_t i = 0; i < n; ++i) {
        double hl = std::log(std::max(highs[i], 1e-8) / std::max(lows[i], 1e-8));
        double co = std::log(std::max(closes[i], 1e-8) / std::max(opens[i], 1e-8));
        sum += 0.5 * hl * hl - coeff * co * co;
    }
    return std::sqrt(std::max((periods_per_year / n) * sum, 0.0));
}

inline double yang_zhang_volatility(const std::vector<double>& opens,
                                    const std::vector<double>& highs,
                                    const std::vector<double>& lows,
                                    const std::vector<double>& closes,
                                    double periods_per_year = 252.0) {
    size_t n = std::min({opens.size(), highs.size(), lows.size(), closes.size()});
    if (n < 3) return 0.0;
    std::vector<double> o_prev_c(n - 1);
    std::vector<double> c_o(n);
    for (size_t i = 1; i < n; ++i) {
        o_prev_c[i - 1] = std::log(opens[i] / closes[i - 1]);
    }
    for (size_t i = 0; i < n; ++i) {
        c_o[i] = std::log(closes[i] / opens[i]);
    }
    double var_o = core::variance(o_prev_c, 1);
    double var_c = core::variance(c_o, 1);
    double rs_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double u = std::log(highs[i] / opens[i]);
        double d = std::log(lows[i] / opens[i]);
        double c = std::log(closes[i] / opens[i]);
        rs_sum += u * (u - c) + d * (d - c);
    }
    double var_rs = rs_sum / n;
    double k = 0.34 / (1.34 + static_cast<double>(n + 1) / (n - 1));
    double total_var = var_o + k * var_c + (1.0 - k) * var_rs;
    return std::sqrt(std::max(total_var * periods_per_year, 0.0));
}

struct HMMRegimeState {
    int current_regime; // 0: Bear, 1: Neutral, 2: Bull
    std::vector<double> state_probs; // [p0, p1, p2]
    double mean_regime;
    double vol_regime;
};

class GaussianHMM3State {
public:
    std::vector<double> means = {-0.0015, 0.0003, 0.0012};
    std::vector<double> vars = {0.000625, 0.000100, 0.000064}; // Stdevs: 2.5%, 1.0%, 0.8%
    core::Matrix transition_matrix;

    GaussianHMM3State() {
        transition_matrix = core::Matrix(3, 3, 0.05);
        transition_matrix(0, 0) = 0.90; transition_matrix(0, 1) = 0.08; transition_matrix(0, 2) = 0.02;
        transition_matrix(1, 0) = 0.05; transition_matrix(1, 1) = 0.90; transition_matrix(1, 2) = 0.05;
        transition_matrix(2, 0) = 0.02; transition_matrix(2, 1) = 0.08; transition_matrix(2, 2) = 0.90;
    }

    HMMRegimeState predict(double current_return, const std::vector<double>& prior_state_probs = {0.2, 0.5, 0.3}) {
        std::vector<double> posterior(3, 0.0);
        double total = 0.0;
        for (int i = 0; i < 3; ++i) {
            double prior = 0.0;
            for (int j = 0; j < 3; ++j) prior += prior_state_probs[j] * transition_matrix(j, i);
            double diff = current_return - means[i];
            double emit = (1.0 / std::sqrt(2.0 * core::PI * vars[i])) * std::exp(-0.5 * (diff * diff) / vars[i]);
            posterior[i] = prior * emit;
            total += posterior[i];
        }
        if (total > 0.0) {
            for (int i = 0; i < 3; ++i) posterior[i] /= total;
        } else {
            posterior = {0.33, 0.33, 0.34};
        }
        int best_state = static_cast<int>(std::distance(posterior.begin(), std::max_element(posterior.begin(), posterior.end())));
        return HMMRegimeState{best_state, posterior, means[best_state], std::sqrt(vars[best_state]) * std::sqrt(252.0)};
    }
};

} // namespace nexus_quant::foundations
