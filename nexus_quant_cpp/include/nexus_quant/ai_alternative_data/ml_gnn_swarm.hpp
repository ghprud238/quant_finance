#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <numeric>

namespace nexus_quant::ai {

inline std::vector<double> fractional_differentiation(const std::vector<double>& series, double d, double threshold = 1e-4) {
    if (series.empty()) return {};
    std::vector<double> w = {1.0};
    size_t k = 1;
    while (true) {
        double next_w = -w.back() / k * (d - k + 1.0);
        if (std::abs(next_w) < threshold || k > series.size()) break;
        w.push_back(next_w);
        k++;
    }
    size_t l = w.size();
    if (series.size() < l) return series;
    std::vector<double> res(series.size() - l + 1, 0.0);
    for (size_t i = 0; i < res.size(); ++i) {
        double val = 0.0;
        for (size_t j = 0; j < l; ++j) {
            val += w[j] * series[i + l - 1 - j];
        }
        res[i] = val;
    }
    return res;
}

class GNNSupplyChainMessagePassing {
public:
    static std::vector<double> forward(const core::Matrix& adjacency_revenue, const std::vector<double>& customer_returns) {
        size_t n = customer_returns.size();
        std::vector<double> supplier_signals(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            double sum_rev = 0.0;
            double signal = 0.0;
            for (size_t j = 0; j < n; ++j) {
                double a_ij = adjacency_revenue(i, j);
                if (a_ij > 0.0) {
                    signal += a_ij * customer_returns[j];
                    sum_rev += a_ij;
                }
            }
            supplier_signals[i] = (sum_rev > 0.0) ? signal / sum_rev : 0.0;
        }
        return supplier_signals;
    }
};

} // namespace nexus_quant::ai

#pragma once
