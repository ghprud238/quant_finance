#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace nexus_quant::strategies {

struct BacktestMetrics {
    double cagr;
    double annual_vol;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double win_rate;
    double profit_factor;
};

class SystematicBacktester {
public:
    double fee_bps;
    double half_spread_bps;
    double borrow_cost_annual;

    SystematicBacktester(double fee = 5.0, double spread = 2.5, double borrow = 0.005)
        : fee_bps(fee), half_spread_bps(spread), borrow_cost_annual(borrow) {}

    BacktestMetrics run(const std::vector<double>& prices, const std::vector<double>& target_weights) {
        size_t n = std::min(prices.size(), target_weights.size());
        if (n < 3) return {0,0,0,0,0,0,0};
        std::vector<double> rets(n - 1);
        std::vector<double> net_rets(n - 1);
        double cum = 1.0;
        double peak = 1.0;
        double max_dd = 0.0;
        size_t wins = 0;
        double gain_sum = 0.0, loss_sum = 0.0;
        double prev_w = 0.0;

        for (size_t t = 1; t < n; ++t) {
            double r = (prices[t] - prices[t - 1]) / prices[t - 1];
            double w = target_weights[t - 1]; // Lagged execution (no lookahead)
            double turnover = std::abs(w - prev_w);
            double cost_drag = turnover * (fee_bps + half_spread_bps) * 1e-4;
            if (w < 0.0) cost_drag += std::abs(w) * (borrow_cost_annual / 252.0);
            double net_r = w * r - cost_drag;
            net_rets[t - 1] = net_r;
            cum *= (1.0 + net_r);
            peak = std::max(peak, cum);
            double dd = (peak - cum) / peak;
            max_dd = std::max(max_dd, dd);
            if (net_r > 0) { wins++; gain_sum += net_r; }
            else if (net_r < 0) { loss_sum += std::abs(net_r); }
            prev_w = w;
        }

        double years = static_cast<double>(net_rets.size()) / 252.0;
        double cagr = std::pow(std::max(cum, 1e-8), 1.0 / years) - 1.0;
        double m = core::mean(net_rets);
        double s = core::stdev(net_rets, 1);
        double ann_vol = s * std::sqrt(252.0);
        double sharpe = (ann_vol > 1e-8) ? (m * 252.0 - 0.02) / ann_vol : 0.0;

        std::vector<double> downs;
        for (double r : net_rets) if (r < 0) downs.push_back(r * r);
        double down_dev = std::sqrt(core::mean(downs)) * std::sqrt(252.0);
        double sortino = (down_dev > 1e-8) ? (m * 252.0 - 0.02) / down_dev : 0.0;
        double win_rate = static_cast<double>(wins) / net_rets.size();
        double pf = (loss_sum > 1e-8) ? gain_sum / loss_sum : 2.0;

        return {cagr, ann_vol, sharpe, sortino, max_dd, win_rate, pf};
    }
};

class OnlineKalmanFilterPairs {
public:
    double alpha = 0.0;
    double beta = 1.0;
    double P[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    double Q[2][2] = {{1e-4, 0.0}, {0.0, 1e-4}};
    double R = 1e-3;

    double update(double y, double x) {
        // Prior prediction
        P[0][0] += Q[0][0]; P[1][1] += Q[1][1];
        // Innovation
        double y_pred = alpha + beta * x;
        double e = y - y_pred;
        double S = P[0][0] + 2.0 * x * P[0][1] + x * x * P[1][1] + R;
        // Kalman Gain
        double K0 = (P[0][0] + x * P[0][1]) / S;
        double K1 = (P[1][0] + x * P[1][1]) / S;
        // State update
        alpha += K0 * e;
        beta += K1 * e;
        // Covariance update
        double P00_new = P[0][0] - K0 * (P[0][0] + x * P[1][0]);
        double P11_new = P[1][1] - K1 * (P[0][1] + x * P[1][1]);
        P[0][0] = P00_new; P[1][1] = P11_new;
        return e / std::sqrt(std::max(S, 1e-8)); // Standardized Kalman Z-Score
    }
};

} // namespace nexus_quant::strategies
