#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>

namespace nexus_quant::microstructure {

struct LimitOrder {
    std::string order_id;
    std::string side; // "buy" or "sell"
    double price;
    double volume;
};

class LimitOrderBook {
public:
    std::map<double, double, std::greater<double>> bids; // Price -> Volume descending
    std::map<double, double, std::less<double>> asks;    // Price -> Volume ascending

    void add_limit_order(const LimitOrder& order) {
        if (order.side == "buy") bids[order.price] += order.volume;
        else asks[order.price] += order.volume;
    }

    double best_bid() const { return bids.empty() ? 0.0 : bids.begin()->first; }
    double best_ask() const { return asks.empty() ? 0.0 : asks.begin()->first; }
    double mid_price() const { return 0.5 * (best_bid() + best_ask()); }
    double spread() const { return best_ask() - best_bid(); }

    double order_book_imbalance() const {
        double vb = bids.empty() ? 0.0 : bids.begin()->second;
        double va = asks.empty() ? 0.0 : asks.begin()->second;
        if (vb + va <= 1e-8) return 0.0;
        return (vb - va) / (vb + va);
    }

    double micro_price() const {
        double pb = best_bid(), pa = best_ask();
        double vb = bids.empty() ? 0.0 : bids.begin()->second;
        double va = asks.empty() ? 0.0 : asks.begin()->second;
        if (vb + va <= 1e-8) return mid_price();
        return (vb * pa + va * pb) / (vb + va);
    }
};

class AlmgrenChrissModel {
public:
    double X0;        // Total shares
    double T;         // Horizon
    size_t N;         // Intervals
    double sigma;     // Volatility
    double eta;       // Temp impact
    double gamma_perm;// Perm impact
    double lambda;    // Risk aversion

    AlmgrenChrissModel(double shares = 1e6, double horizon = 1.0, size_t intervals = 20,
                       double vol = 0.30, double temp_imp = 2.5e-6, double perm_imp = 2.5e-7, double risk_av = 1e-6)
        : X0(shares), T(horizon), N(intervals), sigma(vol), eta(temp_imp), gamma_perm(perm_imp), lambda(risk_av) {}

    std::vector<double> solve_trajectory() const {
        double tau = T / N;
        double kappa = std::sqrt((lambda * sigma * sigma) / eta);
        std::vector<double> x(N + 1);
        for (size_t j = 0; j <= N; ++j) {
            double tj = j * tau;
            x[j] = (std::sinh(kappa * (T - tj)) / std::sinh(kappa * T)) * X0;
        }
        return x;
    }

    double expected_shortfall_cost() const {
        auto traj = solve_trajectory();
        double tau = T / N;
        double cost = 0.5 * gamma_perm * X0 * X0;
        for (size_t j = 1; j <= N; ++j) {
            double nj = traj[j - 1] - traj[j];
            cost += eta * (nj * nj) / tau;
        }
        return cost;
    }
};

class AvellanedaStoikovMarketMaker {
public:
    double gamma; // Risk aversion
    double sigma; // Volatility
    double kappa; // Liquidity parameter
    double T;     // Horizon

    AvellanedaStoikovMarketMaker(double g = 0.10, double s = 0.25, double k = 1.50, double horiz = 1.0)
        : gamma(g), sigma(s), kappa(k), T(horiz) {}

    double reservation_price(double spot, double inventory, double t) const {
        return spot - inventory * gamma * sigma * sigma * (T - t);
    }

    std::pair<double, double> optimal_quotes(double spot, double inventory, double t) const {
        double r = reservation_price(spot, inventory, t);
        double half_spread = (1.0 / gamma) * std::log(1.0 + gamma / kappa) + 0.5 * gamma * sigma * sigma * (T - t);
        return {r - half_spread, r + half_spread}; // {bid, ask}
    }
};

} // namespace nexus_quant::microstructure

#pragma once
