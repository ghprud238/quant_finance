#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace nexus_quant::defi {

class Uniswapv3ConcentratedPool {
public:
    double current_price;
    double fee_rate;
    double liquidity; // Active virtual liquidity L

    Uniswapv3ConcentratedPool(double p = 3000.0, double f = 0.0030, double L = 1e6)
        : current_price(p), fee_rate(f), liquidity(L) {}

    static double capital_efficiency_multiplier(double pa, double pb) {
        return 1.0 / (1.0 - std::sqrt(std::sqrt(pa / pb)));
    }

    double swap_x_for_y(double delta_x) {
        double fee_factor = 1.0 - fee_rate;
        double effective_dx = delta_x * fee_factor;
        double sqrt_p = std::sqrt(current_price);
        double sqrt_p_next = (sqrt_p * liquidity) / (liquidity + effective_dx * sqrt_p);
        double delta_y = liquidity * (sqrt_p - sqrt_p_next);
        current_price = sqrt_p_next * sqrt_p_next;
        return delta_y;
    }
};

class LossVersusRebalancing {
public:
    static double continuous_lvr(double annual_vol, double annual_tvl, double years = 1.0) {
        // LVR = sigma^2 / 8 * S * L
        return (annual_vol * annual_vol / 8.0) * annual_tvl * years;
    }

    static double breakeven_volatility(double fee_revenue, double annual_tvl) {
        if (annual_tvl <= 1e-8) return 0.0;
        return std::sqrt((8.0 * fee_revenue) / annual_tvl);
    }
};

class PredictionMarketKellyArbitrage {
public:
    static double intra_venue_edge(double p_yes_ask, double p_no_ask) {
        double sum = p_yes_ask + p_no_ask;
        return (sum < 1.0) ? (1.0 - sum) : 0.0;
    }

    static double kelly_fraction(double true_win_prob, double combined_price, double fractional_multiplier = 0.50) {
        if (combined_price >= 1.0 || combined_price <= 0.0) return 0.0;
        double b = (1.0 - combined_price) / combined_price;
        double f_star = (b * true_win_prob - (1.0 - true_win_prob)) / b;
        return std::max(f_star * fractional_multiplier, 0.0);
    }
};

} // namespace nexus_quant::defi

#pragma once
