#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>

namespace nexus_quant::equities_etf {

struct ETFBasket {
    std::string etf_ticker;
    std::map<std::string, double> constituent_shares; // Ticker -> share units
    double cash_component;
    double creation_unit_size; // e.g. 50,000 shares
};

class ETFiNAVCalculator {
public:
    ETFBasket basket;

    ETFiNAVCalculator(const ETFBasket& b) : basket(b) {}

    double calculate_inav(const std::map<std::string, double>& prices) const {
        double total = basket.cash_component;
        for (const auto& [ticker, shares] : basket.constituent_shares) {
            auto it = prices.find(ticker);
            if (it != prices.end()) total += shares * it->second;
        }
        return total / basket.creation_unit_size;
    }

    std::pair<double, double> compute_basis(double etf_market_price, const std::map<std::string, double>& prices) const {
        double inav = calculate_inav(prices);
        double basis_usd = etf_market_price - inav;
        double basis_bps = (inav > 1e-6) ? (basis_usd / inav) * 10000.0 : 0.0;
        return {basis_usd, basis_bps};
    }
};

class BarraEquityRiskModel {
public:
    static double estimate_residual_ou_halflife(const std::vector<double>& residuals) {
        if (residuals.size() < 4) return 10.0;
        double sum_xy = 0.0, sum_xx = 0.0;
        double m = core::mean(residuals);
        for (size_t t = 1; t < residuals.size(); ++t) {
            double x = residuals[t - 1] - m;
            double y = residuals[t] - m;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        double b = (sum_xx > 1e-8) ? sum_xy / sum_xx : 0.95;
        b = std::clamp(b, 0.01, 0.999);
        double kappa = -std::log(b);
        return std::log(2.0) / kappa; // Half-life in trading days
    }
};

class CrossAssetDeltaHedger {
public:
    std::string hedge_proxy = "SPY";

    double calculate_hedge_shares(double unhedged_dollar_delta, double hedge_proxy_price) const {
        if (hedge_proxy_price <= 1e-6) return 0.0;
        return -unhedged_dollar_delta / hedge_proxy_price;
    }
};

} // namespace nexus_quant::equities_etf

#pragma once
