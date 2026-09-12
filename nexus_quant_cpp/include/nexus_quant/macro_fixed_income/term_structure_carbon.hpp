#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

namespace nexus_quant::macro {

class NelsonSiegelModel {
public:
    double beta0; // Level
    double beta1; // Slope
    double beta2; // Curvature
    double lambda; // Decay

    NelsonSiegelModel(double b0 = 0.045, double b1 = -0.015, double b2 = 0.020, double lam = 1.35)
        : beta0(b0), beta1(b1), beta2(b2), lambda(lam) {}

    double zero_rate(double tau) const {
        if (tau <= 1e-4) return beta0 + beta1;
        double t_lam = tau / lambda;
        double factor1 = (1.0 - std::exp(-t_lam)) / t_lam;
        double factor2 = factor1 - std::exp(-t_lam);
        return beta0 + beta1 * factor1 + beta2 * factor2;
    }

    double forward_rate(double tau) const {
        double t_lam = tau / lambda;
        return beta0 + beta1 * std::exp(-t_lam) + beta2 * t_lam * std::exp(-t_lam);
    }
};

class CarbonETSModel {
public:
    double efficiency_gas = 0.50;
    double efficiency_coal = 0.38;
    double emission_factor_gas = 0.37; // tCO2/MWh_e
    double emission_factor_coal = 0.95; // tCO2/MWh_e

    double clean_spark_spread(double power, double gas, double carbon) const {
        return power - (gas / efficiency_gas) - emission_factor_gas * carbon;
    }

    double clean_dark_spread(double power, double coal, double carbon) const {
        return power - (coal / efficiency_coal) - emission_factor_coal * carbon;
    }

    double fuel_switching_parity_price(double gas, double coal) const {
        double gas_fuel = gas / efficiency_gas;
        double coal_fuel = coal / efficiency_coal;
        double diff_ef = emission_factor_coal - emission_factor_gas;
        return (gas_fuel - coal_fuel) / diff_ef;
    }

    double greenium_bps(double vanilla_yield, double green_yield) const {
        return (vanilla_yield - green_yield) * 10000.0;
    }
};

} // namespace nexus_quant::macro
