#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace nexus_quant::rigor {

class DeflatedSharpeCalculator {
public:
    static double expected_max_sharpe(double num_trials, double var_sharpe = 0.25, double mean_sharpe = 0.0) {
        if (num_trials <= 1.0) return mean_sharpe;
        double gamma_euler = 0.5772156649;
        double s = std::sqrt(var_sharpe);
        double p1 = 1.0 - (1.0 / num_trials);
        double p2 = 1.0 - (1.0 / (num_trials * std::exp(1.0)));
        double z1 = core::inv_normal_cdf(std::clamp(p1, 1e-6, 1.0 - 1e-6));
        double z2 = core::inv_normal_cdf(std::clamp(p2, 1e-6, 1.0 - 1e-6));
        return mean_sharpe + s * ((1.0 - gamma_euler) * z1 + gamma_euler * z2);
    }

    static double probabilistic_sharpe_ratio(double observed_sr, double benchmark_sr, double sample_length,
                                             double skew = -0.5, double kurt = 4.0) {
        if (sample_length <= 2.0) return 0.5;
        double s_sr = std::sqrt((1.0 - skew * observed_sr + 0.25 * (kurt - 1.0) * observed_sr * observed_sr) / (sample_length - 1.0));
        if (s_sr <= 1e-8) return 1.0;
        double z = (observed_sr - benchmark_sr) / s_sr;
        return core::normal_cdf(z);
    }

    static double deflated_sharpe_ratio(double observed_sr, double num_trials, double sample_length,
                                        double var_sharpe = 0.25, double skew = -0.5, double kurt = 4.0) {
        double sr_star = expected_max_sharpe(num_trials, var_sharpe, 0.0);
        return probabilistic_sharpe_ratio(observed_sr, sr_star, sample_length, skew, kurt);
    }
};

} // namespace nexus_quant::rigor

#pragma once
