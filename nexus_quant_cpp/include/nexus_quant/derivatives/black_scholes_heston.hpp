#pragma once
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

namespace nexus_quant::derivatives {

struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

class BlackScholesModel {
public:
    static double call_price(double S, double K, double T, double r, double sigma, double q = 0.0) {
        if (T <= 0.0) return std::max(S - K, 0.0);
        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        return S * std::exp(-q * T) * core::normal_cdf(d1) - K * std::exp(-r * T) * core::normal_cdf(d2);
    }

    static double put_price(double S, double K, double T, double r, double sigma, double q = 0.0) {
        if (T <= 0.0) return std::max(K - S, 0.0);
        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        return K * std::exp(-r * T) * core::normal_cdf(-d2) - S * std::exp(-q * T) * core::normal_cdf(-d1);
    }

    static Greeks calculate_greeks(double S, double K, double T, double r, double sigma, double q = 0.0) {
        if (T <= 1e-6) return {1.0, 0.0, 0.0, 0.0, 0.0};
        double sqrt_T = std::sqrt(T);
        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
        double d2 = d1 - sigma * sqrt_T;
        double pdf_d1 = core::normal_pdf(d1);
        double disc_q = std::exp(-q * T);
        double disc_r = std::exp(-r * T);

        double delta = disc_q * core::normal_cdf(d1);
        double gamma = (disc_q * pdf_d1) / (S * sigma * sqrt_T);
        double vega = S * disc_q * sqrt_T * pdf_d1 * 0.01; // 1% change
        double theta = (- (S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
                        - r * K * disc_r * core::normal_cdf(d2)
                        + q * S * disc_q * core::normal_cdf(d1)) / 365.0; // 1-day decay
        double rho = (K * T * disc_r * core::normal_cdf(d2)) * 0.01;
        return {delta, gamma, vega, theta, rho};
    }
};

class HestonCOSPricer {
public:
    static std::complex<double> characteristic_function(std::complex<double> u, double S0, double T,
                                                        double r, double q, double v0, double kappa,
                                                        double theta, double xi, double rho) {
        std::complex<double> i(0.0, 1.0);
        std::complex<double> d = std::sqrt(std::pow(kappa - i * rho * xi * u, 2.0) + (xi * xi) * (u * u + i * u));
        std::complex<double> g = (kappa - i * rho * xi * u - d) / (kappa - i * rho * xi * u + d);
        std::complex<double> exp_dT = std::exp(-d * T);
        std::complex<double> C = (r - q) * i * u * T + (kappa * theta / (xi * xi)) *
            ((kappa - i * rho * xi * u - d) * T - 2.0 * std::log((1.0 - g * exp_dT) / (1.0 - g)));
        std::complex<double> D = ((kappa - i * rho * xi * u - d) / (xi * xi)) * ((1.0 - exp_dT) / (1.0 - g * exp_dT));
        return std::exp(C + D * v0 + i * u * std::log(S0));
    }

    static double price_call(double S0, double K, double T, double r, double q,
                             double v0, double kappa, double theta, double xi, double rho, int N = 128, double L = 8.0) {
        std::complex<double> i(0.0, 1.0);
        double x = std::log(S0 / K);
        double c1 = (r - q) * T + (1.0 - std::exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T + x;
        double c2 = (1.0 / (8.0 * std::pow(kappa, 3.0))) * (
            xi * T * kappa * std::exp(-kappa * T) * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
            + kappa * rho * xi * (1.0 - std::exp(-kappa * T)) * (16.0 * theta - 8.0 * v0)
            + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * xi + xi * xi + 4.0 * kappa * kappa)
            + xi * xi * ((theta - 2.0 * v0) * std::exp(-2.0 * kappa * T) + theta * (6.0 * std::exp(-kappa * T) - 7.0) + 2.0 * v0)
            + 8.0 * kappa * kappa * (v0 - theta) * (1.0 - std::exp(-kappa * T))
        );
        c2 = std::max(1e-4, std::abs(c2));
        double a = c1 - L * std::sqrt(c2);
        double b = c1 + L * std::sqrt(c2);

        auto chi_k = [&](int k_idx, double c, double d_val) {
            double omega = k_idx * core::PI / (b - a);
            return (std::cos(omega * (d_val - a)) * std::exp(d_val) - std::cos(omega * (c - a)) * std::exp(c)
                    + omega * std::sin(omega * (d_val - a)) * std::exp(d_val) - omega * std::sin(omega * (c - a)) * std::exp(c)) / (1.0 + omega * omega);
        };

        auto psi_k = [&](int k_idx, double c, double d_val) {
            double omega = k_idx * core::PI / (b - a);
            if (k_idx == 0) return d_val - c;
            return (std::sin(omega * (d_val - a)) - std::sin(omega * (c - a))) / omega;
        };

        double sum_terms = 0.0;
        for (int k = 0; k < N; ++k) {
            double u_k = k * core::PI / (b - a);
            std::complex<double> phi_y = characteristic_function(std::complex<double>(u_k, 0.0), S0, T, r, q, v0, kappa, theta, xi, rho) * std::exp(-i * u_k * std::log(K));
            double H_k = (2.0 / (b - a)) * (chi_k(k, 0.0, b) - psi_k(k, 0.0, b));
            double term = (phi_y * std::exp(-i * u_k * a)).real() * H_k;
            if (k == 0) term *= 0.5;
            sum_terms += term;
        }
        return std::max(K * std::exp(-r * T) * sum_terms, 0.0);
    }
};

inline std::vector<double> extract_breeden_litzenberger_density(
    double S0, const std::vector<double>& strikes, double T, double r, double q, double sigma) {
    size_t n = strikes.size();
    if (n < 3) return {};
    std::vector<double> calls(n);
    for (size_t i = 0; i < n; ++i) calls[i] = BlackScholesModel::call_price(S0, strikes[i], T, r, sigma, q);
    std::vector<double> density(n, 0.0);
    double disc = std::exp(r * T);
    for (size_t i = 1; i < n - 1; ++i) {
        double dK = strikes[i + 1] - strikes[i];
        double d2C = (calls[i + 1] - 2.0 * calls[i] + calls[i - 1]) / (dK * dK);
        density[i] = std::max(disc * d2C, 0.0);
    }
    double total = 0.0;
    for (double d : density) total += d;
    if (total > 0.0) for (double& d : density) d /= total;
    return density;
}

} // namespace nexus_quant::derivatives
