#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace nexus_quant::core {

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT_2PI = 2.50662827463100050242;

inline double normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double inv_normal_cdf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument("Probability must be in (0, 1)");
    }
    // Beasley-Springer-Moro rational approximation
    double a[4] = { 2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637 };
    double b[4] = { -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833 };
    double c[9] = { 0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                    0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
                    0.0000321767881768, 0.0000002888167364, 0.0000003960315187 };
    double y = p - 0.5;
    if (std::abs(y) < 0.42) {
        double r = y * y;
        return y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) /
                   ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
    }
    double r = p;
    if (y > 0.0) r = 1.0 - p;
    r = std::log(-std::log(r));
    double x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))));
    if (y < 0.0) x = -x;
    return x;
}

class Matrix {
public:
    size_t rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(size_t r, size_t c, double init = 0.0) : rows(r), cols(c), data(r * c, init) {}

    double& operator()(size_t r, size_t c) { return data[r * cols + c]; }
    const double& operator()(size_t r, size_t c) const { return data[r * cols + c]; }

    Matrix transpose() const {
        Matrix res(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                res(j, i) = (*this)(i, j);
        return res;
    }

    static Matrix identity(size_t n) {
        Matrix I(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) I(i, i) = 1.0;
        return I;
    }

    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) throw std::invalid_argument("Matrix dimension mismatch");
        Matrix res(rows, other.cols, 0.0);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                double r_ik = (*this)(i, k);
                for (size_t j = 0; j < other.cols; ++j) {
                    res(i, j) += r_ik * other(k, j);
                }
            }
        }
        return res;
    }

    std::vector<double> multiply_vector(const std::vector<double>& v) const {
        if (cols != v.size()) throw std::invalid_argument("Matrix-vector dimension mismatch");
        std::vector<double> res(rows, 0.0);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                res[i] += (*this)(i, j) * v[j];
            }
        }
        return res;
    }

    Matrix cholesky() const {
        if (rows != cols) throw std::invalid_argument("Must be square for Cholesky");
        Matrix L(rows, cols, 0.0);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < j; ++k) s += L(i, k) * L(j, k);
                if (i == j) {
                    double val = (*this)(i, i) - s;
                    L(i, j) = std::sqrt(std::max(val, 1e-12));
                } else {
                    L(i, j) = ((*this)(i, j) - s) / L(j, j);
                }
            }
        }
        return L;
    }
};

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double variance(const std::vector<double>& v, int ddof = 1) {
    if (v.size() <= static_cast<size_t>(ddof)) return 0.0;
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / (v.size() - ddof);
}

inline double stdev(const std::vector<double>& v, int ddof = 1) {
    return std::sqrt(variance(v, ddof));
}

inline double skewness(const std::vector<double>& v) {
    size_t n = v.size();
    if (n < 3) return 0.0;
    double m = mean(v);
    double s = stdev(v, 1);
    if (s <= 1e-12) return 0.0;
    double sum3 = 0.0;
    for (double x : v) sum3 += std::pow((x - m) / s, 3.0);
    return (sum3 * n) / ((n - 1.0) * (n - 2.0));
}

inline double kurtosis(const std::vector<double>& v, bool excess = true) {
    size_t n = v.size();
    if (n < 4) return 0.0;
    double m = mean(v);
    double s = stdev(v, 1);
    if (s <= 1e-12) return 0.0;
    double sum4 = 0.0;
    for (double x : v) sum4 += std::pow((x - m) / s, 4.0);
    double k = (n * (n + 1.0) * sum4) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    if (excess) {
        k -= (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0));
    }
    return k;
}

} // namespace nexus_quant::core
