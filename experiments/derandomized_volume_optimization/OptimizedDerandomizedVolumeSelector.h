#ifndef OPTIMIZED_DERANDOMIZED_VOLUME_SELECTOR_H
#define OPTIMIZED_DERANDOMIZED_VOLUME_SELECTOR_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <MatSubset/SelectorBase.h>
#include <MatSubset/Utils.h>

namespace MatSubset {

/*!
 * @brief Optimized version of DerandomizedVolumeSelector (work in progress).
 *
 * Must produce identical results to DerandomizedVolumeSelector for the same
 * inputs. Only the runtime should differ.
 */
template <typename Scalar>
class OptimizedDerandomizedVolumeSelector : public SelectorBase<Scalar> {
  public:
    explicit OptimizedDerandomizedVolumeSelector(
        Scalar tolerance = std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        : tolerance(tolerance) {}

    std::string getAlgorithmName() const override {
        return "optimized derandomized volume";
    }

  protected:
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     Eigen::Index *swap_count) override {
        // Initialization
        Eigen::Index m = X.rows();
        Eigen::Index n = X.cols();

        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        std::vector<Eigen::Index> selected_indices;
        std::vector<Eigen::Index> remaining_indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            remaining_indices[static_cast<size_t>(j)] = j;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver;
        Eigen::VectorX<Scalar> lambda = Eigen::VectorX<Scalar>::Zero(m);
        Eigen::Index r = 0;

        // Main loop
        for (Eigen::Index t = 0; t < k; ++t) {
            // Deflation
            const bool r_less_m = (r < m);
            Eigen::Index x_minus_1_deg = n - m - t - 1;
            assert(x_minus_1_deg >= -1 &&
                   "x_minus_1_deg can't be less than -1");

            auto [start, len] = deflateAndCheckOverflow(lambda, x_minus_1_deg);
            Eigen::VectorX<Scalar> lambda_deflated = lambda.segment(start, len);
            Eigen::MatrixX<Scalar> Q_deflated(len + r_less_m, n - t);
            Q_deflated.bottomRows(len) = Q.middleRows(start, len);
            if (r_less_m) {
                Q_deflated.row(0) = Q.topRows(start).colwise().norm();
            }

            // Target degrees
            const Eigen::Index min_deg_p = n - k - 1 - r_less_m;
            const Eigen::Index max_deg_p = n - k;

            const Eigen::Index min_deg_g = n - k - 1;
            const Eigen::Index max_deg_g = n - k;

            // Construction of auxiliary polynomials
            Eigen::VectorX<Scalar> lambda_inv = lambda_deflated.cwiseInverse();
            Eigen::VectorX<Scalar> p_0 = buildPolynomialFromRoots(lambda_inv);
            Eigen::MatrixX<Scalar> g = buildQuotentPolynomials(lambda_inv);

            // Computation of characteristic polynomial coefficients
            Eigen::MatrixX<Scalar> p(2, n - t);
            if (x_minus_1_deg >= 0) {
                p_0 = applyBinomials(p_0, x_minus_1_deg, min_deg_p, max_deg_p);
                g = applyBinomials(g, x_minus_1_deg, min_deg_g, max_deg_g);

                Eigen::MatrixX<Scalar> Lambda_inv_Q_squared =
                    lambda_inv.asDiagonal() *
                    Q_deflated.bottomRows(len).cwiseAbs2();
                Eigen::RowVectorX<Scalar> p_factor =
                    1 + Lambda_inv_Q_squared.colwise().sum().array();
                Eigen::RowVectorX<Scalar> p_x_factor =
                    Eigen::RowVectorX<Scalar>::Zero(n - t);
                if (r_less_m) {
                    p_x_factor = -Q_deflated.row(0).cwiseAbs2();
                }
                Eigen::MatrixX<Scalar> g_factor =
                    lambda_inv.asDiagonal() * Lambda_inv_Q_squared;

                p = p_0.tail(2) * p_factor + p_0.head(2) * p_x_factor +
                    g * g_factor;
            } else {
                p_0 = p_0.segment(min_deg_g, 2); // min_deg_g is intentional
                g = g.middleRows(min_deg_g, 2);

                Eigen::RowVectorX<Scalar> p_factor =
                    -Q_deflated.row(0).cwiseAbs2();
                Eigen::MatrixX<Scalar> g_factor =
                    (lambda_inv.array() / (1 - lambda_deflated.array()))
                        .matrix()
                        .asDiagonal() *
                    Q_deflated.bottomRows(len).cwiseAbs2();

                p = p_0 * p_factor + g * g_factor;
            }

            Eigen::ArrayX<Scalar> ratios =
                (p.row(0).array() / p.row(1).array()).abs();
            Eigen::Index s;
            ratios.minCoeff(&s);

            Eigen::VectorX<Scalar> q_s = Q.col(s);
            selected_indices.push_back(
                remaining_indices[static_cast<size_t>(s)]);
            if (static_cast<Eigen::Index>(remaining_indices.size()) - 1 != s) {
                remaining_indices[s] = remaining_indices.back();
                Q.col(s) = Q.col(Q.cols() - 1);
            }
            remaining_indices.pop_back();
            Q.conservativeResize(Eigen::NoChange, Q.cols() - 1);

            Eigen::MatrixX<Scalar> M = q_s * q_s.transpose();
            M.diagonal() += lambda;
            eigensolver.compute(M);
            lambda = eigensolver.eigenvalues();
            Q = eigensolver.eigenvectors().transpose() * Q;

            r = (lambda.array() > tolerance).count();
        }

        return selected_indices;
    }

  private:
    Scalar tolerance;

    Eigen::VectorX<Scalar>
    buildPolynomialFromRoots(const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index deg = roots.size();
        Eigen::VectorX<Scalar> p = Eigen::VectorX<Scalar>::Zero(deg + 1);
        p(deg) = static_cast<Scalar>(1);

        for (Eigen::Index i = 1; i <= deg; ++i) {
            p.segment(deg - i, i) -= roots(i - 1) * p.tail(i).eval();
        }

        return p;
    }

    std::pair<Eigen::Index, Eigen::Index>
    deflateAndCheckOverflow(const Eigen::VectorX<Scalar> &lambda,
                            Eigen::Index &x_minus_1_deg) const {
        Eigen::Index start = 0;

        while (start < lambda.size() && lambda[start] < tolerance) {
            start++;
        }

        Eigen::Index ones_start = start;
        while (ones_start < lambda.size() &&
               lambda[ones_start] < Scalar(1) - tolerance) {
            ones_start++;
        }
        x_minus_1_deg += lambda.size() - ones_start;

        const Eigen::Index len = ones_start - start;

        const Scalar max_log_val =
            std::log10(std::numeric_limits<Scalar>::max());
        const Scalar safety_buffer =
            static_cast<Scalar>(len + 1) / 3 + std::log10(tolerance);
        const Scalar safe_limit = max_log_val - safety_buffer;

        Scalar log_magnitude = 0;
        for (Eigen::Index i = start; i < start + len; ++i) {
            log_magnitude += std::log10(1 / lambda[i]);
        }

        if (log_magnitude > safe_limit) {
            std::cerr << "Warning: Predicted overflow! Log magnitude sum: "
                      << log_magnitude << std::endl;
        }

        return {start, len};
    }

    Eigen::MatrixX<Scalar>
    buildQuotentPolynomials(const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index num_roots = roots.size();
        const Eigen::Index g_deg = num_roots - 1;
        Eigen::MatrixX<Scalar> g(num_roots, num_roots);

        for (Eigen::Index i = 0; i < num_roots; ++i) {
            Eigen::VectorX<Scalar> other_roots(g_deg);
            other_roots.head(i) = roots.head(i);
            other_roots.tail(g_deg - i) = roots.tail(g_deg - i);

            g.col(i) = buildPolynomialFromRoots(other_roots);
        }

        return g;
    }

    Eigen::MatrixX<Scalar> applyBinomials(const Eigen::MatrixX<Scalar> &poly,
                                          Eigen::Index x_minus_1_deg,
                                          Eigen::Index min_deg,
                                          Eigen::Index max_deg) const {
        const Eigen::Index input_deg = poly.rows() - 1;

        const Eigen::Index min_binom_deg =
            std::max(min_deg - input_deg, static_cast<Eigen::Index>(0));
        const Eigen::Index max_binom_deg = std::min(max_deg, x_minus_1_deg);
        const Eigen::Index num_coeffs = max_binom_deg - min_binom_deg + 1;
        
        Eigen::MatrixX<Scalar> poly_new =
            Eigen::MatrixX<Scalar>::Zero(max_deg - min_deg + 1, poly.cols());
        // No valid coefficients: poly is empty, or the requested output
        // degree range lies entirely outside [0, input_deg + x_minus_1_deg]
        if (poly.rows() == 0 || num_coeffs <= 0) {
            return poly_new;
        }

        Eigen::VectorX<Scalar> binoms(num_coeffs);
        binoms(0) = 1;
        for (Eigen::Index i = 1; i < num_coeffs; ++i) {
            const Scalar idx = static_cast<Scalar>(max_binom_deg - i);
            binoms(i) = -binoms(i - 1) * (idx + 1) / (x_minus_1_deg - idx);
        }

        for (Eigen::Index i = min_deg; i <= max_deg; ++i) {
            Eigen::Index shift = num_coeffs + min_binom_deg - i - 1;
            Eigen::Index j_start =
                std::max(static_cast<Eigen::Index>(0), -shift);
            Eigen::Index j_end = std::min(input_deg, binoms.size() - shift - 1);
            Eigen::Index len = j_end - j_start + 1;

            if (len >= 1) {
                poly_new.row(i - min_deg) =
                    binoms.segment(j_start + shift, len).transpose() *
                    poly.middleRows(j_start, len);
            }
        }
        return poly_new;
    }
};

} // namespace MatSubset

#endif // OPTIMIZED_DERANDOMIZED_VOLUME_SELECTOR_H
