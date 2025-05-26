#ifndef MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H
#define MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H

#include <cmath> // For std::sqrt, std::pow

#include <Eigen/Eigenvalues>             // For SelfAdjointEigenSolver
#include <Eigen/SVD>                     // For Eigen::
#include <unsupported/Eigen/Polynomials> // For Eigen::PolynomialSolver

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Selects a column subset using an algorithm based on interlacing
 * families of polynomials.
 * @tparam Scalar The underlying Scalar type (e.g., `float`, `double`).
 *
 * This class implements a deterministic greedy selection algorithm inspired by
 * the techniques presented in Xie and Xu (2021), "Subset Selection for Matrices
 * with Fixed Blocks" (Algorithm 1), which itself builds upon the work of
 * Marcus, Spielman, and Srivastava on interlacing families of polynomials.
 *
 * @warning This algorithm is numerically unstable due to its reliance on
 * polynomial root finding and transformations from roots to
 * coefficients and back. It is included primarily for demonstration.
 */
template <typename Scalar>
class InterlacingFamiliesSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `InterlacingFamiliesSelector`.
     * @param eps Tolerance value for polynomial root finding.
     *            Defaults to \f$ 1e-4 \f$.
     */
    InterlacingFamiliesSelector(Scalar eps = 1e-4) : eps_(eps) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "interlacing families".
     */
    std::string getAlgorithmName() const override {
        return "interlacing families";
    }

  protected:
    /*!
     * @brief Core implementation for selecting \f$ k \f$ columns.
     * @param X The \f$ m \times n \f$ input matrix \f$ X \f$.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` of selected column indices.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {

        Eigen::Index m = X.rows();
        Eigen::Index n = X.cols();

        Eigen::BDCSVD svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();
        Eigen::MatrixX<Scalar> Y = Eigen::MatrixX<Scalar>::Zero(m, m);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> decomposition(m);
        Eigen::PolynomialSolver<Scalar, Eigen::Dynamic> poly_solver;

        std::vector<Eigen::Index> cols_remaining(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            cols_remaining[static_cast<size_t>(j)] = j;
        }

        std::vector<Eigen::Index> cols_selected;
        cols_selected.reserve(k);

        Scalar lambda_max_prev = 0;

        for (Eigen::Index i = 1; i <= k; ++i) {
            Eigen::VectorX<Scalar> lambdas(cols_remaining.size());

            Eigen::ArrayX<Scalar> PtoF = PtoFArray(m, n, k, i);
            Eigen::Index f_len = PtoF.size();
            Eigen::MatrixX<Scalar> YtoZ =
                YtoZMatrix(f_len, 1 - lambda_max_prev);

            for (Eigen::Index j = 0; j < cols_remaining.size(); ++j) {
                decomposition.compute(Y + V.col(j) * V.col(j).transpose(),
                                      Eigen::EigenvaluesOnly);
                Eigen::ArrayX<Scalar> p_roots_x = decomposition.eigenvalues();
                // y = x - 1
                Eigen::ArrayX<Scalar> p_roots_y = p_roots_x.array() - 1;
                Eigen::ArrayX<Scalar> p_y = polyFromRoots(p_roots_y);
                Eigen::VectorX<Scalar> f_y = PtoF * p_y.tail(f_len);
                // z = x - \lambda_m(Y) = y + 1 - \lambda_m(Y) obtained on
                // previous step changing the basis to make the problem of
                // finding roots well conditioned
                Eigen::VectorX<Scalar> f_z = YtoZ * f_y;

                poly_solver.compute(f_z);
                bool has_root;
                lambdas(j) = poly_solver.smallestRealRoot(has_root, eps_);
            }

            Eigen::Index j_max;
            lambdas.maxCoeff(&j_max);
            Y += V.col(j_max) * V.col(j_max).transpose();
            lambda_max_prev += lambdas(j_max);

            decomposition.compute(Y);

            cols_selected.push_back(cols_remaining[static_cast<size_t>(j_max)]);
            cols_remaining[static_cast<size_t>(j_max)] = cols_remaining.back();
            cols_remaining.pop_back();
            V.col(j_max) = V.col(V.cols() - 1);
            V.conservativeResize(Eigen::NoChange, V.cols() - 1);
        }

        return cols_selected;
    }

    /*!
     * @brief Calculates the theoretical bound for Interlacing families
     * algorithm.
     * @param m Number of rows in the original matrix (\f$ m \f$).
     * @param n Number of columns in the original matrix (\f$ n \f$).
     * @param k Number of selected columns (\f$ k \f$).
     * @param norm The norm type (`Norm::Frobenius` or `Norm::Spectral`).
     * @return The calculated bound based on Theorem 4.1 of Xie and Xu (2021).
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        Scalar term1_sqrt_arg =
            static_cast<Scalar>(k + 1) * static_cast<Scalar>(n - m);
        Scalar term2_sqrt_arg =
            static_cast<Scalar>(m) * static_cast<Scalar>(n - k - 1);

        Scalar sqrt_term1 = std::sqrt(term1_sqrt_arg);
        Scalar sqrt_term2 = std::sqrt(term2_sqrt_arg);

        Scalar numerator_b = sqrt_term1 - sqrt_term2;
        Scalar denominator_b = static_cast<Scalar>(n);

        Scalar ratio = numerator_b / denominator_b;
        return std::pow(ratio, 2);
    }
};

private:
Scalar eps_;

/*! @brief Constructs a polynomial from its roots.
 *  Polynomial coefficients are returned in an order compatible with Eigen's
 * `PolynomialSolver` (ascending powers: poly(0) is const, poly(l) is coeff
 * of \f$ x^l \f$).
 *  @param roots An `Eigen::VectorX<Scalar>` of the roots of the polynomial.
 *  @return An `Eigen::VectorX<Scalar>` representing the polynomial
 * coefficients.
 */
Eigen::VectorX<Scalar>
polyFromRoots(const Eigen::VectorX<Scalar> &roots) const {

    Eigen::Index l = roots.size();
    Eigen::VectorX<Scalar> poly = Eigen::VectorX<Scalar>::Zero(l + 1);
    poly(l) = 1;

    for (Scalar root : roots) {
        poly.head(l) -= root * poly.tail(l);
    }

    return poly;
}

/*! @brief Calculates coefficients for the \f$ P \to F \f$ polynomial
 * transformation based on Xie and Xu (2021).
 *  @param m Number of rows of original matrix \f$ X \f$.
 *  @param n Number of columns of original matrix \f$ X \f$.
 *  @param k Target number of columns to select.
 *  @param iter Current iteration number (1-based, up to \f$ k \f$).
 *  @return An `Eigen::ArrayX<Scalar>` of transformation coefficients.
 */
Eigen::ArrayX<Scalar> PtoFArray(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                                Eigen::Index iter) const { // Renamed i to iter

    Eigen::ArrayX<Scalar> arr;

    if (k <= n - m) {
        arr = Eigen::ArrayX<Scalar>::Constant(m + 1, static_cast<Scalar>(1.0));
        for (Eigen::Index j = 1; j < m + 1; ++j) {
            Scalar num = static_cast<Scalar>(j + n - m - iter);
            Scalar den = static_cast<Scalar>(j + n - m - k);
            arr(j) = arr(j - 1) * num / den;
        }
    } else { // k > n - m
        Eigen::Index arr_size = n - k + 1;
        arr =
            Eigen::ArrayX<Scalar>::Constant(arr_size, static_cast<Scalar>(1.0));
        for (Eigen::Index j = 1; j < arr_size; ++j) {
            Scalar num = static_cast<Scalar>(j + k - iter);
            Scalar den = static_cast<Scalar>(j);
            arr(j) = arr(j - 1) * num / den;
        }
    }
    return arr;
}

/*! @brief Constructs the \f$ Y \to Z \f$ polynomial transformation matrix
 * for \f$ p(y) \to p(y - \text{shift}) \f$. The resulting matrix `M`
 * transforms coefficient vectors `c_y` of \f$ p(y) \f$ to `c_z` of \f$ p(z)
 * \f$ via `c_z = M * c_y`, where \f$ z = y + \text{shift} \f$ (so \f$ y = z
 * - \text{shift} \f$). Coefficients are assumed to be in ascending order of
 * power.
 *  @param poly_coeffs_len Length of the polynomial coefficient vector
 * (degree + 1). In the algorithm it always equals
 *  @param shift The amount by which the variable is shifted (\f$ z = y +
 * \text{shift} \f$.
 *  @return The transformation `Eigen::MatrixX<Scalar>` `M`.
 */
Eigen::MatrixX<Scalar> YtoZMatrix(Eigen::Index poly_coeffs_len,
                                  Scalar shift) const {

    Eigen::MatrixX<Scalar> M =
        Eigen::MatrixX<Scalar>::Zero(poly_coeffs_len, poly_coeffs_len);

    M(0, 0) = 1;
    for (Eigen::Index i = 1; i < poly_coeffs_len; ++i) {
        M.col(i).tail(poly_coeffs_len - 1) =
            M.col(i - 1).head(poly_coeffs_len - 1);
        M.col(i) -= shift * M.col(i - 1);
    }

    return M;
}
} // namespace MatSubset

#endif // MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H