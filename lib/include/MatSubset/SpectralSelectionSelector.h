#ifndef MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H
#define MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H

#include <cmath> // For std::sqrt

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjoitEigenSolver
#include <Eigen/QR>          // For Eigen::ColPivHouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Class for approximating subset selection problem for matrices using a
 * novel spectral selection algorithm.
 * @tparam Scalar The underlying Scalar type (e.g., `float`, `double`).
 *
 * This class implements a custom algorithm, developed by our team (A. I.
 * Osinsky, I. N. Kozyrev), which builds upon the theoretical foundations of
 * barrier-based methods for matrix sparsification, similar in spirit to dual
 * set (Avron and Boutsidis, 2012) but with distinct derivations for its update
 * rules and parameters.
 *
 * The `eps_` constructor parameter is a tolerance used within the binary
 * search.
 */
template <typename Scalar>
class SpectralSelectionSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `SpectralSelectionSelector`.
     * @param eps_ Tolerance for the binary search used in updating parameter
     * \f$ l \f$. Defaults to \f$ 1e-6 \f$.
     */
    SpectralSelectionSelector(Scalar eps = 1e-4) : eps_(eps) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "spectral selection".
     */
    std::string getAlgorithmName() const override {

        return "spectral selection";
    }

  protected:
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {
                                                
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q_full = qr.matrixQ(); // Q_full is n x n
        Eigen::MatrixX<Scalar> V = Q_full.leftCols(m).transpose(); // V is m x n

        std::vector<Eigen::Index> cols_remaining(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            cols_remaining[static_cast<size_t>(j)] = j;
        }

        std::vector<Eigen::Index> cols_selected;
        cols_selected.reserve(k);

        Eigen::MatrixX<Scalar> Y = Eigen::MatrixX<Scalar>::Zero(m, m);
        Eigen::MatrixX<Scalar> U =
            Eigen::MatrixX<Scalar>::Identity(m, m); // Initial U
        Eigen::ArrayX<Scalar> S =
            Eigen::ArrayX<Scalar>::Zero(m); // Initial S (eigenvalues of Y=0)

        // Initializes for m x m matrices
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> decomposition(m);

        Scalar epsilon = calculateEpsilon(m, n, k);
        Scalar l_0 = -(static_cast<Scalar>(m) / epsilon);
        Scalar l = l_0;

        while (cols_selected.size() < k) {
            Scalar delta =
                calculateDelta(m, n, k, epsilon, l, cols_remaining.size());

            Eigen::MatrixX<Scalar> M =
                U * (S - (l + delta)).inverse().matrix().asDiagonal() *
                U.transpose() * V;

            Eigen::ArrayX<Scalar> Phi =
                (S - (l + delta)).inverse().sum() -
                M.colwise().squaredNorm().transpose().array() /
                    (static_cast<Scalar>(1.0) +
                     (V.transpose() * M).diagonal().array());

            // Index relative to current V and cols_remaining
            Eigen::Index j_min;
            Phi.minCoeff(&j_min);
            Y += V.col(j_min) * V.col(j_min).transpose();

            cols_selected.push_back(cols_remaining[static_cast<size_t>(j_min)]);

            if (static_cast<Eigen::Index>(cols_remaining.size()) - 1 != j_min) {
                cols_remaining[j_min] = cols_remaining.back();
                V.col(j_min) = V.col(V.cols() - 1);
            }
            cols_remaining.pop_back();
            V.conservativeResize(Eigen::NoChange, V.cols() - 1);

            decomposition.compute(Y);
            U = decomposition.eigenvectors();
            S = decomposition.eigenvalues().array();

            auto f = [&S, &epsilon](Scalar lambda_val) {
                return (S - lambda_val).inverse().sum() - epsilon;
            };

            l = binarySearch(l, S(0), f, delta * eps_);
        }
        return cols_selected;
    }

    /*!
     * @brief Calculates theoretical lower bounds for the DUALSET selection
     * strategy.
     * @param m Number of rows (\f$ m \f$).
     * @param n Number of columns (\f$ n \f$).
     * @param k Number of selected columns (\f$ k \f$).
     * @param norm_type The norm type (`Norm::Frobenius` or `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        Scalar epsilon = calculateEpsilon(m, n, k);
        Scalar l = -(static_cast<Scalar>(m) / epsilon);

        for (Eigen::Index i = 0; i < k; ++i) {
            l += calculateDelta(m, n, k, epsilon, l_bound, n - i);
        }

        return l + static_cast<Scalar>(1.0) / epsilon;
    }

  private:
    Scalar eps_; // Tolerance for binary search

    /*! @brief Calculates the algorithm-specific parameter \f$ \epsilon \f$.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @return The calculated \f$ \epsilon \f$ value.
     */
    Scalar calculateEpsilon(Eigen::Index m, Eigen::Index n,
                            Eigen::Index k) const {

        Scalar epsilon;
        if (m == 1) {
            epsilon = static_cast<Scalar>(0.5);
        } else {
            Scalar S_n = static_cast<Scalar>(n);
            Scalar S_m = static_cast<Scalar>(m);
            Scalar S_k = static_cast<Scalar>(k);

            Scalar S_1 = static_cast<Scalar>(1);
            Scalar S_2 = static_cast<Scalar>(2);
            Scalar S_3 = static_cast<Scalar>(3);

            Scalar alpha = std::sqrt((S_k - S_1) * S_m + S_1);
            epsilon = n *
                      (S_2 * (alpha - S_1) + S_m * (S_k * (alpha + S_m - S_2) -
                                                    S_2 * alpha - S_m + S_3)) /
                      ((S_k - S_1) * S_m * (S_k - S_m + S_1));
        }
        return epsilon;
    }

    /*! @brief Calculates the algorithm-specific parameter \f$ \delta \f$.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @param epsilon Current value of \f$ \epsilon \f$.
     * @param l Current value of parameter \f$ l \f$.
     * @param cols_remaining_size Number of columns currently remaining.
     * @return The calculated \f$ \delta \f$ value.
     * @note Solves a quadratic equation. Assumes discriminant \f$ D \ge 0 \f$,
     * assumes `cols_remaining_size > 0`. Both those conditions are proved
     * to hold.
     */
    Scalar calculateDelta(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                          Scalar epsilon, Scalar l,
                          Eigen::Index cols_remaining_size) const {

        Scalar S_n = static_cast<Scalar>(n);
        Scalar S_m = static_cast<Scalar>(m);
        Scalar S_k = static_cast<Scalar>(k);
        Scalar S_cols_remaining_size = static_cast<Scalar>(cols_remaining_size);

        Scalar S_1 = static_cast<Scalar>(1);
        Scalar S_2 = static_cast<Scalar>(2);
        Scalar S_3 = static_cast<Scalar>(3);

        Scalar a = epsilon / S_m;
        Scalar b =
            -S_1 - epsilon * (1 - l - S_m / epsilon) / S_cols_remaining_size;
        Scalar c = (S_1 - l - S_m / epsilon) / S_cols_remaining_size;

        Scalar D = b * b - 4 * a * c;
        return (-b - std::sqrt(D)) / (2 * a);
    }

    /*! @brief Performs binary search to find a root of \f$ f(\lambda) = 0 \f$
     * within `[l, r]`.
     * @param l Lower bound of the search interval.
     * @param r Upper bound of the search interval.
     * @param f The function \f$ f(\lambda) \f$ whose root is sought.
     * @param tol Tolerance for convergence.
     * @return The approximate root.
     * @note Assumes \f$ f(l) \f$ and \f$ f(r) \f$ have opposite signs, or \f$ f
     * \f$ is monotonic and a root exists in the interval. Assumes \f$ l \le r
     * \f$.
     */
    Scalar binarySearch(Scalar l, Scalar r,
                        const std::function<Scalar(Scalar)> &f,
                        Scalar tol) const {

        assert(l > r && "in binyry search right bound must be >= left bound");
        Scalar f_l = f(l); // Store f(l)
        Scalar f_r = f(r); // Store f(r)

        Scalar current_l = l;
        Scalar current_r = r;

        while (current_r - current_l > tol) {
            Scalar m = (current_r + current_l) / 2.0;
            Scalar f_m = f(m);

            if (f_m > 0) {
                current_r = m;
            } else {
                current_l = m;
            }
        }
        return (current_r + current_l) / 2.0;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H