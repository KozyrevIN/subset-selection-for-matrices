#ifndef MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H
#define MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H

#include <cmath> // For std::sqrt

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjoitEigenSolver
#include <Eigen/QR>          // For Eigen::HouseholderQR

#include "SelectorBase.h" // Base class

#include <iostream>

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using a
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
     * @param eps Tolerance for the binary search used in updating parameter
     * \f$ l \f$. Defaults to `1e-6`.
     */
    SpectralSelectionSelector(Scalar eps = 1e-6) : eps_(eps) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "spectral selection".
     */
    std::string getAlgorithmName() const override {

        return "spectral selection";
    }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m);
        Eigen::MatrixX<Scalar> V = Q.transpose();

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

        Scalar epsilon = computeEpsilon0(m, n, k);
        Scalar l_0 = -(static_cast<Scalar>(m) / epsilon);
        Scalar l = l_0;
        const Scalar B0 = computeB(m, n, k, l, 0, S);

        while (cols_selected.size() < k) {

            Scalar l_trial;
            if (m > 1) {

                Scalar l_opt = optimizeB(m, n, k, cols_selected.size(), S);
                Scalar l_min =
                    minimizeL(m, n, k, cols_selected.size(), l_opt, S);

                if (cols_selected.size() < k - m) {
                    l_trial = l_min;
                } else {
                    Scalar lambda =
                        (k - cols_selected.size() - 1) / static_cast<Scalar>(m);
                    l_trial = lambda * l_min + (1 - lambda) * l_opt;
                }
            }

            if ((m == 1) ||
                (computeB(m, n, k, l_trial, cols_selected.size(), S) < B0)) {
                l = computeL(epsilon, l, S);
            } else {
                l = l_trial;
                epsilon = computeEpsilon(l, S);
            }

            Scalar delta =
                computeDelta(m, n, k, l, epsilon, cols_remaining.size());

            Eigen::ArrayX<Scalar> D = (S - (l + delta)).inverse();
            Eigen::MatrixX<Scalar> M_1 = D.matrix().asDiagonal();
            Eigen::MatrixX<Scalar> M_2 = D.square().matrix().asDiagonal();
            Eigen::MatrixX<Scalar> M_3 = U.transpose() * V;
            Eigen::ArrayX<Scalar> Phi =
                -(M_3.transpose() * M_2 * M_3).diagonal().array() /
                (1 + (M_3.transpose() * M_1 * M_3).diagonal().array());

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
        }
        return cols_selected;
    }

    /*!
     * @brief Calculates theoretical lower bounds for the spectral selection
     * strategy.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @param norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        if (m == 1) {
            return static_cast<Scalar>(k) / static_cast<Scalar>(n);
        } else {
            const Scalar alpha = std::sqrt(static_cast<Scalar>(k - 1) * m + 1);
            const Scalar base = (k - alpha) / (alpha - 1);
            return (static_cast<Scalar>(m) / n) * base * base;
        }
    }

  private:
    Scalar eps_; // Relative tolerance for binary and golden-section search

    /*!
     * @brief Calculates the initial value of parameter \f$ \epsilon \f$.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @return The calculated \f$ \epsilon \f$ value.
     */
    Scalar computeEpsilon0(Eigen::Index m, Eigen::Index n,
                           Eigen::Index k) const {

        Scalar epsilon;
        if (m == 1) {
            epsilon = static_cast<Scalar>(0.5);
        } else {
            Scalar S_n = static_cast<Scalar>(n);
            Scalar S_m = static_cast<Scalar>(m);
            Scalar S_k = static_cast<Scalar>(k);

            Scalar alpha = std::sqrt((S_k - 1) * S_m + 1);
            epsilon = n *
                      (2 * (alpha - 1) +
                       S_m * (S_k * (alpha + S_m - 2) - 2 * alpha - S_m + 3)) /
                      ((S_k - 1) * S_m * (S_k - S_m + 1));
        }
        return epsilon;
    }

    /*!
     * @brief Calculates the value of the parameter \f$ \epsilon \f$ depending
     * on barrier level \f$ l \f$.
     * @param l The value of parameter \f$ l \f$.
     * @param eigenvalues Eiganvalues of matrix \f$ Y \f$.
     * @return The calculated value of \f$ \epsilon \f$.
     */
    Scalar computeEpsilon(Scalar l,
                          const Eigen::ArrayX<Scalar> &eigenvalues) const {

        assert(l < eigenvalues(0) &&
               "l must be smaller then the smallest eigenvalue");
        return (eigenvalues - l).inverse().sum();
    }

    /*! @brief Calculates the parameter \f$ \delta \f$.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @param l Current value of parameter \f$ l \f$.
     * @param epsilon Current value of \f$ \epsilon \f$.
     * @param cols_remaining_size Number of columns currently remaining.
     * @return The calculated \f$ \delta \f$ value.
     * @note Solves a quadratic equation. Assumes discriminant \f$ D \ge 0 \f$,
     * assumes `cols_remaining_size > 0`. Both those conditions provably hold.
     */
    Scalar computeDelta(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                        Scalar l, Scalar epsilon,
                        Eigen::Index cols_remaining_size) const {

        Scalar S_n = static_cast<Scalar>(n);
        Scalar S_m = static_cast<Scalar>(m);
        Scalar S_k = static_cast<Scalar>(k);
        Scalar S_cols_remaining_size = static_cast<Scalar>(cols_remaining_size);

        Scalar a = epsilon / S_m;
        Scalar b = -static_cast<Scalar>(1) -
                   epsilon * (1 - l - S_m / epsilon) / S_cols_remaining_size;
        Scalar c = (1 - l - S_m / epsilon) / S_cols_remaining_size;

        Scalar D = b * b - 4 * a * c;
        return (-b - std::sqrt(D)) / (2 * a);
    }

    /*! @brief Performs binary search to find a root of \f$ f(l) = 0
     * \f$ within `[left, right]`.
     * @param left Lower bound of the search interval.
     * @param right Upper bound of the search interval.
     * @param f The function \f$ f(\lambda) \f$ whose root is sought.
     * @return The approximate root.
     * @note Assumes \f$ f(l) \f$ and \f$ f(r) \f$ have opposite signs. Assumes
     * \f$ l \le r \f$.
     */
    Scalar bisectionMethod(Scalar left, Scalar right,
                           const std::function<Scalar(Scalar)> &f) const {

        assert(left <= right && "The search interval must be non-empty.");
        Scalar f_left = f(left);
        Scalar f_right = f(right);

        const int MAX_ITERATIONS = 100;
        int iter = 0;
        while ((right - left) > eps_ && iter < MAX_ITERATIONS) {
            Scalar mid = (left + right) / static_cast<Scalar>(2);
            Scalar f_mid = f(mid);

            if (f_mid > 0) {
                right = mid;
            } else {
                left = mid;
            }
        }

        return (left + right) / static_cast<Scalar>(2);
    }

    /*!
     * @brief Calculates the value of the parameter \f$ l \f$ that gives
     * specified value of \f$ \epsilon \f$ using bisection method.
     * @param epsilon The value of parameter \f$ \epsilon \f$.
     * @param l_prev Previous value of \f$ l \f$, serving as a lower bound.
     * @param eigenvalues Eiganvalues of matrix \f$ Y \f$.
     * @return The calculated value of \f$ l \f$.
     */
    Scalar computeL(Scalar epsilon, Scalar l_prev,
                    const Eigen::ArrayX<Scalar> &eigenvalues) const {

        auto f = [&eigenvalues, &epsilon](Scalar l) {
            return (eigenvalues - l).inverse().sum() - epsilon;
        };

        return bisectionMethod(l_prev, eigenvalues(0), f);
    }

    /*! @brief Computes \f$ B_i(l) \f$, guaranteed lower bound for the final
     * value of \f$ \lambda_m(Y) \f$ from the current intermediate state,
     * defined by parameter \f$ l \f$ and eigenvalues of matrix \f$ Y \f$.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @param l Current value of parameter \f$ l \f$.
     * @param cols_selected_size Number of columns currently selected.
     * @param eigenvalues Eiganvalues of matrix \f$ Y \f$.
     * @return The calculated \f$ B_i(l) \f$ value.
     * @note Part of the heuristic update strategy for \f$ l \f$, meant to
     * enhance practical performance. It is backed up by a principled approach.
     */
    Scalar computeB(Eigen::Index m, Eigen::Index n, Eigen::Index k, Scalar l,
                    Eigen::Index cols_selected_size,
                    const Eigen::ArrayX<Scalar> &eigenvalues) const {

        Scalar epsilon = computeEpsilon(l, eigenvalues);
        Scalar delta =
            computeDelta(m, n, k, l, epsilon, n - cols_selected_size);
        return l + (k - cols_selected_size) * delta + 1 / epsilon;
    }

    /*! @brief Identifies \f$ l_{opt} \f$, candidate maximizer of \f$ B_i(l) \f$
     * using golden-section search.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @param cols_selected_size Number of columns currently selected.
     * @param eigenvalues Eiganvalues of matrix \f$ Y \f$.
     * @return The calculated \f$ \delta \f$ value.
     * @note Part of the heuristic update strategy for \f$ l \f$, meant to
     * enhance practical performance. It is backed up by a principled approach.
     * @note We do not prove that \f$ B_i(l) \f$ is unimodal, although it was
     * consistently well-behaved in experiments. There is no guarantee that the
     * algorithtm will find the maximizer, and this is fine, since this is a
     * part of the heuristic approach.
     */
    Scalar optimizeB(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Eigen::Index cols_selected_size,
                     const Eigen::ArrayX<Scalar> &eigenvalues) const {

        assert(m > 1 &&
               "In the heuristic update strategy, m must be greater then 1.");
        Scalar left = -static_cast<Scalar>(m - 1) / static_cast<Scalar>(m + 1);
        Scalar right = eigenvalues(0);
        const Scalar GOLDEN_RATIO = (1 + std::sqrt(5)) / static_cast<Scalar>(2);

        Scalar l1 = right - (right - left) / GOLDEN_RATIO;
        Scalar l2 = left + (right - left) / GOLDEN_RATIO;
        Scalar b1 = computeB(m, n, k, l1, cols_selected_size, eigenvalues);
        Scalar b2 = computeB(m, n, k, l2, cols_selected_size, eigenvalues);

        const int MAX_ITERATIONS = 100;
        int iter = 0;
        while ((right - left) > eps_ && iter < MAX_ITERATIONS) {
            if (b1 > b2) {
                // The maximum is in the left interval [left, l2].
                right = l2;
                l2 = l1;
                b2 = b1;
                l1 = right - (right - left) / GOLDEN_RATIO;
                b1 = computeB(m, n, k, l1, cols_selected_size, eigenvalues);
            } else {
                // The maximum is in the right interval [l1, right].
                left = l1;
                l1 = l2;
                b1 = b2;
                l2 = left + (right - left) / GOLDEN_RATIO;
                b2 = computeB(m, n, k, l2, cols_selected_size, eigenvalues);
            }
            iter++;
        }

        return (left + right) / static_cast<Scalar>(2);
    }

    /*! @brief Identifies \f$ l_{min} \f$, candidate minimum \f$ l \f$ such that
     * \f$ B_i(l) \ge B_0 \f$ using bisection method.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of columns to select.
     * @param cols_selected_size Number of columns currently selected.
     * @param l_opt \f$ l_{opt} \f$, which is a candidte maximizer of \f$
     * B_i(l). It servse as a tight boundary.
     * @param eigenvalues Eiganvalues of matrix \f$ Y \f$.
     * @return The calculated \f$ \delta \f$ value.
     * @note Part of the heuristic update strategy for \f$ l \f$, meant to
     * enhance practical performance. It is backed up by a principled approach.
     * @note We do not prove that \f$ B_i(l) \f$ is unimodal, although it was
     * consistently well-behaved in experiments. There is no guarantee that the
     * algorithtm will find this minimal point, and this is fine, since this is
     * a part of the heuristic approach.
     */
    Scalar minimizeL(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Eigen::Index cols_selected_size, Scalar l_opt,
                     const Eigen::ArrayX<Scalar> &eigenvalues) const {

        assert(m > 1 &&
               "In the heuristic update strategy, m must be greater then 1.");
        auto f = [&m, &n, &k, &cols_selected_size, &eigenvalues,
                  this](Scalar l) {
            return this->computeB(m, n, k, l, cols_selected_size, eigenvalues);
        };

        Scalar left = -static_cast<Scalar>(m - 1) / static_cast<Scalar>(m + 1);
        Scalar right = l_opt;

        return bisectionMethod(left, right, f);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H