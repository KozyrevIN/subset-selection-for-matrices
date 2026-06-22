#ifndef MAT_SUBSET_REVERSE_ITERATIVE_VOLUME_SAMPLING_SELECTOR_H
#define MAT_SUBSET_REVERSE_ITERATIVE_VOLUME_SAMPLING_SELECTOR_H

#include <algorithm> // For std::find
#include <cmath>     // For std::max
#include <mutex>     // For std::lock_guard
#include <random> // For std::uniform_int_distribution, std::bernoulli_distribution
#include <vector> // For std::vector

#include <Eigen/Dense> // For matrix operations and inverse

#include "RandomizedBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Selector that uses Reverse Iterative Volume Sampling.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements the Fast version of reverse iterative volume
 * sampling algorithm from Derezinsky and Warmuth (2018) "Reverse iterative
 * volume sampling for linear regression". The algorithm starts with all n
 * columns and iteratively removes columns using rejection sampling until the
 * desired subset size is reached.
 *
 * @note The `selectSubset()` method is thread-safe and can be called
 * concurrently from multiple threads on the same instance. The RNG state is
 * protected by a mutex to ensure reproducible sequences.
 */
template <typename Scalar>
class ReverseIterativeVolumeSamplingSelector : public RandomizedBase<Scalar> {
  public:
    /*!
     * @brief Constructor with a random seed.
     *
     * This constructor uses a high-quality random seed from the system's
     * random device.
     */
    ReverseIterativeVolumeSamplingSelector() : RandomizedBase<Scalar>() {}

    /*!
     * @brief Constructor with specified seed for reproducibility.
     * @param seed Seed for the random number generator.
     *
     * Using the same seed will produce the same sequence of column selections
     * across multiple runs, enabling reproducible experiments.
     */
    explicit ReverseIterativeVolumeSamplingSelector(
        std::mt19937::result_type seed)
        : RandomizedBase<Scalar>(seed) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "reverse iterative volume sampling".
     */
    std::string getAlgorithmName() const override {
        return "reverse iterative volume sampling";
    }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k,
                                               Eigen::Index *swap_count) override {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Obtain relevant matrices
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();
        Eigen::MatrixX<Scalar> M = Eigen::MatrixX<Scalar>::Identity(m, m);

        // Initialize index set
        std::vector<Eigen::Index> selected_indices(static_cast<size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            selected_indices[static_cast<size_t>(i)] = i;
        }

        // First phase of iterative removal process
        Eigen::Index target_size = std::max(k, 2 * m);
        for (Eigen::Index t = n; t > target_size; --t) {
            // Rejection sampling loop
            Eigen::Index j;
            bool accepted = false;
            Eigen::VectorX<Scalar> q_j;
            Scalar p_j;
            while (!accepted) {
                {
                    std::lock_guard<std::mutex> lock(this->gen_mutex);
                    std::uniform_int_distribution<Eigen::Index> index_dist(
                        0, t - 1);
                    j = index_dist(this->gen);
                }
                q_j = Q.col(j);
                p_j = static_cast<Scalar>(1) - q_j.dot(M * q_j);
                {
                    std::lock_guard<std::mutex> lock(this->gen_mutex);
                    std::bernoulli_distribution bernoulli(p_j);
                    accepted = bernoulli(this->gen);
                }
            }

            // Update M
            Eigen::VectorX<Scalar> M_q_j = M * q_j;
            M += M_q_j * M_q_j.transpose() / p_j;

            // Remove column j from selected set by swapping with last
            std::swap(selected_indices[static_cast<size_t>(j)],
                      selected_indices[static_cast<size_t>(t - 1)]);
            Q.col(j).swap(Q.col(t - 1));

            selected_indices.resize(static_cast<size_t>(t - 1));
            Q.conservativeResize(Eigen::NoChange, t - 1);
        }

        // For t < 2m rejection sampling is disabled
        if (k < 2 * m) {

            // Pseudoinverse via Q^+ = Q^T (Q Q^T)^{-1} = Q^T M
            Eigen::MatrixX<Scalar> Q_dag = Q.transpose() * M;
            Eigen::ArrayX<Scalar> d =
                static_cast<Scalar>(1) - (Q_dag * Q).diagonal().array();

            for (Eigen::Index t = Q.cols(); t > k; --t) {

                // Sample index to remove proportionally to d
                Eigen::Index j;
                {
                    std::lock_guard<std::mutex> lock(this->gen_mutex);
                    std::discrete_distribution<Eigen::Index> dist(
                        d.data(), d.data() + t);
                    j = dist(this->gen);
                }

                Scalar d_j = d(j);
                Eigen::VectorX<Scalar> q_j = Q.col(j);
                Eigen::VectorX<Scalar> q_dag_j = Q_dag.row(j).transpose();

                // Remove column j from selected set
                d(j) = d(t - 1);
                selected_indices[static_cast<size_t>(j)] =
                    selected_indices[static_cast<size_t>(t - 1)];
                Q.col(j) = Q.col(t - 1);
                Q_dag.row(j) = Q_dag.row(t - 1);

                selected_indices.resize(static_cast<size_t>(t - 1));
                d.conservativeResize(t - 1);
                Q.conservativeResize(Eigen::NoChange, t - 1);
                Q_dag.conservativeResize(t - 1, Eigen::NoChange);

                // Update arrays and matrices
                d -= (Q_dag * q_j).array().square() / d_j;
                Q_dag += (Q.transpose() * q_dag_j) * q_dag_j.transpose() / d_j;
            }
        }

        return selected_indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_REVERSE_ITERATIVE_VOLUME_SAMPLING_SELECTOR_H
