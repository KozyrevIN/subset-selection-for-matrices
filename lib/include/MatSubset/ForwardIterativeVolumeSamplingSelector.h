#ifndef MAT_SUBSET_FORWARD_ITERATIVE_VOLUME_SAMPLING_SELECTOR_H
#define MAT_SUBSET_FORWARD_ITERATIVE_VOLUME_SAMPLING_SELECTOR_H

#include <algorithm> // For std::shuffle
#include <mutex>     // For std::lock_guard
#include <random> // For std::uniform_int_distribution, std::uniform_real_distribution, std::bernoulli_distribution
#include <utility> // For std::pair
#include <vector>  // For std::vector

#include <Eigen/QR> // For Eigen::HouseholderQR

#include "RandomizedBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Selector that samples subsets of size k with probabilities
 * proportional to their squared volume.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements a two-stage randomized algorithm:
 * 1. Uses the ARP algorithm from Cortinovis and Cressner (2026) "Adaptive
 *    Randomized Pivoting for Column Subset Selection, DEIM, and Low-Rank
 *    Approximation" to select the first \f$ m \f$ columns.
 * 2. If \f$ k > m \f$, selects the remaining \f$ k - m \f$ columns uniformly
 *    at random from the columns not yet selected.
 *
 * @note The `selectSubset()` method is thread-safe and can be called
 * concurrently from multiple threads on the same instance. The RNG state is
 * protected by a mutex to ensure reproducible sequences.
 */
template <typename Scalar>
class ForwardIterativeVolumeSamplingSelector : public RandomizedBase<Scalar> {
  public:
    /*!
     * @brief Constructor with a random seed.
     *
     * This constructor uses a high-quality random seed from the system's
     * random device.
     */
    ForwardIterativeVolumeSamplingSelector() : RandomizedBase<Scalar>() {}

    /*!
     * @brief Constructor with a specified seed for reproducibility.
     * @param seed Seed for the random number generator.
     *
     * Using the same seed will produce the same sequence of column selections
     * across multiple runs, enabling reproducible experiments.
     */
    explicit ForwardIterativeVolumeSamplingSelector(
        std::mt19937::result_type seed)
        : RandomizedBase<Scalar>(seed) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "forward iterative volume sampling".
     */
    std::string getAlgorithmName() const override {
        return "forward iterative volume sampling";
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

        // Here first m indices are selected
        std::vector<Eigen::Index> indices = arpAlgorithm(X);

        // If k > m, select remaining k - m columns uniformly at random
        if (k > m) {
            std::lock_guard<std::mutex> lock(this->gen_mutex);
            std::shuffle(indices.begin() + m, indices.end(), this->gen);
        }

        indices.resize(static_cast<size_t>(k));
        return indices;
    }

  private:
    /*!
     * @brief Adaptive Randomized Pivoting (ARP) algorithm.
     * @param X The input matrix (dimensions \f$ m \times n \f$).
     * @return A pair containing:
     *   - first: `std::vector` of selected column indices (size \f$ m \f$)
     *   - second: `std::vector` of remaining column indices (size \f$ n - m
     * \f$)
     *
     * This method implements the ARP algorithm from Cortinovis and Cressner
     * (2026). The algorithm adaptively selects columns using randomized
     * pivoting to construct a well-conditioned basis.
     */
    std::vector<Eigen::Index> arpAlgorithm(const Eigen::MatrixX<Scalar> &X) {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Compute the LQ of X
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        // Prepare index set
        std::vector<Eigen::Index> indices(static_cast<size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            indices[static_cast<size_t>(i)] = i;
        }

        // Pre-allocate full orthonormal basis matrix (m × m)
        Eigen::MatrixX<Scalar> B_full(m, m);

        std::uniform_real_distribution<Scalar> uniform_dist(
            static_cast<Scalar>(0), static_cast<Scalar>(1));

        for (Eigen::Index t = 0; t < m; t++) {
            // B is a view of the first t columns (already selected)
            auto B = B_full.leftCols(t);

            Eigen::Index j;
            bool accepted = false;
            while (!accepted) {
                {
                    std::lock_guard<std::mutex> lock(this->gen_mutex);
                    std::uniform_int_distribution<Eigen::Index> index_dist(
                        t, n - 1);
                    j = index_dist(this->gen);
                }

                // Project q_j onto orthogonal complement: (I - BB^T) q_j
                Eigen::VectorX<Scalar> q_j_ort = Q.col(j);
                if (t > 0) {
                    q_j_ort -= B * (B.transpose() * q_j_ort);
                }

                // Compute acceptance probability p_j = ||q_j_ort||^2
                Scalar p_j = q_j_ort.squaredNorm();

                // Accept with probability p_j (where p_j <= 1)
                {
                    std::lock_guard<std::mutex> lock(this->gen_mutex);
                    std::bernoulli_distribution bernoulli(p_j);
                    accepted = bernoulli(this->gen);
                }
            }

            // Update orthonormal basis using Modified Gram-Schmidt
            Eigen::VectorX<Scalar> q_j_ort = Q.col(j);
            for (Eigen::Index i = 0; i < t; i++) {
                q_j_ort -= B_full.col(i).dot(q_j_ort) * B_full.col(i);
            }

            // Normalize and store in column t (no resize needed!)
            Scalar norm = q_j_ort.norm();
            B_full.col(t) = q_j_ort / norm;

            // Swap selected index to position t
            std::swap(indices[static_cast<size_t>(j)],
                      indices[static_cast<size_t>(t)]);
            Q.col(j) = Q.col(t);
        }

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FORWARD_ITERATIVE_VOLUME_SAMPLING_SELECTOR_H
