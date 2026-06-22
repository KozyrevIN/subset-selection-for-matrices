#ifndef MAT_SUBSET_LEVERAGE_SCORES_SELECTOR_H
#define MAT_SUBSET_LEVERAGE_SCORES_SELECTOR_H

#include <algorithm> // For std::sort
#include <mutex>     // For std::lock_guard
#include <random>    // For std::uniform_real_distribution

#include <Eigen/QR> // For Eigen::HouseholderQR

#include "RandomizedBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Selector that samples \f$ k \f$ columns from the input matrix using
 * leverage score probabilities.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements a randomized algorithm that selects columns with
 * probability proportional to the squared norm of th corresponding column in an
 * orthonormal factor of X.
 *
 * @note The `selectSubset()` method is thread-safe and can be called
 * concurrently from multiple threads on the same instance. The RNG state is
 * protected by a mutex to ensure reproducible sequences.
 */
template <typename Scalar>
class LeverageScoresSelector : public RandomizedBase<Scalar> {
  public:
    /*!
     * @brief Constructor with a random seed.
     *
     * This constructor uses a high-quality random seed from the system's
     * random device.
     */
    LeverageScoresSelector() : RandomizedBase<Scalar>() {}

    /*!
     * @brief Constructor with a specified seed for reproducibility.
     * @param seed Seed for the random number generator.
     *
     * Using the same seed will produce the same sequence of column selections
     * across multiple runs, enabling reproducible experiments.
     */
    explicit LeverageScoresSelector(std::mt19937::result_type seed)
        : RandomizedBase<Scalar>(seed) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "leverage scores".
     */
    std::string getAlgorithmName() const override { return "leverage scores"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns
     * using leverage scores.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns (without replacement).
     *
     * This method computes the leverage scores via LQ and then samples k
     * distinct columns according to weighted sampling without replacement using
     * the Efraimidis-Spirakis algorithm.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k,
                                               Eigen::Index *swap_count) override {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Compute the LQ of X
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        // Compute leverage scores
        std::vector<Scalar> leverage_scores(static_cast<size_t>(n));
        for (Eigen::Index j = 0; j < n; ++j) {
            leverage_scores[static_cast<size_t>(j)] = Q.col(j).squaredNorm();
        }

        // Lock mutex for thread-safe access to RNG
        std::lock_guard<std::mutex> lock(this->gen_mutex);

        // Use Efraimidis-Spirakis algorithm for weighted sampling without
        // replacement: For each item with weight w, generate key = u^(1/w)
        // where u ~ Uniform(0,1) Then select k items with largest keys
        std::uniform_real_distribution<Scalar> dist(static_cast<Scalar>(0),
                                                    static_cast<Scalar>(1));

        std::vector<std::pair<Scalar, Eigen::Index>> keys;
        keys.reserve(static_cast<size_t>(n));

        for (Eigen::Index j = 0; j < n; ++j) {
            Scalar u = dist(this->gen);
            Scalar key =
                std::exp(std::log(u) / leverage_scores[static_cast<size_t>(j)]);
            keys.emplace_back(key, j);
        }

        // Partial sort to get the k largest keys
        std::partial_sort(keys.begin(), keys.begin() + k, keys.end(),
                          [](const std::pair<Scalar, Eigen::Index> &a,
                             const std::pair<Scalar, Eigen::Index> &b) {
                              return a.first > b.first;
                          });

        // Extract the indices of the k selected columns
        std::vector<Eigen::Index> selected_indices(static_cast<size_t>(k));
        for (Eigen::Index i = 0; i < k; ++i) {
            selected_indices[static_cast<size_t>(i)] =
                keys[static_cast<size_t>(i)].second;
        }

        return selected_indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_LEVERAGE_SCORES_SELECTOR_H