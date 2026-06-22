#ifndef MAT_SUBSET_RANDOM_COLUMNS_SELECTOR_H
#define MAT_SUBSET_RANDOM_COLUMNS_SELECTOR_H

#include <algorithm> // For std::shuffle
#include <mutex>     // For std::lock_guard

#include "RandomizedBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Selector that randomly selects \f$ k \f$ columns from the input
 * matrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector provides a baseline random selection strategy. It can be used
 * for comparison with more sophisticated selection algorithms or when no
 * particular column preference is needed.
 *
 * The random number generator can be seeded for reproducibility. The selection
 * process is thread-safe when called on the same instance from multiple
 * threads.
 *
 * @note The `selectSubset()` method is thread-safe and can be called
 * concurrently from multiple threads on the same instance. The RNG state is
 * protected by a mutex to ensure reproducible sequences: for a given seed, any
 * sequence of selections will always be identical regardless of thread
 * interleaving.
 */
template <typename Scalar>
class RandomColumnsSelector : public RandomizedBase<Scalar> {
  public:
    /*!
     * @brief Constructor with a random seed.
     *
     * This constructor uses a high-quality random seed from the system's
     * random device.
     */
    RandomColumnsSelector() : RandomizedBase<Scalar>() {}

    /*!
     * @brief Constructor with a specified seed for reproducibility.
     * @param seed Seed for the random number generator.
     *
     * Using the same seed will produce the same sequence of column selections
     * across multiple runs, enabling reproducible experiments.
     */
    explicit RandomColumnsSelector(std::mt19937::result_type seed)
        : RandomizedBase<Scalar>(seed) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "random columns".
     */
    std::string getAlgorithmName() const override { return "random columns"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns
     * randomly.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the randomly selected columns.
     *
     * This method uses a Fisher-Yates shuffle to randomly select k columns
     * from the n available columns. The selection is uniform random (each
     * subset of k columns has equal probability).
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k,
                                               Eigen::Index *swap_count) override {
        const Eigen::Index n = X.cols();

        // Create a vector of all column indices
        std::vector<Eigen::Index> indices(static_cast<size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            indices[static_cast<size_t>(i)] = i;
        }

        // Lock mutex for thread-safe access to RNG and reproducible sequence
        std::lock_guard<std::mutex> lock(this->gen_mutex);

        // Shuffle and take the first k elements
        std::shuffle(indices.begin(), indices.end(), this->gen);
        indices.resize(static_cast<size_t>(k));

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_RANDOM_COLUMNS_SELECTOR_H
