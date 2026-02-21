#ifndef MAT_SUBSET_RANDOMIZED_BASE_H
#define MAT_SUBSET_RANDOMIZED_BASE_H

#include <algorithm> // For std::generate
#include <array>     // For std::array
#include <mutex>     // For std::mutex, std::lock_guard
#include <random>    // For std::mt19937, std::random_device, std::seed_seq

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Abstract base class for randomized subset selection algorithms.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class extends SelectorBase to provide common random number generation
 * infrastructure for algorithms that use randomization. It manages a Mersenne
 * Twister random number generator (std::mt19937) with thread-safe access.
 *
 * Derived classes (such as RandomColumnsSelector and LeverageScoresSelector)
 * inherit the RNG and can use it to implement their randomized selection
 * strategies.
 *
 * @note The RNG is thread-safe: the `gen` and `gen_mutex` members allow
 * derived classes to safely access the generator from multiple threads while
 * maintaining reproducible sequences for a given seed.
 */
template <typename Scalar>
class RandomizedBase : public SelectorBase<Scalar> {
  protected:
    mutable std::mt19937 gen; ///< Mersenne Twister random number generator.
    mutable std::mutex
        gen_mutex; ///< Mutex protecting the RNG for thread-safety.

    /*!
     * @brief Creates and returns a fully-seeded mt19937 generator.
     * @return A std::mt19937 object with its state initialized by
     * std::seed_seq.
     *
     * This helper function encapsulates the logic for high-quality seeding. It
     * gathers entropy from std::random_device and uses std::seed_seq to
     * initialize the large state of the mt19937 engine.
     */
    static std::mt19937 create_seeded_generator() {
        std::random_device rd;

        // Gather 8 integers of entropy from the random_device
        std::array<std::random_device::result_type, 8> seed_data;
        std::generate(seed_data.begin(), seed_data.end(), std::ref(rd));

        std::seed_seq seq(seed_data.begin(), seed_data.end());

        // Construct and return the fully seeded generator
        return std::mt19937(seq);
    }

  public:
    /*!
     * @brief Constructor with a random seed.
     *
     * This constructor uses a high-quality random seed from the system's
     * random device.
     */
    RandomizedBase() : gen{create_seeded_generator()} {}

    /*!
     * @brief Constructor with a specified seed for reproducibility.
     * @param seed Seed for the random number generator.
     *
     * Using the same seed will produce the same sequence of selections across
     * multiple runs, enabling reproducible experiments.
     */
    explicit RandomizedBase(std::mt19937::result_type seed) : gen{seed} {}

    /*!
     * @brief Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~RandomizedBase() = default;
};

} // namespace MatSubset

#endif // MAT_SUBSET_RANDOMIZED_BASE_H
