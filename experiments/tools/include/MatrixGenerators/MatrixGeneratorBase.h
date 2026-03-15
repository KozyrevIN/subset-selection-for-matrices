#ifndef MAT_SUBSET_EXPERIMENTS_MATRIX_GENERATOR_BASE_H
#define MAT_SUBSET_EXPERIMENTS_MATRIX_GENERATOR_BASE_H

#include <mutex>   // For std::mutex, std::lock_guard
#include <random>  // For std::mt19937, std::random_device
#include <string>  // For std::string
#include <utility> // For std::pair

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

namespace MatSubset::Experiments {

/*!
 * @brief Base class for generating random matrices.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 *
 * This base class provides common functionality for matrix generation,
 * including storing matrix dimensions and managing a random number generator.
 * Derived classes implement specific matrix generation algorithms.
 *
 * @note The `generateMatrix()` method is thread-safe and can
 * be called concurrently from multiple threads on the same instance. The RNG
 * state is protected by a mutex to ensure reproducible sequences: for a given
 * seed, any sequence of generated matrices will always be identical regardless
 * of thread interleaving.
 */
template <typename Scalar> class MatrixGeneratorBase {
  protected:
    mutable std::mt19937 gen; ///< Mersenne Twister random number generator.
    mutable std::mutex gen_mutex; ///< Mutex protecting the RNG for thread-safety.
    const std::pair<Eigen::Index, Eigen::Index>
        matrixSize; ///< Dimensions of the matrix (rows, columns).

  public:
    /*!
     * @brief Constructor for MatrixGeneratorBase with a random seed.
     * @param m Number of rows.
     * @param n Number of columns.
     *
     * This constructor uses a helper function to properly seed the generator
     * in the member initializer list, which is the most efficient method.
     */
    MatrixGeneratorBase(Eigen::Index m, Eigen::Index n)
        : matrixSize{m, n}, gen{create_seeded_generator()} {}

    /*!
     * @brief Constructor for MatrixGeneratorBase with a specified seed.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param seed Seed for the random number generator.
     */
    MatrixGeneratorBase(Eigen::Index m, Eigen::Index n,
                    std::mt19937::result_type seed)
        : matrixSize{m, n}, gen{seed} {}

    /*! @brief Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~MatrixGeneratorBase() = default;

    /*!
     * @brief Gets the dimensions of the matrix.
     * @return A `std::pair` containing the number of rows (first) and columns
     * (second).
     */
    [[nodiscard]] std::pair<Eigen::Index, Eigen::Index> getMatrixSize() const {
        return matrixSize;
    }

    /*!
     * @brief Gets a string description of the matrix type.
     * @return A string representing the type of matrix generated.
     * Derived classes should override this for specific descriptions.
     */
    [[nodiscard]] virtual std::string getMatrixType() const {
        return "zero matrix";
    }

    /*!
     * @brief Generates a matrix.
     * @return An Eigen::MatrixX<Scalar> of the specified dimensions.
     * Derived classes MUST override this to implement their specific generation
     * logic. The default implementation returns a zero-initialized matrix.
     */
    [[nodiscard]] virtual Eigen::MatrixX<Scalar> generateMatrix() {
        auto [m, n] = matrixSize;
        return Eigen::MatrixX<Scalar>::Zero(static_cast<Eigen::Index>(m),
                                            static_cast<Eigen::Index>(n));
    }

  private:
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
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_MATRIX_GENERATOR_BASE_H