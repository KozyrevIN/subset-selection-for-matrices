#ifndef MAT_SUBSET_BENCH_WEIGHTED_GRAPH_INCIDENCE_MATRIX_GENERATOR_H
#define MAT_SUBSET_BENCH_WEIGHTED_GRAPH_INCIDENCE_MATRIX_GENERATOR_H

#include <Eigen/SVD> // For Eigen::BDCSVD

#include "GraphIncidenceMatrixGenerator.h" // For the base class

namespace MatSubset::Bench {

/*!
 * @brief Generator of a matrix of right singular vectors of an oriented
 * edge-vertex incidence matrix of a random weighted fully-connected graph with
 * m + 1 verticies and n edges.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 */
template <typename Scalar>
class WeightedGraphIncidenceMatrixGenerator
    : public GraphIncidenceMatrixGenerator<Scalar> {
  public:
    /*!
     * @brief Constructor for WeightedGraphIncidenceMatrixGenerator with a
     * random seed.
     * @param m Number of rows.
     * @param n Number of columns.
     */
    WeightedGraphIncidenceMatrixGenerator(Eigen::Index m, Eigen::Index n)
        : GraphIncidenceMatrixGenerator<Scalar>(m, n) {}

    /*!
     * @brief Constructor for WeightedGraphIncidenceMatrixGenerator with a
     * specified seed.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param seed Seed for the random number generator.
     */
    WeightedGraphIncidenceMatrixGenerator(Eigen::Index m, Eigen::Index n,
                                          std::mt19937::result_type seed)
        : GraphIncidenceMatrixGenerator<Scalar>(m, n, seed) {}

    /*!
     * @brief Gets a string description of the matrix type.
     * @return A string identifying the matrix as a weighted graph incidence
     * matrix.
     */
    [[nodiscard]] std::string getMatrixType() const override {
        return "weighted graph incidence matrix";
    }

    /*!
     * @brief Generates a matrix of right singular vectors from a weighted
     * graph incidence matrix.
     * @return An Eigen::MatrixX<Scalar> of the specified dimensions.
     *
     * The generation process involves:
     * 1. Generating the incidence matrix of a non-weighted graph using a method
     * of `GraphIncidenceMatrixGenerator`.
     * 2. Multiplying columns of this matrix by a random number from uniform
     * distribution on [0, 1] segment.
     * 3. Computing the SVD of the resulting weighted incidence matrix.
     * 4. Returning the transpose of the matrix of right singular vectors (V).
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        auto [m, n] = this->matrixSize;

        std::uniform_real_distribution<Scalar> dis(0, 1);
        Eigen::VectorX<Scalar> weights(n);
        weights.setConstant(1);
        for (Eigen::Index i = 0; i < n; ++i) {
            weights(i) -= dis(this->gen);
        }

        Eigen::MatrixX<Scalar> M =
            this->incidenceMatrix() * weights.cwiseSqrt().asDiagonal();

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(M, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV();
        V.conservativeResize(Eigen::NoChange, this->matrixSize.first);
        return V.transpose();
    }
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_BENCH_WEIGHTED_GRAPH_INCIDENCE_MATRIX_GENERATOR_H
