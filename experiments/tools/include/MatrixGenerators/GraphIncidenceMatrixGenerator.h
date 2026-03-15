#ifndef MAT_SUBSET_EXPERIMENTS_GRAPH_INCIDENCE_MATRIX_GENERATOR_H
#define MAT_SUBSET_EXPERIMENTS_GRAPH_INCIDENCE_MATRIX_GENERATOR_H

#include <algorithm> // For std::shuffle
#include <cassert>   // For assert
#include <list>      // For std::list
#include <set>       // For std::set
#include <stdexcept> // For std::runtime_error
#include <vector>    // For std::vector

#include <Eigen/SVD> // For Eigen::BDCSVD

#include "MatrixGeneratorBase.h" // For the base class

namespace MatSubset::Experiments {

/*!
 * @brief Generator of a matrix of right singular vectors of an oriented
 * edge-vertex incidence matrix of a fully-connected graph with m + 1 verticies
 * and n edges from uniform distribution.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 */
template <typename Scalar>
class GraphIncidenceMatrixGenerator : public MatrixGeneratorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for GraphIncidenceMatrixGenerator with a random seed.
     * @param m Number of rows.
     * @param n Number of columns.
     */
    GraphIncidenceMatrixGenerator(Eigen::Index m, Eigen::Index n)
        : MatrixGeneratorBase<Scalar>(m, n) {
        assert(
            m <= n &&
            "Number of edges (n) must be at least number of vertices - 1 (m)");
        Eigen::Index max_edges = m * (m + 1) / 2;
        assert(n <= max_edges &&
               "Number of edges (n) must be smaller then maximum possible "
               "amount of edges in a graph with m + 1 verticies");
    }

    /*!
     * @brief Constructor for GraphIncidenceMatrixGenerator with a specified
     * seed.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param seed Seed for the random number generator.
     */
    GraphIncidenceMatrixGenerator(Eigen::Index m, Eigen::Index n,
                                  std::mt19937::result_type seed)
        : MatrixGeneratorBase<Scalar>(m, n, seed) {
        assert(
            m <= n &&
            "Number of edges (n) must be at least number of vertices - 1 (m)");
        Eigen::Index max_edges = m * (m + 1) / 2;
        assert(n <= max_edges &&
               "Number of edges (n) must be smaller then maximum possible "
               "amount of edges in a graph with m + 1 verticies");
    }

    /*!
     * @brief Gets a string description of the matrix type.
     * @return A string identifying the matrix as related to graph incidence
     * matrix.
     */
    [[nodiscard]] std::string getMatrixType() const override {
        return "graph incidence matrix";
    }

    /*!
     * @brief Generates a matrix of right singular vectors from a graph
     * incidence matrix.
     * @return An Eigen::MatrixX<Scalar> of the specified dimensions.
     *
     * The generation process involves:
     * 1. Generating a random graph with m + 1 verticies and n edges.
     * 2. Verifying it's connectivity, and retrying if needed.
     * 2. Constructing the corresponding edge-vertex incidence matrix.
     * 3. Computing the SVD of the incidence matrix.
     * 4. Returning the transpose of the matrix of right singular vectors (V).
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        Eigen::MatrixX<Scalar> M;
        {
            // Lock mutex only for RNG access, not for expensive SVD
            std::lock_guard<std::mutex> lock(this->gen_mutex);
            M = incidenceMatrix();
        } // Mutex unlocked here

        // Perform SVD without holding the lock
        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(M, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV();
        V.conservativeResize(Eigen::NoChange, this->matrixSize.first);
        return V.transpose();
    }

  protected:
    /*!
     * @brief Generates the incidence matrix of a random connected graph.
     * @return An Eigen::MatrixX<Scalar> representing the incidence matrix.
     *
     * @note: This method must be called while holding gen_mutex (typically
     * from generateMatrix() which acquires the lock).
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> incidenceMatrix() {
        auto [m, n] = this->matrixSize;

        const int max_iterations = 100;
        for (int i = 0; i < max_iterations; ++i) {
            auto edge_list = randomEdgeList();
            if (checkConnectivity(edge_list)) {
                Eigen::MatrixX<Scalar> M =
                    Eigen::MatrixX<Scalar>::Zero(m + 1, n);
                for (Eigen::Index j = 0; j < n; ++j) {
                    auto [v_1, v_2] = edge_list[j];
                    M(v_1, j) = static_cast<Scalar>(1);
                    M(v_2, j) = static_cast<Scalar>(-1);
                }
                return M;
            }
        }

        throw std::runtime_error(
            "Failed to generate a connected graph within " +
            std::to_string(max_iterations) + " iterations.");
    }

  private:
    /*!
     * @brief Generates a random list of edges.
     * @return A vector of pairs representing the edges.
     */
    [[nodiscard]] std::vector<std::pair<Eigen::Index, Eigen::Index>>
    randomEdgeList() {
        auto [m, n] = this->matrixSize;
        Eigen::Index num_v = m + 1;
        Eigen::Index num_e = n;

        std::vector<std::pair<Eigen::Index, Eigen::Index>> edge_list;
        edge_list.reserve((num_v * (num_v - 1) / 2));

        for (Eigen::Index i = 0; i < num_v - 1; ++i) {
            for (Eigen::Index j = i + 1; j < num_v; ++j) {
                edge_list.push_back(std::make_pair(i, j));
            }
        }

        std::shuffle(edge_list.begin(), edge_list.end(), this->gen);
        edge_list.resize(num_e);

        return edge_list;
    }

    /*!
     * @brief Checks if the graph is connected by counting the connectivity
     * components.
     * @param edge_list The list of edges in the graph.
     * @return True if the graph is connected, false otherwise.
     */
    [[nodiscard]] bool checkConnectivity(
        const std::vector<std::pair<Eigen::Index, Eigen::Index>> &edge_list)
        const {
        if (edge_list.empty()) {
            return (this->matrixSize.first + 1) <= 1;
        }

        auto [v_1, v_2] = edge_list[0];
        std::set<Eigen::Index> seed{v_1, v_2};
        std::list<std::set<Eigen::Index>> components{seed};

        for (size_t i = 1; i < edge_list.size(); ++i) {
            auto [v_1_edge, v_2_edge] = edge_list[i];

            auto find_component = [&](Eigen::Index v) {
                for (auto it = components.begin(); it != components.end();
                     ++it) {
                    if (it->count(v)) {
                        return it;
                    }
                }
                return components.end();
            };

            auto v_1_component = find_component(v_1_edge);
            auto v_2_component = find_component(v_2_edge);

            if (v_1_component != components.end() &&
                v_2_component != components.end()) {
                if (v_1_component != v_2_component) {
                    v_1_component->merge(*v_2_component);
                    components.erase(v_2_component);
                }
            } else if (v_1_component != components.end()) {
                v_1_component->insert(v_2_edge);
            } else if (v_2_component != components.end()) {
                v_2_component->insert(v_1_edge);
            } else {
                components.push_back(
                    std::set<Eigen::Index>{v_1_edge, v_2_edge});
            }
        }

        return components.size() == 1;
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_GRAPH_INCIDENCE_MATRIX_GENERATOR_H
