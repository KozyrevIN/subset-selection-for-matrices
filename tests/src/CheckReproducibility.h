#include <concepts>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include <Eigen/Core>

#include <MatSubset/RandomizedBase.h>

/*!
 * @brief Helper function to generate a random test matrix.
 * @tparam Scalar The scalar type (float or double).
 * @param m Number of rows.
 * @param n Number of columns.
 * @param seed Seed for reproducible matrix generation.
 * @return A random m x n matrix.
 */
template <typename Scalar>
Eigen::MatrixX<Scalar>
generate_test_matrix(Eigen::Index m, Eigen::Index n,
                     std::mt19937::result_type seed = 42) {
    Eigen::MatrixX<Scalar> X(m, n);
    std::mt19937 gen(seed);
    std::normal_distribution<Scalar> dist(static_cast<Scalar>(-1.0),
                                          static_cast<Scalar>(1.0));
    for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            X(i, j) = dist(gen);
        }
    }
    return X;
}

/*!
 * @brief Checks reproducibility of a randomized selector.
 * @tparam Selector The selector type (must inherit from RandomizedBase).
 * @tparam Scalar The scalar type (float or double).
 * @param m Number of rows in the test matrix.
 * @param n Number of cols in the test matrix.
 *
 * This function verifies that two selectors with the same seed produce
 * identical results.
 *
 * @note This function only accepts selectors that inherit from
 * RandomizedBase<Scalar>. The constraint is enforced at compile-time via
 * C++20 requires clause.
 */
template <typename Selector>
    requires std::derived_from<
        Selector, MatSubset::RandomizedBase<typename Selector::Scalar>>
void check_reproducibility(Eigen::Index m, Eigen::Index n) {

    using Scalar = Selector::Scalar;
    // Matrix Setup
    Eigen::MatrixX<Scalar> X = generate_test_matrix<Scalar>(m, n, 42);
    const std::mt19937::result_type seed = 12345;

    for (Eigen::Index k = m; k <= n; ++k) {
        SUBCASE("Same seed produces identical results") {
            Selector selector1(seed);
            Selector selector2(seed);

            // Make three sequential calls on each selector
            std::vector<Eigen::Index> indices1_call1 =
                selector1.selectSubset(X, k);
            std::vector<Eigen::Index> indices2_call1 =
                selector2.selectSubset(X, k);
            CHECK(indices1_call1 == indices2_call1);

            std::vector<Eigen::Index> indices1_call2 =
                selector1.selectSubset(X, k);
            std::vector<Eigen::Index> indices2_call2 =
                selector2.selectSubset(X, k);
            CHECK(indices1_call2 == indices2_call2);

            std::vector<Eigen::Index> indices1_call3 =
                selector1.selectSubset(X, k);
            std::vector<Eigen::Index> indices2_call3 =
                selector2.selectSubset(X, k);
            CHECK(indices1_call3 == indices2_call3);
        }
    }
}
