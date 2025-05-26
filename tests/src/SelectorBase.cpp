#include <doctest/doctest.h>

#include <MatSubset/SelectorBase.h>

#include <set>
#include <algorithm>

TEST_CASE_TEMPLATE("selector base", scalar, float, double) {
    MatSubset::SelectorBase<scalar> selector;

    // Algorithm name
    CHECK(selector.getAlgorithmName() == "first k columns");

    // Algorithm itself
    Eigen::Index m = 3;
    Eigen::Index n = 5;
    Eigen::Index k = 3;

    Eigen::MatrixX<scalar> X(m, n);
    // clang-format off
    X <<  1,  2,  3,  4,  5,
          6,  7,  8,  9, 10,
         11, 12, 13, 14, 15;
    // clang-format on

    std::vector<Eigen::Index> indices = selector.selectSubset(X, k);
    std::sort(indices.begin(), indices.end());

    CHECK(indices.size() == k);
    CHECK(indices[0] >= 0);
    CHECK(indices[k - 1] <= n - 1);
    std::set<Eigen::Index> indices_set(indices.begin(), indices.end());
    CHECK(indices.size() == indices_set.size());

    // Bounds
    SUBCASE("Frobenius norm bounds") {
        CHECK(selector.template bound<MatSubset::Norm::Frobenius>(X, k) == 0);
        CHECK(selector.template bound<MatSubset::Norm::Frobenius>(m, n, k) == 0);
    }

    SUBCASE("Spectral norm bounds") {
        CHECK(selector.template bound<MatSubset::Norm::Spectral>(X, k) == 0);
        CHECK(selector.template bound<MatSubset::Norm::Spectral>(m, n, k) == 0); 
    }
}