#include <algorithm>
#include <set>
#include <vector>

#include <doctest/doctest.h>

#include <Eigen/Core>

#include <MatSubset/SelectorBase.h>

template <typename Scalar>
void check_subset(MatSubset::SelectorBase<Scalar> *selector,
                  Eigen::Index k_begin, Eigen::Index k_end) {

    // Matrix Setup
    const Eigen::Index m = 3;
    const Eigen::Index n = 5;

    Eigen::MatrixX<Scalar> X(m, n);
    // clang-format off
    X <<  1,  2,  3,  4,  5,
          6,  7,  8,  9, 10,
         11, 12, 13, 14, 15;
    // clang-format on

    // Checking that selected "subset" is indeed subset of column indices for
    // each k
    for (Eigen::Index k = k_begin; k <= k_end; ++k) {
        SUBCASE("Returned vector is subset of column indices for given k") {
            std::vector<Eigen::Index> indices = selector->selectSubset(X, k);

            std::sort(indices.begin(), indices.end());
            CHECK(indices.size() == k);
            CHECK(indices[0] >= 0);
            CHECK(indices[static_cast<size_t>(k - 1)] <= X.cols() - 1);

            std::set<Eigen::Index> indices_set(indices.begin(), indices.end());
            CHECK(indices.size() == indices_set.size());
        }
    }
}