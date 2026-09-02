#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/SVD>

#include <doctest/doctest.h>

#include <MatSubset/GappyPodSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("gappy pod+e", Scalar, float, double) {

    std::unique_ptr<MatSubset::GappyPodSelector<Scalar>> selector =
        std::make_unique<MatSubset::GappyPodSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "gappy pod+e");

    // Check that returned vector is subset of column indices for every k
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_bounds(selector.get(), 3, 5);

    SUBCASE("For k = m the selection is the Q-DEIM one") {
        // Without oversampling the greedy phase never runs, so what is left is
        // plain column-pivoted QR, which always takes the largest-norm column
        // first. With an orthogonal X the pivots are pinned deterministically.
        const Eigen::Index m = 3;
        const Eigen::Index n = 5;

        Eigen::MatrixX<Scalar> X = Eigen::MatrixX<Scalar>::Zero(m, n);
        X(0, 0) = 1;
        X(1, 1) = 1;
        X(2, 4) = 1;

        std::vector<Eigen::Index> indices = selector->selectSubset(X, m);
        std::sort(indices.begin(), indices.end());

        CHECK(indices[0] == 0);
        CHECK(indices[1] == 1);
        CHECK(indices[2] == 4);
    }

    SUBCASE("Oversampling extends the previous selection") {
        // Points are only appended, never exchanged, so the set selected for k
        // is contained in the one selected for k + 1.
        const Eigen::Index m = 3;
        const Eigen::Index n = 7;

        Eigen::MatrixX<Scalar> X(m, n);
        // clang-format off
        X <<  1,  2,  3,  4,  5,  6,  7,
              0,  6,  7,  8,  9, 10, 11,
              0,  0, 10, 11, 12, 13, 14;
        // clang-format on

        for (Eigen::Index k = m; k < n; ++k) {
            std::vector<Eigen::Index> indices = selector->selectSubset(X, k);
            std::vector<Eigen::Index> indices_next =
                selector->selectSubset(X, k + 1);

            std::sort(indices.begin(), indices.end());
            std::sort(indices_next.begin(), indices_next.end());

            CHECK(std::includes(indices_next.begin(), indices_next.end(),
                                indices.begin(), indices.end()));
        }
    }

    SUBCASE("Oversampling does not shrink the smallest singular value") {
        // The greedy criterion targets sigma_min of the selected submatrix,
        // and appending a column can only increase it.
        const Eigen::Index m = 3;
        const Eigen::Index n = 7;

        Eigen::MatrixX<Scalar> X(m, n);
        // clang-format off
        X <<  1,  2,  3,  4,  5,  6,  7,
              0,  6,  7,  8,  9, 10, 11,
              0,  0, 10, 11, 12, 13, 14;
        // clang-format on

        Scalar previous_sigma_min = 0;
        for (Eigen::Index k = m; k <= n; ++k) {
            std::vector<Eigen::Index> indices = selector->selectSubset(X, k);
            Eigen::MatrixX<Scalar> X_S = X(Eigen::all, indices);

            Eigen::JacobiSVD<Eigen::MatrixX<Scalar>> svd(X_S);
            Scalar sigma_min = svd.singularValues()(m - 1);

            CHECK(sigma_min >= previous_sigma_min *
                                   (1 - 100 * std::numeric_limits<
                                                  Scalar>::epsilon()));
            previous_sigma_min = sigma_min;
        }
    }
}
