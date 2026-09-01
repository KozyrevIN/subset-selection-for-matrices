#include <memory>
#include <random>
#include <set>
#include <vector>

#include <doctest/doctest.h>

#include <Eigen/Core>
#include <Eigen/QR>

#include <MatSubset/RectMaxvolSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

namespace {

// Rect-maxvol written the slow, obviously-correct way: at every step rebuild
// C = X_S^+ X_R from scratch and take the remaining column of largest squared
// norm. The selector computes the same greedy choice by rank-1 updates, so the
// two must agree exactly on the index set.
//
// This is the ground truth the incremental recurrence is worth having: a wrong
// update rule still returns a plausible index set of the right size, so
// check_subset cannot see it. It only shows up as a selection that stops
// improving once k passes m.
template <typename Scalar>
std::vector<Eigen::Index> rect_maxvol_reference(const Eigen::MatrixX<Scalar> &X,
                                                Eigen::Index k,
                                                std::vector<Eigen::Index> start) {
    const Eigen::Index n = X.cols();
    std::vector<Eigen::Index> selected = std::move(start);

    while (static_cast<Eigen::Index>(selected.size()) < k) {
        std::set<Eigen::Index> chosen(selected.begin(), selected.end());
        std::vector<Eigen::Index> remaining;
        for (Eigen::Index j = 0; j < n; ++j) {
            if (!chosen.count(j)) {
                remaining.push_back(j);
            }
        }

        Eigen::MatrixX<Scalar> X_S = X(Eigen::all, selected);
        Eigen::MatrixX<Scalar> C =
            X_S.completeOrthogonalDecomposition().pseudoInverse() *
            X(Eigen::all, remaining);

        Eigen::Index j_max;
        C.colwise().squaredNorm().maxCoeff(&j_max);
        selected.push_back(remaining[static_cast<size_t>(j_max)]);
    }
    return selected;
}

template <typename Scalar>
Eigen::MatrixX<Scalar> gaussian(Eigen::Index m, Eigen::Index n,
                                std::mt19937 &gen) {
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixX<Scalar> X(m, n);
    for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            X(i, j) = static_cast<Scalar>(dist(gen));
        }
    }
    return X;
}

} // namespace

TEST_CASE_TEMPLATE("rect-maxvol", Scalar, float, double) {

    std::unique_ptr<MatSubset::RectMaxvolSelector<Scalar>> selector =
        std::make_unique<MatSubset::RectMaxvolSelector<Scalar>>(1.01);

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "rect-maxvol");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_bounds(selector.get(), 3, 5);

    // Oversampling must keep paying: past k = m the selector leaves the square
    // maxvol set and every added column has to go on shrinking ||X_S^+||_F at
    // roughly the rate the naive recomputation achieves. A stale score vector
    // — one whose update forgot that C grows a row per added column — still
    // produces a legal subset, and still improves a little, so only a
    // comparison against the ground truth catches it.
    SUBCASE("oversampled selection matches the naive recomputation") {
        std::mt19937 gen(7);
        for (Eigen::Index m : {4, 8}) {
            const Eigen::Index n = 60;
            Eigen::MatrixX<Scalar> X = gaussian<Scalar>(m, n, gen);

            // The reference has to start where the selector starts, so seed it
            // with the selector's own square-maxvol set.
            std::vector<Eigen::Index> start = selector->selectSubset(X, m);

            for (Eigen::Index k : {m + 1, 2 * m, 4 * m}) {
                std::vector<Eigen::Index> got = selector->selectSubset(X, k);
                std::vector<Eigen::Index> want =
                    rect_maxvol_reference<Scalar>(X, k, start);

                std::sort(got.begin(), got.end());
                std::sort(want.begin(), want.end());
                CHECK(got == want);
            }
        }
    }
}
