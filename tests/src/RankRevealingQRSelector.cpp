#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/RankRevealingQRSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("rank-revealing QR", Scalar, float, double) {

    std::unique_ptr<MatSubset::RankRevealingQRSelector<Scalar>> selector =
        std::make_unique<MatSubset::RankRevealingQRSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "rank-revealing QR");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 3);

    // Check bounds
    check_subset(selector.get(), 3, 3);
}