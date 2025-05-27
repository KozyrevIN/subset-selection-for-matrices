#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/DualSetSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("dual set", Scalar, float, double) {

    std::unique_ptr<MatSubset::DualSetSelector<Scalar>> selector =
        std::make_unique<MatSubset::DualSetSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "dual set");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 4, 5);

    // Check bounds
    check_subset(selector.get(), 4, 5);
}