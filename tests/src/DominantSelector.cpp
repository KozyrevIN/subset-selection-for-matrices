#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/DominantSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("dominant", Scalar, float, double) {

    std::unique_ptr<MatSubset::DominantSelector<Scalar>> selector =
        std::make_unique<MatSubset::DominantSelector<Scalar>>(1.01);

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "dominant");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}