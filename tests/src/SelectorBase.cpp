#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/SelectorBase.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("selector base", Scalar, float, double) {

    std::unique_ptr<MatSubset::SelectorBase<Scalar>> selector =
        std::make_unique<MatSubset::SelectorBase<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "first k columns");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}