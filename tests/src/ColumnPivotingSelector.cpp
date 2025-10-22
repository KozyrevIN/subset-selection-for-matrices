#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/ColumnPivotingSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("column pivoting", Scalar, float, double) {

    std::unique_ptr<MatSubset::ColumnPivotingSelector<Scalar>> selector =
        std::make_unique<MatSubset::ColumnPivotingSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "column pivoting");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 3);

    // Check bounds
    check_subset(selector.get(), 3, 3);
}