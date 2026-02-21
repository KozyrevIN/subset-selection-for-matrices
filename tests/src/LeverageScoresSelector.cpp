#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/LeverageScoresSelector.h>

#include "CheckBounds.h"
#include "CheckReproducibility.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("leverage scores", Scalar, float, double) {

    std::unique_ptr<MatSubset::LeverageScoresSelector<Scalar>> selector =
        std::make_unique<MatSubset::LeverageScoresSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "leverage scores");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);

    // Check reproducibility
    check_reproducibility<MatSubset::LeverageScoresSelector<Scalar>>(5, 20);
}
