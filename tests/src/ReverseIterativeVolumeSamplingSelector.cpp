#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/ReverseIterativeVolumeSamplingSelector.h>

#include "CheckBounds.h"
#include "CheckReproducibility.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("reverse iterative volume sampling", Scalar, float,
                   double) {

    std::unique_ptr<MatSubset::ReverseIterativeVolumeSamplingSelector<Scalar>>
        selector = std::make_unique<
            MatSubset::ReverseIterativeVolumeSamplingSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() ==
          "reverse iterative volume sampling");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);

    // Check reproducibility
    check_reproducibility<
        MatSubset::ReverseIterativeVolumeSamplingSelector<Scalar>>(5, 20);
}
