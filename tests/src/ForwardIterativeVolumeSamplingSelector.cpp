#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/ForwardIterativeVolumeSamplingSelector.h>

#include "CheckBounds.h"
#include "CheckReproducibility.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("forward iterative volume sampling", Scalar, float,
                   double) {

    std::unique_ptr<MatSubset::ForwardIterativeVolumeSamplingSelector<Scalar>>
        selector = std::make_unique<
            MatSubset::ForwardIterativeVolumeSamplingSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "forward iterative volume sampling");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);

    // Check reproducibility
    check_reproducibility<
        MatSubset::ForwardIterativeVolumeSamplingSelector<Scalar>>(5, 20);
}
