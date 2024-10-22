#ifndef GREEDY_REMOVAL_2_SELECTOR_H
#define GREEDY_REMOVAL_2_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection
{

template <typename scalar>
class GreedyRemoval2Selector : public SubsetSelector<scalar>
{
private:
    double eps;

public:
    GreedyRemoval2Selector(scalar eps = 1e-6); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

}

#include "../../src/subset_selection_algorithms/greedy_removal_2_selector.hpp"

#endif