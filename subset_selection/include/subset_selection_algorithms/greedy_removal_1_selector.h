#ifndef GREEDY_REMOVAL_1_SELECTOR
#define GREEDY_REMOVAL_1_SELECTOR

#include "subset_selector.h"

namespace SubsetSelection
{

template <typename scalar>
class GreedyRemoval1Selector : public SubsetSelector<scalar>
{
private:
    scalar eps;
public:
    GreedyRemoval1Selector(scalar eps = 1e-8); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};

}

#include "../../src/subset_selection_algorithms/greedy_removal_1_selector.hpp"

#endif