#ifndef GREEDY_SELECTION_1_SELECTOR_H
#define GREEDY_SELECTION_1_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection
{

template <typename scalar>
class GreedySelection1Selector : public SubsetSelector<scalar> 
{    
public:
    GreedySelection1Selector(); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

}

#include "../../src/subset_selection_algorithms/greedy_selection_1_selector.hpp"

#endif