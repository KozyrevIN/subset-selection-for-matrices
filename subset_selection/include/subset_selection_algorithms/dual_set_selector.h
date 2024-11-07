#ifndef DUAL_SET_SELECTOR_H
#define DUAL_SET_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection
{

template <typename scalar>
class DualSetSelector : public SubsetSelector<scalar> 
{    
public:
    DualSetSelector(); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

}

#include "../../src/subset_selection_algorithms/dual_set_selector.hpp"

#endif