#include <unordered_set>

#ifndef subset_selector_header
    #include "subset_selector.h"
    #define subset_selector_header
#endif

template <typename scalar>
class GreedyRemoval1Selector : public SubsetSelector<scalar>
{
private:
    scalar eps;
public:
    GreedyRemoval1Selector(scalar eps); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};