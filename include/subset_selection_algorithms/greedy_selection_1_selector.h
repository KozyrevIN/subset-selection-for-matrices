#ifndef subset_selector_header
    #define subset_selector_header
    #include "subset_selector.h"
#endif

template <typename scalar>
class GreedySelection1Selector : public SubsetSelector<scalar> 
{    
public:
    GreedySelection1Selector(); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;

    scalar frobeniusBound(uint m, uint n, uint k) override;
    scalar l2Bound(uint m, uint n, uint k) override;
};