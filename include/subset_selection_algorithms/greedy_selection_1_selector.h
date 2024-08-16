#ifndef subset_selector_header
    #include "subset_selector.h"
    #define subset_selector_header
#endif

template <typename scalar>
class GreedySelection1Selector : public SubsetSelector<scalar> {
    
public:
    GreedySelection1Selector(); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};