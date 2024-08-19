#ifndef subset_selector_header
    #define subset_selector_header
    #include "subset_selector.h"
#endif

template <typename scalar>
class GradientDescentSelector : public SubsetSelector<scalar> {
    
public:
    GradientDescentSelector(); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};