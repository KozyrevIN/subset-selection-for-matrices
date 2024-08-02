#ifndef subset_selector_header
    #include "subset_selector.h"
    #define subset_selector_header
#endif

template <typename scalar>
class GradientBasedReplacementSelector : public SubsetSelector<scalar>
{
private:
    scalar eps;
public:
    GradientBasedReplacementSelector(scalar eps = 1e-8); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};