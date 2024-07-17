#ifndef subset_selector_header
    #include "subset_selector.h"
    #define subset_selector_header
#endif

template <typename scalar>
class GreedyRemoval2Selector : public SubsetSelector<scalar>
{
private:
    double eps;

public:
    GreedyRemoval2Selector(double eps = 1e-8); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};