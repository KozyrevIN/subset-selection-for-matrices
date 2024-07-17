#ifndef subset_selector_header
    #include "subset_selector.h"
    #define subset_selector_header
#endif

template <typename scalar>
class RankRevealingQRSelector : public SubsetSelector<scalar>
{
private:

public:
    RankRevealingQRSelector(); 

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k) override;
};