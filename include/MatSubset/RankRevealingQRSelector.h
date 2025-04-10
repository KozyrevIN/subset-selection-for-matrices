#ifndef MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H
#define MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H

#include "subset_selector.h"

namespace MatSubset {

template <typename scalar>
class RankRevealingQRSelector : public SubsetSelector<scalar> {
  public:
    RankRevealingQRSelector();

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace MatSubset

#include "../../src/subset_selection_algorithms/rank_revealing_QR_selector.hpp"

#endif