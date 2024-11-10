#ifndef RANK_REVEALING_QR_SELECTOR_H
#define RANK_REVEALING_QR_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class RankRevealingQRSelector : public SubsetSelector<scalar> {
  public:
    RankRevealingQRSelector();

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/rank_revealing_QR_selector.hpp"

#endif