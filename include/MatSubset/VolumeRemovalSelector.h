#ifndef VOLUME_REMOVAL_SELECTOR
#define VOLUME_REMOVAL_SELECTOR

#include "subset_selector.h"

namespace MatSubset {

template <typename scalar>
class VolumeRemovalSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

  public:
    VolumeRemovalSelector(scalar eps = 1e-6);

    std::string getAlgorithmName() const override;

    void removeByIdx(std::vector<uint> &cols, Eigen::ArrayX<scalar> &d,
                     Eigen::MatrixX<scalar> &V, Eigen::MatrixX<scalar> &V_dag,
                     uint j) const;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace MatSubset

#include "../../src/subset_selection_algorithms/volume_removal_selector.hpp"

#endif