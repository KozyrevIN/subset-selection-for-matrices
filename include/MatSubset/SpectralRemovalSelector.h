#ifndef MAT_SUBSET_SPECTRAL_REMOVAL_H
#define MAT_SUBSET_SPECTRAL_REMOVAL_H

#include "FrobeniusRemovalSelector.h"

namespace MatSubset {

template <typename scalar>
class SpectralRemovalSelector : public FrobeniusRemovalSelector<scalar> {
  public:
    SpectralRemovalSelector(scalar eps = 1e-6)
        : FrobeniusRemovalSelector<scalar>(eps) {}

    std::string getAlgorithmName() const override { return "spectral removal"; }

    std::vector<Eigen::Index> selectSubset(const Eigen::MatrixX<scalar> &X,
                                           Eigen::Index k) override {

        Eigen::BDCSVD svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<scalar> V = svd.matrixV().transpose();
        return FrobeniusRemovalSelector<scalar>::selectSubset(V, k);
    }

  private:
    scalar boundInternal(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                         Norm norm) const override {

        scalar bound;
        if (norm == Norm::L2) {
            bound = 1 / (1 + (scalar)(m * (n - k)) / (k - m + 1));
        } else if (norm == Norm::Frobenius) {
            bound = (scalar)(k - m + 1) / (n - m + 1) / m;
        }

        return bound;
    }
};

} // namespace MatSubset

#endif