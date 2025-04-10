#ifndef MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H
#define MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class RankRevealingQRSelector : public SelectorBase<scalar> {
  public:
    RankRevealingQRSelector() {}

    std::string getAlgorithmName() const override {

        return "rank-revealing QR";
    }

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &X,
                                   uint k) override {

        Eigen::ColPivHouseholderQR<Eigen::MatrixX<scalar>> qr(x);
        Eigen::MatrixX<scalar> P = qr.colsPermutation();

        std::vector<uint> vect(k);

        for (int j = 0; j < k; ++j) {
            int i = 0;
            for (; std::abs(P(i, j)) == 0; ++i)
            vect[j] = i;
        }

        std::sort(vect.begin(), vect.end());

        return vect;
    }
};

} // namespace MatSubset

#endif