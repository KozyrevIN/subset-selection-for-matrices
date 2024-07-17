#include <vector>

#include "../../include/subset_selection_algorithms/rank_revealing_QR_selector.h"

template <typename scalar>
RankRevealingQRSelector<scalar>::RankRevealingQRSelector(): SubsetSelector<scalar>("rank_revealing_QR") {
    //do nothing
}

template <typename scalar>
std::vector<uint> RankRevealingQRSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& x, uint k) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixX<scalar>> qr(x);
    Eigen::MatrixX<scalar> P = qr.colsPermutation();

    std::vector<uint> vect(k);

    for (int j = 0; j < k; ++j) {
        int i = 0;
        for(; std::abs(P(i, j)) == 0; ++i);
        vect[j] = i;
    }

    std::sort(vect.begin(), vect.end());

    return vect;
}

template class RankRevealingQRSelector<float>;
template class RankRevealingQRSelector<double>;
template class RankRevealingQRSelector<std::complex<float>>;
template class RankRevealingQRSelector<std::complex<double>>;