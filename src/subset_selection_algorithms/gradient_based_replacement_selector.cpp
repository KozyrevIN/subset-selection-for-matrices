#include <vector>

#include "../../include/subset_selection_algorithms/gradient_based_replacement_selector.h"

template <typename scalar>
GradientBasedReplacementSelector<scalar>::GradientBasedReplacementSelector(scalar eps): SubsetSelector<scalar>("gradient_based_replacement"), eps(eps) {
    //do nothing
}

template <typename scalar>
std::vector<uint> GradientBasedReplacementSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& x, uint k) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixX<scalar>> qr(x);
    Eigen::MatrixX<scalar> P = qr.colsPermutation();

    std::vector<uint> vect(k);

    for (int j = 0; j < k; ++j) {
        int i = 0;
        for(; std::abs(P(i, j)) == 0; ++i);
        vect[j] = i;
    }

    grad

    std::sort(vect.begin(), vect.end());

    return vect;
}

template class GradientBasedReplacementSelector<float>;
template class GradientBasedReplacementSelector<double>;
template class GradientBasedReplacementSelector<std::complex<float>>;
template class GradientBasedReplacementSelector<std::complex<double>>;