#include <complex>

#include "../../include/subset_selection_algorithms/subset_selector.h"

template <typename scalar>
SubsetSelector<scalar>::SubsetSelector(const std::string& algorithm_name): algorithmName(algorithm_name) {
    //do nothing
}

template <typename scalar>
std::vector<uint> SubsetSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& x, uint k) {
    std::vector<uint> vect(k);
    for (int i = 0; i < k; ++i) {
        vect[i] = i;
    }
    return vect;
}

template class SubsetSelector<float>;
template class SubsetSelector<double>;
template class SubsetSelector<std::complex<float>>;
template class SubsetSelector<std::complex<double>>;
