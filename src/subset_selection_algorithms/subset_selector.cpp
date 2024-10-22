#include <complex>

#include "../../include/subset_selection_algorithms/subset_selector.h"

template <typename scalar>
SubsetSelector<scalar>::SubsetSelector(const std::string& algorithm_name): algorithmName(algorithm_name) {
    //do nothing
}

template <typename scalar>
std::vector<uint> SubsetSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols(k);
    for (int i = 0; i < k; ++i) {
        cols[i] = i;
    }

    return cols;
}

template <typename scalar>
scalar SubsetSelector<scalar>::frobeniusBound(const Eigen::MatrixX<scalar>& X, uint k) {
    return frobeniusBound(X.rows(), X.cols(), k);
}

template <typename scalar>
scalar SubsetSelector<scalar>::frobeniusBound(uint m, uint n, uint k) {
    return 0;
}

template <typename scalar>
scalar SubsetSelector<scalar>::l2Bound(const Eigen::MatrixX<scalar>& X, uint k) {
    return l2Bound(X.rows(), X.cols(), k);
}

template <typename scalar>
scalar SubsetSelector<scalar>::l2Bound(uint m, uint n, uint k) {
    return 0;
}

template class SubsetSelector<float>;
template class SubsetSelector<double>;
