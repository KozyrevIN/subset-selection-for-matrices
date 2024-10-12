#include "../../include/subset_selection_algorithms/gradient_descent_selector.h"

template <typename scalar> 
GradientDescentSelector<scalar>::GradientDescentSelector(): SubsetSelector<scalar>("gradient_descent") {
    //do nothing
}

template <typename scalar>
std::vector<uint> GradientDescentSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    Eigen::VectorX<scalar> S(X.cols());
    Eigen::VectorX<scalar> grad(X.cols());

    std::vector<uint> cols(k);
    for (int i = 0; i < k; ++i) {
        cols[i] = i;
    }

    return cols;
}

template class GradientDescentSelector<float>;
template class GradientDescentSelector<double>;