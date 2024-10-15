#include <vector>

#include "../../include/subset_selection_algorithms/greedy_removal_2_selector.h"

template <typename scalar>
GreedyRemoval2Selector<scalar>::GreedyRemoval2Selector(scalar eps): SubsetSelector<scalar>("greedy_removal_2"), eps(eps) {
    //do nothing
}

template <typename scalar>
std::vector<uint> GreedyRemoval2Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    uint m = X.rows();
    uint n = X.cols();

    std::vector<uint> cols(n);
    for (uint j = 0; j < n; ++j) {
        cols[j] = j;
    }

    if (k < n) {
        Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<scalar> V = svd.matrixV().transpose(); 
        Eigen::VectorX<scalar> S_inv2 = svd.singularValues().array().inverse().square();

        Eigen::MatrixX<scalar> VVT_invV = (V * V.transpose()).inverse() * V;
        Eigen::ArrayX<scalar> l = (VVT_invV.transpose() * S_inv2.asDiagonal() * VVT_invV).diagonal();
        Eigen::ArrayX<scalar> d = 1 - (V.transpose() * VVT_invV).diagonal().array();

        for (uint cols_remaining = n; cols_remaining >= k; --cols_remaining) {
            uint j_min = 0;
            for (uint j = 0; j < cols_remaining; ++j) {
                if (d(j) > eps and l(j) * d(j_min) < l(j_min) * d(j)) {
                    j_min = j;
                }
            }

            std::swap(cols[j_min], cols[cols_remaining - 1]);
            std::swap(l(j_min), l(cols_remaining - 1));
            std::swap(d(j_min), d(cols_remaining - 1));
            V.col(j_min).swap(V.col(cols_remaining - 1));
            VVT_invV.col(j_min).swap(VVT_invV.col(cols_remaining - 1));

            Eigen::ArrayX<scalar> wTVVT_invV = (V.col(cols_remaining - 1).transpose() * VVT_invV.leftCols(cols_remaining)).array();
            Eigen::ArrayX<scalar> wTVVT_inv2V = (VVT_invV.col(cols_remaining - 1).transpose() * S_inv2.asDiagonal() *
                                                VVT_invV.leftCols(cols_remaining)).array();
            l.head(cols_remaining) += 2 * wTVVT_invV * wTVVT_inv2V / d(cols_remaining - 1) + 
                                      wTVVT_invV.square() * wTVVT_inv2V(cols_remaining - 1) / (d(cols_remaining - 1) * d(cols_remaining - 1));
            //l.head(cols_remaining) = (VVT_invV.leftCols(cols_remaining).transpose() * S_inv2.asDiagonal() * VVT_invV.leftCols(cols_remaining)).diagonal();
            //d.head(cols_remaining) = 1 - (V.leftCols(cols_remaining).transpose() * VVT_invV.leftCols(cols_remaining)).diagonal().array();
            /*d.head(cols_remaining) -= (V.col(cols_remaining - 1).transpose() * 
                                      VVT_invV.leftCols(cols_remaining)).array().square() / d(cols_remaining - 1);*/

            d.head(cols_remaining - 1) -= (V.col(cols_remaining - 1).transpose() * 
                                      VVT_invV.leftCols(cols_remaining - 1)).array().square() / d(cols_remaining - 1);

            VVT_invV.leftCols(cols_remaining) += VVT_invV.col(cols_remaining - 1) *
                                                 (VVT_invV.col(cols_remaining - 1).transpose() *
                                                 V.leftCols(cols_remaining)) / d(cols_remaining - 1);
        }
    }

    cols.resize(k);
    return cols;
}

template class GreedyRemoval2Selector<float>;
template class GreedyRemoval2Selector<double>;