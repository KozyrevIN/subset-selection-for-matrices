#include <vector>
#include <functional>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <iostream>

namespace SubsetSelection
{

template <typename scalar> 
GreedySelection1Selector<scalar>::GreedySelection1Selector(): SubsetSelector<scalar>("greedy_selection_1") {
    //do nothing
}

template <typename scalar> 
scalar binary_search(scalar l, scalar r, const std::function<scalar(scalar)>& f, scalar eps) {
    scalar f_l = f(l); 
    scalar f_r = f(r);

    while (r - l > eps) {
        scalar m = (r + l) / 2;
        scalar f_m = f(m);

        if (f_m > 0) {
            r = m;
            f_r = f_m;
        } else {
            l = m;
            f_l = f_m;
        }
    }

    return (r + l) / 2;
}

template <typename scalar>
std::vector<uint> GreedySelection1Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    uint m = X.rows();
    uint n = X.cols();

    Eigen::MatrixX<scalar> V = X;

    std::vector<uint> cols(n);
    for (uint j = 0; j < X.cols(); ++j) {
        cols[j] = j;
    }

    if (k < n) {
        Eigen::MatrixX<scalar> Y = Eigen::MatrixX<scalar>::Zero(m, m);
        scalar eps = (n + 1) * std::pow(m - 1, 0.5) / (std::pow(k, 0.5) - std::pow(m -1, 0.5));
        scalar l_0 = -(m / eps);
        scalar l = l_0;
        Eigen::MatrixX<scalar> I = Eigen::MatrixX<scalar>::Identity(m , m);

        for (uint cols_selected = 0; cols_selected <= k; ++cols_selected) {
            scalar delta = 1 / (eps + (n - cols_selected) / (1 - (l - l_0)));
            Eigen::MatrixX<scalar> YmlI_invV = (Y - (l + delta) * I).inverse() * V.rightCols(n - cols_selected);
            Eigen::ArrayX<scalar> Phi = (Y - (l + delta) * I).inverse().trace() - 
                                        YmlI_invV.colwise().squaredNorm().transpose().array() /
                                        (1 + (V.rightCols(n - cols_selected).transpose() * YmlI_invV).diagonal().array());
            
            uint s;
            Phi.minCoeff(&s);
            std::swap(cols[cols_selected + s], cols[cols_selected]);
            V.col(cols_selected + s).swap(V.col(cols_selected));
            Y += V.col(cols_selected) * V.col(cols_selected).transpose();
            
            Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(Y);
            Eigen::ArrayX<scalar> S = svd.singularValues().array(); 
            
            auto f = [&S](scalar l){ return (S - l).inverse().sum(); };
            l = binary_search<scalar>(l, S(S.rows() - 1), f, 1e-6);
        }
    }
    
    cols.resize(k);
    return cols;
}

template <typename scalar>
scalar GreedySelection1Selector<scalar>::bound(uint m, uint n, uint k, Norm norm) {
    return (std::pow(k, 0.5) - std::pow(m - 1, 0.5)) / std::pow(n, 0.5);
}

}