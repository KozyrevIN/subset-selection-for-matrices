#include <eigen3/Eigen/Eigenvalues>
#include <functional>
#include <iostream>
#include <vector>

namespace SubsetSelection {

template <typename scalar>
SpectralSelectionSelector<scalar>::SpectralSelectionSelector()
    : SubsetSelector<scalar>("spectral_selection") {
    // do nothing
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::binarySearch(
    scalar l, scalar r, const std::function<scalar(scalar)> &f, scalar eps) {

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
std::vector<uint>
SpectralSelectionSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X,
                                                uint k) {
    uint m = X.rows();
    uint n = X.cols();

    Eigen::MatrixX<scalar> V = X;

    std::vector<uint> cols_remaining(n);
    for (uint j = 0; j < X.cols(); ++j) {
        cols_remaining[j] = j;
    }

    std::vector<uint> cols_selected;
    cols_selected.reserve(k);

    Eigen::MatrixX<scalar> Y = Eigen::MatrixX<scalar>::Zero(m, m);
    Eigen::MatrixX<scalar> U = Eigen::MatrixX<scalar>::Identity(m, m);
    Eigen::ArrayX<scalar> S = Eigen::ArrayX<scalar>::Zero(m);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(m);

    scalar eps = (n + 1) * std::sqrt(m - 1) / (std::sqrt(k) - std::sqrt(m - 1));
    scalar l_0 = -(m / eps);
    scalar l = l_0;

    while (cols_selected.size() < k) {
        scalar delta = 1 / (eps + (n - cols_selected.size()) / (1 - (l - l_0)));

        Eigen::MatrixX<scalar> M =
            U * (S - (l + delta)).inverse().matrix().asDiagonal() *
            U.transpose() * V;
        Eigen::ArrayX<scalar> Phi =
            (S - (l + delta)).inverse().sum() -
            M.colwise().squaredNorm().transpose().array() /
                (1 + (V.transpose() * M).diagonal().array());

        uint j_min;
        Phi.minCoeff(&j_min);
        Y += V.col(j_min) * V.col(j_min).transpose();

        cols_selected.push_back(cols_remaining[j_min]);
        cols_remaining[j_min] = cols_remaining.back();
        cols_remaining.pop_back();
        V.col(j_min) = V.col(V.cols() - 1);
        V.conservativeResize(Eigen::NoChange, V.cols() - 1);
        
        decomposition.compute(Y);
        U = decomposition.eigenvectors();
        S = decomposition.eigenvalues().array();

        auto f = [&S](scalar l) { return (S - l).inverse().sum(); };
        l = binarySearch(l, S(0), f, 1e-6);
    }

    return cols_selected;
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::bound(uint m, uint n, uint k,
                                                Norm norm) {
    return (std::pow(k, 0.5) - std::pow(m - 1, 0.5)) / std::pow(n, 0.5);
}

} // namespace SubsetSelection