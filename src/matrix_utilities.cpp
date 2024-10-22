#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <complex>

#include "../include/matrix_utilities.h"

template <typename scalar>
scalar pinv_frobenius_norm(Eigen::MatrixX<scalar> X) {
    return X.completeOrthogonalDecomposition().pseudoInverse().norm();
}

template float pinv_frobenius_norm<float>(Eigen::MatrixX<float> X);
template double pinv_frobenius_norm<double>(Eigen::MatrixX<double> X);

template <typename scalar>
scalar pinv_l2_norm(Eigen::MatrixX<scalar> X) {
    auto pinv = X.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(pinv);
    return svd.singularValues()(0);
}

template float pinv_l2_norm<float>(Eigen::MatrixX<float> X);
template double pinv_l2_norm<double>(Eigen::MatrixX<double> X);