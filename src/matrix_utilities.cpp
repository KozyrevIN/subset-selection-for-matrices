#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <complex>

#include "../include/matrix_utilities.h"

template <typename scalar>
double pinv_frobenius_norm(Eigen::MatrixX<scalar> x) {
    return x.completeOrthogonalDecomposition().pseudoInverse().norm();
}

template double pinv_frobenius_norm<float>(Eigen::MatrixX<float> x);
template double pinv_frobenius_norm<double>(Eigen::MatrixX<double> x);
template double pinv_frobenius_norm<std::complex<float>>(Eigen::MatrixX<std::complex<float>> x);
template double pinv_frobenius_norm<std::complex<double>>(Eigen::MatrixX<std::complex<double>> x);

template <typename scalar>
double pinv_l_2_norm(Eigen::MatrixX<scalar> x) {
    auto pinv = x.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(pinv);
    return svd.singularValues()(0);
}

template double pinv_l_2_norm<float>(Eigen::MatrixX<float> x);
template double pinv_l_2_norm<double>(Eigen::MatrixX<double> x);
template double pinv_l_2_norm<std::complex<float>>(Eigen::MatrixX<std::complex<float>> x);
template double pinv_l_2_norm<std::complex<double>>(Eigen::MatrixX<std::complex<double>> x);