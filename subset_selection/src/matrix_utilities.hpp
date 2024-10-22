#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>

namespace SubsetSelection
{

template <typename scalar, Norm norm>
scalar pinv_norm(Eigen::MatrixX<scalar> X) {
    if constexpr (norm == Norm::Frobenius) {
        return X.completeOrthogonalDecomposition().pseudoInverse().norm();
    } else if constexpr (norm == Norm::L2) {
        auto pinv = X.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(pinv);
        return svd.singularValues()(0);
    } else {
        static_assert(false, "This norm is unsupported!");
    }
}

}