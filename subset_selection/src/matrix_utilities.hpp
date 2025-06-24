#include <cassert>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "../include/enums.h"

namespace SubsetSelection {

template <typename scalar, Norm norm>
scalar pinv_norm(Eigen::MatrixX<scalar> X) {
    Eigen::BDCSVD svd(X);
    Eigen::ArrayX<scalar> S = svd.singularValues();

    scalar tolerance = S(0) * std::numeric_limits<scalar>::epsilon();
    if (S(S.size() - 1) <= tolerance) {
        return std::numeric_limits<scalar>::infinity();
    }

    if constexpr (norm == Norm::Frobenius) {
        return std::sqrt(S.inverse().square().sum());
    } else if constexpr (norm == Norm::L2) {
        return static_cast<scalar>(1.0) / S(S.size() - 1);
    } else {
        static_assert(false, "This norm is unsupported!");
    }
}

} // namespace SubsetSelection