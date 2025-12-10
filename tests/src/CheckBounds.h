#include <cmath>
#include <vector>

#include <doctest/doctest.h>

#include <Eigen/Core>

#include <MatSubset/Enums.h>
#include <MatSubset/SelectorBase.h>
#include <MatSubset/Utils.h>

template <typename Scalar>
void check_bounds(MatSubset::SelectorBase<Scalar> *selector,
                  Eigen::Index k_begin, Eigen::Index k_end) {

    // Matrix Setup
    const Eigen::Index m = 3;
    const Eigen::Index n = 5;

    Eigen::MatrixX<Scalar> X(m, n);
    // clang-format off
    X <<  1,  2,  3,  4,  5,
          0,  6,  7,  8,  9,
          0,  0, 10, 11, 12;
    // clang-format on

    // Helper lambda to check bounds for a specific norm
    auto check_for_norm = [&](auto norm_constant) {
        constexpr MatSubset::Norm norm = norm_constant;
        for (Eigen::Index k = k_begin; k <= k_end; ++k) {
            SUBCASE("Check bounds for given k and norm") {
                std::vector<Eigen::Index> indices =
                    selector->selectSubset(X, k);
                Eigen::MatrixX<Scalar> X_S = X(Eigen::all, indices);

                Scalar X_pinv = MatSubset::Utils::pinv_norm<Scalar, norm>(X);
                Scalar X_S_pinv =
                    MatSubset::Utils::pinv_norm<Scalar, norm>(X_S);

                Scalar X_pinv_sqr = std::pow(X_pinv, 2);
                Scalar X_S_pinv_sqr = std::pow(X_S_pinv, 2);

                Scalar bound_1 = selector->template bound<norm>(X, k);
                Scalar bound_2 =
                    selector->template bound<norm>(X.rows(), X.cols(), k);

                CHECK(bound_1 == bound_2);
                CHECK(X_pinv_sqr / X_S_pinv_sqr >= bound_1);
            }
        }
    };

    // Check bounds for each norm type
    check_for_norm(
        std::integral_constant<MatSubset::Norm, MatSubset::Norm::Frobenius>{});
    check_for_norm(
        std::integral_constant<MatSubset::Norm, MatSubset::Norm::Spectral>{});
}