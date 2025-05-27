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
          6,  7,  8,  9, 10,
         11, 12, 13, 14, 15;
    // clang-format on

    // Checking bounds for each k and norm
    std::vector<MatSubset::Norm> norms{MatSubset::Norm::Frobenius,
                                       MatSubset::Norm::Spectral};
    for (auto norm : norms) {
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
    }
}