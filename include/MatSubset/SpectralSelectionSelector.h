#ifndef MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H
#define MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H

#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class SpectralSelectionSelector : public SelectorBase<scalar> {
  public:
    SpectralSelectionSelector(scalar eps = 1e-4) : eps(eps) {}

    std::string getAlgorithmName() const override {

        return "spectral selection";
    }

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &X,
                                   uint k) override {
        uint m = X.rows();
        uint n = X.cols();

        Eigen::ColPivHouseholderQR<Eigen::MatrixX<scalar>> qr(X.transpose());
        Eigen::MatrixX<scalar> Q_full = qr.matrixQ();
        Eigen::MatrixX<scalar> V = Q_full.leftCols(m).transpose();

        std::vector<uint> cols_remaining(n);
        for (uint j = 0; j < V.cols(); ++j) {
            cols_remaining[j] = j;
        }

        std::vector<uint> cols_selected;
        cols_selected.reserve(k);

        Eigen::MatrixX<scalar> Y = Eigen::MatrixX<scalar>::Zero(m, m);
        Eigen::MatrixX<scalar> U = Eigen::MatrixX<scalar>::Identity(m, m);
        Eigen::ArrayX<scalar> S = Eigen::ArrayX<scalar>::Zero(m);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(m);

        scalar epsilon = calculateEpsilon(m, n, k);
        scalar l_0 = -(m / epsilon);
        scalar l = l_0;

        while (cols_selected.size() < k) {
            scalar delta =
                calculateDelta(m, n, k, epsilon, l, cols_remaining.size());

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

            auto f = [&S, &epsilon](scalar l) {
                return (S - l).inverse().sum() - epsilon;
            };
            l = binarySearch(l, S(0), f, delta * eps);
        }

        return cols_selected;
    }

  private:
    scalar eps;

    scalar calculateEpsilon(uint m, uint n, uint k) const {
        scalar epsilon;
        if (m == 1) {
            epsilon = 0.5;
        } else {
            scalar alpha = std::sqrt((k - 1) * m + 1);
            epsilon = n *
                      (2 * (alpha - 1) +
                       m * (k * (alpha + m - 2) - 2 * alpha - m + 3)) /
                      ((k - 1) * m * (k - m + 1));
        }
        return epsilon;
    }

    scalar calculateDelta(uint m, uint n, uint k, scalar epsilon, scalar l,
                          uint cols_remaining_size) const {

        scalar a = epsilon / m;
        scalar b = -1 - epsilon * (1 - l - m / epsilon) / cols_remaining_size;
        scalar c = (1 - l - m / epsilon) / cols_remaining_size;

        scalar D = b * b - 4 * a * c;
        return (-b - std::sqrt(D)) / (2 * a);
    }

    scalar binarySearch(scalar l, scalar r,
                        const std::function<scalar(scalar)> &f,
                        scalar tol) const {

        scalar f_l = f(l);
        scalar f_r = f(r);

        while (r - l > tol) {
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

    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override {
        scalar epsilon = calculateEpsilon(m, n, k);
        scalar l = -(m / epsilon);

        for (uint i = 0; i < k; ++i) {
            l += calculateDelta(m, n, k, epsilon, l, n - i);
        }

        return l + 1 / epsilon;
    }
};

} // namespace MatSubset

#endif