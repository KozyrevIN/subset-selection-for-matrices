#ifndef MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H
#define MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H

#include <functional>

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class InterlacingFamiliesSelector : public SelectorBase<scalar> {
  public:
    InterlacingFamiliesSelector(scalar eps = 1e-6) : eps(eps) {}

    std::string getAlgorithmName() const override {

        return "interlacing families";
    }

    std::vector<Eigen::Index> selectSubset(const Eigen::MatrixX<scalar> &x,
                                           Eigen::Index k) override {

        Eigen::Index m = X.rows();
        Eigen::Index n = X.cols();

        Eigen::BDCSVD svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<scalar> V = svd.matrixV().transpose();
        Eigen::MatrixX<scalar> Y = Eigen::MatrixX<scalar>::Zero(m, m);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(m);
        Eigen::PolynomialSolver<scalar, Eigen::Dynamic> poly_solver;

        std::vector<Eigen::Index> cols_remaining(n);
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            cols_remaining[j] = j;
        }

        std::vector<Eigen::Index> cols_selected;
        cols_selected.reserve(k);

        scalar lambda_max_prev = 0;

        for (Eigen::Index i = 1; i <= k; ++i) {
            Eigen::VectorX<scalar> lambdas(cols_remaining.size());

            Eigen::ArrayX<scalar> PtoF = PtoFArray(m, n, k, i);
            Eigen::Index f_len = PtoF.size();
            Eigen::MatrixX<scalar> YtoZ =
                YtoZMatrix(f_len, 1 - lambda_max_prev);

            for (Eigen::Index j = 0; j < cols_remaining.size(); ++j) {
                decomposition.compute(Y + V.col(j) * V.col(j).transpose(),
                                      Eigen::EigenvaluesOnly);
                Eigen::ArrayX<scalar> p_roots_x = decomposition.eigenvalues();
                // y = x - 1
                Eigen::ArrayX<scalar> p_roots_y = p_roots_x.array() - 1;
                Eigen::ArrayX<scalar> p_y = polyFromRoots(p_roots_y);
                Eigen::VectorX<scalar> f_y = PtoF * p_y.tail(f_len);
                // z = x - \lambda_m(Y) = y + 1 - \lambda_m(Y) obtained on
                // previous step changing the basis to make the problem of
                // finding roots well conditioned
                Eigen::VectorX<scalar> f_z = YtoZ * f_y;

                poly_solver.compute(f_z);
                bool has_root;
                lambdas(j) = poly_solver.smallestRealRoot(has_root, 0.01);
            }

            Eigen::Index j_max;
            lambdas.maxCoeff(&j_max);
            Y += V.col(j_max) * V.col(j_max).transpose();
            lambda_max_prev += lambdas(j_max);

            decomposition.compute(Y);

            cols_selected.push_back(cols_remaining[j_max]);
            cols_remaining[j_max] = cols_remaining.back();
            cols_remaining.pop_back();
            V.col(j_max) = V.col(V.cols() - 1);
            V.conservativeResize(Eigen::NoChange, V.cols() - 1);
        }

        return cols_selected;
    }

  private:
    scalar eps;

    Eigen::VectorX<scalar>
    polyFromRoots(const Eigen::VectorX<scalar> &roots) const {

        Eigen::Index l = roots.size();
        Eigen::VectorX<scalar> poly = Eigen::VectorX<scalar>::Zero(l + 1);
        poly(l) = 1;

        for (scalar root : roots) {
            poly.head(l) -= root * poly.tail(l);
        }

        return poly;
    }

    Eigen::ArrayX<scalar> PtoFArray(Eigen::Index m, Eigen::Index n,
                                    Eigen::Index k, Eigen::Index i) const {

        Eigen::ArrayX<scalar> arr;

        if (k <= n - m) {
            arr = Eigen::ArrayX<scalar>::Constant(m + 1, 1);
            for (Eigen::Index j = 1; j < m + 1; ++j) {
                arr(j) = arr(j - 1) * (j + n - m - i) / (j + n - m - k);
            }
        } else {
            arr = Eigen::ArrayX<scalar>::Constant(n - k + 1, 1);
            for (Eigen::Index j = 1; j < n - k + 1; ++j) {
                arr(j) = arr(j - 1) * (j + k - i) / j;
            }
        }

        return arr;
    }

    Eigen::MatrixX<scalar> YtoZMatrix(Eigen::Index m, scalar shift) const {

        Eigen::MatrixX<scalar> M = Eigen::MatrixX<scalar>::Zero(m, m);

        M(0, 0) = 1;
        for (Eigen::Index i = 1; i < m; ++i) {
            M.col(i).tail(m - 1) = M.col(i - 1).head(m - 1);
            M.col(i) -= shift * M.col(i - 1);
        }

        return M;
    }

    scalar boundInternal(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                         Norm norm) const override {

        return std::pow(
            (std::sqrt((k + 1) * (n - m)) - std::sqrt(m * (n - k - 1))) / n, 2);
    }
};

} // namespace MatSubset

#endif