#ifndef MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class FrobeniusRemovalSelector : public SelectorBase<scalar> {
  public:
    FrobeniusRemovalSelector(scalar eps = 1e-6) : eps(eps) {}

    std::string getAlgorithmName() const override {

        return "frobenius removal";
    }

    std::vector<Eigen::Index> selectSubset(const Eigen::MatrixX<scalar> &X,
                                           Eigen::Index k) override {

        Eigen::Index m = X.rows();
        Eigen::Index n = X.cols();

        std::vector<Eigen::Index> cols(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            cols[j] = j;
        }

        Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<scalar> V = svd.matrixV().transpose();
        Eigen::VectorX<scalar> S_inv2 =
            svd.singularValues().array().inverse().square();

        Eigen::MatrixX<scalar> V_dag = (V * V.transpose()).inverse() * V;
        Eigen::ArrayX<scalar> l =
            (V_dag.transpose() * S_inv2.asDiagonal() * V_dag).diagonal();
        Eigen::ArrayX<scalar> d =
            1 - (V.transpose() * V_dag).diagonal().array();

        while (cols.size() > k) {

            Eigen::Index j_min = 0;
            for (Eigen::Index j = 0; j < cols.size(); ++j) {
                if (d(j) > eps and l(j) * d(j_min) < l(j_min) * d(j)) {
                    j_min = j;
                }
            }

            Eigen::VectorX<scalar> w = V.col(j_min);
            Eigen::VectorX<scalar> w_dag = V_dag.col(j_min);
            scalar d_min = d(j_min);

            removeByIdx(cols, l, d, V, V_dag, j_min);

            Eigen::ArrayX<scalar> mul_1 = w.transpose() * V_dag;
            Eigen::ArrayX<scalar> mul_2 =
                w_dag.transpose() * S_inv2.asDiagonal() * V_dag;

            l += 2 * mul_1 * mul_2 / d_min +
                 mul_1.square() * mul_2(cols.size() - 1) / (d_min * d_min);
            d -= (w.transpose() * V_dag).array().square() / d_min;

            V_dag += w_dag * (w_dag.transpose() * V) / d_min;
        }

        return cols;
    }

  private:
    scalar eps;

    void removeByIdx(std::vector<Eigen::Index> &cols, Eigen::ArrayX<scalar> &l,
                     Eigen::ArrayX<scalar> &d, Eigen::MatrixX<scalar> &V,
                     Eigen::MatrixX<scalar> &V_dag, Eigen::Index j) const {

        Eigen::Index new_size = cols.size() - 1;

        cols[j] = cols[new_size];
        cols.resize(new_size);

        l(j) = l(new_size);
        l.conservativeResize(new_size);

        d(j) = d(new_size);
        d.conservativeResize(new_size);

        V.col(j) = V.col(new_size);
        V.conservativeResize(Eigen::NoChange, new_size);

        V_dag.col(j) = V_dag.col(new_size);
        V_dag.conservativeResize(Eigen::NoChange, new_size);
    }

    scalar boundInternal(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                         Norm norm) const override {
        scalar bound = (scalar)(k - m + 1) / (scalar)(n - m + 1);
        if (norm == Norm::L2) {
            bound /= m;
        }

        return bound;
    }
};

} // namespace MatSubset

#endif