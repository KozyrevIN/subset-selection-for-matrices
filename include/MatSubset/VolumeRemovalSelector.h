#ifndef MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H
#define MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class VolumeRemovalSelector : public SelectorBase<scalar> {
  public:
    VolumeRemovalSelector(scalar eps = 1e-6) : eps(eps) {}

    std::string getAlgorithmName() const override { return "volume removal"; }

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

        Eigen::MatrixX<scalar> V_dag = (V * V.transpose()).inverse() * V;
        Eigen::ArrayX<scalar> d =
            1 - (V.transpose() * V_dag).diagonal().array();

        while (cols.size() > k) {
            Eigen::Index j_max;
            scalar d_max = d.maxCoeff(&j_max);

            Eigen::VectorX<scalar> w = V.col(j_max);
            Eigen::VectorX<scalar> w_dag = V_dag.col(j_max);

            removeByIdx(cols, d, V, V_dag, j_max);

            d -= (w.transpose() * V_dag).array().square() / d_max;
            V_dag += w_dag * (w_dag.transpose() * V) / d_max;
        }

        return cols;
    }

  private:
    scalar eps;

    void removeByIdx(std::vector<Eigen::Index> &cols, Eigen::ArrayX<scalar> &d,
                     Eigen::MatrixX<scalar> &V, Eigen::MatrixX<scalar> &V_dag,
                     Eigen::Index j) const {

        Eigen::Index new_size = cols.size() - 1;

        cols[j] = cols[new_size];
        cols.resize(new_size);

        d(j) = d(new_size);
        d.conservativeResize(new_size);

        V.col(j) = V.col(new_size);
        V.conservativeResize(Eigen::NoChange, new_size);

        V_dag.col(j) = V_dag.col(new_size);
        V_dag.conservativeResize(Eigen::NoChange, new_size);
    }
};

} // namespace MatSubset

#endif