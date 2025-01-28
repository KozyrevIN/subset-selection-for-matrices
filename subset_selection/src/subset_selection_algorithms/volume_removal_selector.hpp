#include <vector>

namespace SubsetSelection {

template <typename scalar>
VolumeRemovalSelector<scalar>::VolumeRemovalSelector(scalar eps) : eps(eps) {}

template <typename scalar>
std::string VolumeRemovalSelector<scalar>::getAlgorithmName() const {

    return "volume removal";
}

template <typename scalar>
void VolumeRemovalSelector<scalar>::removeByIdx(std::vector<uint> &cols,
                                                Eigen::ArrayX<scalar> &d,
                                                Eigen::MatrixX<scalar> &V,
                                                Eigen::MatrixX<scalar> &V_dag,
                                                uint j) const {

    uint new_size = cols.size() - 1;

    cols[j] = cols[new_size];
    cols.resize(new_size);

    d(j) = d(new_size);
    d.conservativeResize(new_size);

    V.col(j) = V.col(new_size);
    V.conservativeResize(Eigen::NoChange, new_size);

    V_dag.col(j) = V_dag.col(new_size);
    V_dag.conservativeResize(Eigen::NoChange, new_size);
}

template <typename scalar>
std::vector<uint>
VolumeRemovalSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X,
                                            uint k) {
    uint m = X.rows();
    uint n = X.cols();

    std::vector<uint> cols(n);
    for (uint j = 0; j < n; ++j) {
        cols[j] = j;
    }

    Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(X, Eigen::ComputeThinV);
    Eigen::MatrixX<scalar> V = svd.matrixV().transpose();

    Eigen::MatrixX<scalar> V_dag = (V * V.transpose()).inverse() * V;
    Eigen::ArrayX<scalar> d = 1 - (V.transpose() * V_dag).diagonal().array();

    while (cols.size() > k) {
        uint j_max;
        scalar d_max = d.maxCoeff(&j_max);

        Eigen::VectorX<scalar> w = V.col(j_max);
        Eigen::VectorX<scalar> w_dag = V_dag.col(j_max);

        removeByIdx(cols, d, V, V_dag, j_max);

        d -= (w.transpose() * V_dag).array().square() / d_max;
        V_dag += w_dag * (w_dag.transpose() * V) / d_max;
    }

    return cols;
}

} // namespace SubsetSelection