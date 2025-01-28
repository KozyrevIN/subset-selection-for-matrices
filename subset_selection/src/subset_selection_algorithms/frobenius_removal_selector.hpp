#include <vector>

namespace SubsetSelection {

template <typename scalar>
FrobeniusRemovalSelector<scalar>::FrobeniusRemovalSelector(scalar eps)
    : eps(eps) {}

template <typename scalar>
std::string FrobeniusRemovalSelector<scalar>::getAlgorithmName() const {

    return "frobenius removal";
}

template <typename scalar>
void FrobeniusRemovalSelector<scalar>::removeByIdx(
    std::vector<uint> &cols, Eigen::ArrayX<scalar> &l, Eigen::ArrayX<scalar> &d,
    Eigen::MatrixX<scalar> &V, Eigen::MatrixX<scalar> &V_dag, uint j) const {

    uint new_size = cols.size() - 1;

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

template <typename scalar>
std::vector<uint>
FrobeniusRemovalSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X,
                                               uint k) {
    uint m = X.rows();
    uint n = X.cols();

    std::vector<uint> cols(n);
    for (uint j = 0; j < n; ++j) {
        cols[j] = j;
    }

    Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(X, Eigen::ComputeThinV);
    Eigen::MatrixX<scalar> V = svd.matrixV().transpose();
    Eigen::VectorX<scalar> S_inv2 =
        svd.singularValues().array().inverse().square();

    Eigen::MatrixX<scalar> V_dag = (V * V.transpose()).inverse() * V;
    Eigen::ArrayX<scalar> l =
        (V_dag.transpose() * S_inv2.asDiagonal() * V_dag).diagonal();
    Eigen::ArrayX<scalar> d = 1 - (V.transpose() * V_dag).diagonal().array();

    while (cols.size() > k) {

        uint j_min = 0;
        for (uint j = 0; j < cols.size(); ++j) {
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

template <typename scalar>
scalar FrobeniusRemovalSelector<scalar>::boundInternal(uint m, uint n, uint k,
                                                       Norm norm) const {
    scalar bound = std::sqrt((scalar)(k - m + 1) / (scalar)(n - m + 1));
    if (norm == Norm::L2) {
        bound /= std::sqrt(n);
    }

    return bound;
}

} // namespace SubsetSelection