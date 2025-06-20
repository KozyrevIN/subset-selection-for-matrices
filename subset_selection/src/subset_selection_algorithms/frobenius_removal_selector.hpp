#include <cassert>
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

    Eigen::MatrixX<scalar> V_dag = V;
    Eigen::ArrayX<scalar> l =
        (V_dag.transpose() * S_inv2.asDiagonal() * V_dag).diagonal();
    Eigen::ArrayX<scalar> d = (V.transpose() * V_dag).diagonal().array();

    while (cols.size() > k) {

        uint j_min = 0;
        for (; (j_min < cols.size()) && (d(j_min) >= 1 - eps); ++j_min);
        --j_min;
        assert(d(j_min) < 1 - eps &&
               "Have not found a column with d_j < 1 - eps.");

        for (uint j = j_min + 1; j < cols.size(); ++j) {
            if (d(j) < 1 - eps and
                l(j) + l(j_min) * d(j) < l(j_min) + l(j) * d(j_min)) {
                j_min = j;
            }
        }

        Eigen::VectorX<scalar> w = V.col(j_min);
        Eigen::VectorX<scalar> w_dag = V_dag.col(j_min);
        scalar denom = static_cast<scalar>(1) - d(j_min);

        removeByIdx(cols, l, d, V, V_dag, j_min);

        Eigen::ArrayX<scalar> mul_1 = w.transpose() * V_dag;
        Eigen::ArrayX<scalar> mul_2 =
            w_dag.transpose() * S_inv2.asDiagonal() * V_dag;
        scalar mul_3 =
            (w_dag.transpose() * S_inv2.asDiagonal() * w_dag).value();

        d += mul_1.square() / denom;
        mul_1 /= denom;
        l += mul_1 * (2 * mul_2 + mul_1 * mul_3);

        V_dag += w_dag * (w_dag.transpose() * V) / denom;
    }

    return cols;
}

template <typename scalar>
scalar FrobeniusRemovalSelector<scalar>::boundInternal(uint m, uint n, uint k,
                                                       Norm norm) const {
    scalar bound = (scalar)(k - m + 1) / (scalar)(n - m + 1);
    if (norm == Norm::L2) {
        bound /= m;
    }

    return bound;
}

} // namespace SubsetSelection