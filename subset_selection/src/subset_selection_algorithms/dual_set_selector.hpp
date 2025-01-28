namespace SubsetSelection {

template <typename scalar> DualSetSelector<scalar>::DualSetSelector() {}

template <typename scalar>
std::string DualSetSelector<scalar>::getAlgorithmName() const {

    return "dual set";
}

template <typename scalar>
Eigen::ArrayX<scalar> DualSetSelector<scalar>::calculateL(
    const Eigen::MatrixX<scalar> &V, scalar delta_l,
    const Eigen::MatrixX<scalar> &A, scalar l) const {

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(A);
    Eigen::MatrixX<scalar> U = decomposition.eigenvectors();
    Eigen::ArrayX<scalar> S = decomposition.eigenvalues().array();

    Eigen::ArrayX<scalar> D = (S - (l + delta_l)).inverse();
    Eigen::MatrixX<scalar> M_1 = D.matrix().asDiagonal();
    Eigen::MatrixX<scalar> M_2 =
        (D.square() / (D.sum() - (S - l).inverse().sum()))
            .matrix()
            .asDiagonal();
    return (V.transpose() * U * (M_2 - M_1) * U.transpose() * V).diagonal();
}

template <typename scalar>
Eigen::ArrayX<scalar> DualSetSelector<scalar>::calculateU(
    scalar delta_u, const Eigen::ArrayX<scalar> &B, scalar u) const {

    return ((u + delta_u) - B).inverse() +
           ((u + delta_u) - B).inverse().square() /
               ((u - B).inverse().sum() - ((u + delta_u) - B).inverse().sum());
}

template <typename scalar>
std::vector<uint>
DualSetSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X, uint k) {

    uint m = X.rows();
    uint n = X.cols();

    Eigen::MatrixX<scalar> V = X;
    Eigen::MatrixX<scalar> A = Eigen::MatrixX<scalar>::Zero(m, m);
    Eigen::VectorX<scalar> s = Eigen::VectorX<scalar>::Zero(n);

    scalar delta_l = 1;
    scalar l = -std::sqrt(k * m);

    scalar delta_u =
        (std::sqrt(n) + std::sqrt(k)) / (std::sqrt(k) - std::sqrt(m));
    scalar u = delta_u * std::sqrt(k * n);

    for (uint i = 0; i < k; ++i) {
        Eigen::VectorX<scalar> L = calculateL(V, delta_l, A, l);
        Eigen::VectorX<scalar> U = calculateU(delta_u, s, u);

        l += delta_l;
        u += delta_u;

        uint max_idx;
        (L - U).maxCoeff(&max_idx);
        scalar t = 2 / (L(max_idx) + U(max_idx));

        s(max_idx) += t;
        A += t * V.col(max_idx) * V.col(max_idx).transpose();
    }

    std::vector<uint> indices;
    for (uint i = 0; i < s.size(); i++) {
        if (s(i) > 0) {
            indices.push_back(i);
        }
    }

    uint i = 0;
    while (indices.size() < k) {
        if (s(i) <= 0) {
            indices.push_back(i);
        }
        ++i;
    }

    return indices;
}

template <typename scalar>
scalar DualSetSelector<scalar>::bound(uint m, uint n, uint k, Norm norm) const {
    return (std::pow(k, 0.5) - std::pow(m, 0.5)) /
           (std::pow(n, 0.5) - std::pow(k, 0.5));
}

} // namespace SubsetSelection