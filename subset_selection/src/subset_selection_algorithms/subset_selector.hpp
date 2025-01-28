namespace SubsetSelection {

template <typename scalar> SubsetSelector<scalar>::SubsetSelector() {}

template <typename scalar>
std::string SubsetSelector<scalar>::getAlgorithmName() const {

    return "random columns";
}

template <typename scalar>
std::vector<uint>
SubsetSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X, uint k) {
    std::vector<uint> cols(k);
    for (int i = 0; i < k; ++i) {
        cols[i] = i;
    }

    return cols;
}

template <typename scalar>
template <Norm norm>
scalar SubsetSelector<scalar>::bound(const Eigen::MatrixX<scalar> &X, uint k) {
    static_assert(norm == Norm::Frobenius || norm == Norm::L2,
                  "This norm is unsopported!");
    return boundInternal(X.rows(), X.cols(), k, norm);
}

template <typename scalar>
template <Norm norm>
scalar SubsetSelector<scalar>::bound(uint m, uint n, uint k) {
    static_assert(norm == Norm::Frobenius || norm == Norm::L2,
                  "This norm is unsopported!");
    return boundInternal(m, n, k, norm);
}

template <typename scalar>
scalar SubsetSelector<scalar>::boundInternal(uint m, uint n, uint k,
                                             Norm norm) {
    return 0;
}

} // namespace SubsetSelection