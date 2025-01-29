#include <vector>

namespace SubsetSelection {

template <typename scalar>
SpectralRemovalSelector<scalar>::SpectralRemovalSelector(scalar eps)
    : FrobeniusRemovalSelector<scalar>(eps) {}

template <typename scalar>
std::string SpectralRemovalSelector<scalar>::getAlgorithmName() const {

    return "spectral removal";
}

template <typename scalar>
std::vector<uint>
SpectralRemovalSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X,
                                              uint k) {

    Eigen::BDCSVD svd(X, Eigen::ComputeThinV);
    Eigen::MatrixX<scalar> V = svd.matrixV().transpose();
    return FrobeniusRemovalSelector<scalar>::selectSubset(V, k);
}

template <typename scalar>
scalar SpectralRemovalSelector<scalar>::boundInternal(uint m, uint n, uint k,
                                                       Norm norm) const {
    scalar bound;
    if (norm == Norm::L2) {
        bound = 1 / (1 + (scalar)(m * (n - k)) / (k - m + 1));
    } else if (norm == Norm::Frobenius) {
        bound = (scalar)(k - m + 1) / (n - m + 1) / m;
    }

    return bound;
}

} // namespace SubsetSelection