#include <vector>

namespace SubsetSelection {

template <typename scalar>
RankRevealingQRSelector<scalar>::RankRevealingQRSelector() {}

template <typename scalar>
std::string RankRevealingQRSelector<scalar>::getAlgorithmName() const {

    return "rank-revealing QR";
}

template <typename scalar>
std::vector<uint>
RankRevealingQRSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &x,
                                              uint k) {

    Eigen::ColPivHouseholderQR<Eigen::MatrixX<scalar>> qr(x);
    Eigen::MatrixX<scalar> P = qr.colsPermutation();

    std::vector<uint> vect(k);

    for (int j = 0; j < k; ++j) {
        int i = 0;
        for (; std::abs(P(i, j)) == 0; ++i)
            ;
        vect[j] = i;
    }

    std::sort(vect.begin(), vect.end());

    return vect;
}

} // namespace SubsetSelection
