#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/QR>

namespace SubsetSelection {

template <typename scalar>
SpectralSelection2Selector<scalar>::SpectralSelection2Selector(scalar eps)
    : eps(eps) {}

template <typename scalar>
std::string SpectralSelection2Selector<scalar>::getAlgorithmName() const {

    return "spectral selection 2";
}

template <typename scalar>
scalar SpectralSelection2Selector<scalar>::computeEpsilon(
    scalar l, const Eigen::ArrayX<scalar> &eigenvalues) const {

    assert(l < eigenvalues(0) && "l must be less then smallest eigenvalue");
    return (eigenvalues - l).inverse().sum();
}

template <typename scalar>
scalar SpectralSelection2Selector<scalar>::computeDelta(uint m, uint i, uint n,
                                                        scalar l,
                                                        scalar epsilon) const {

    assert(
        i < n &&
        "i must be less then n, where i is number of already selected columns");

    scalar a = epsilon / m;
    scalar b = -1 - epsilon * (1 - l - m / epsilon) / (n - i);
    scalar c = (1 - l - m / epsilon) / (n - i);

    scalar D = b * b - 4 * a * c;
    return (-b - std::sqrt(D)) / (2 * a);
}

template <typename scalar>
scalar SpectralSelection2Selector<scalar>::computeBound(
    uint m, uint i, uint k, uint n, scalar l,
    const Eigen::ArrayX<scalar> &eigenvalues) const {

    assert(i <= k && "i must be less or equal to k, where i is number of "
                     "already selected columns");

    scalar epsilon = computeEpsilon(l, eigenvalues);
    for (uint j = i; j < k; ++j) {
        l += computeDelta(m, j, n, l, epsilon);
    }

    return l + 1 / (epsilon - (m - 1) / (1 - l));
}

template <typename scalar>
scalar SpectralSelection2Selector<scalar>::optimizeBound(
    uint m, uint i, uint k, uint n,
    const Eigen::ArrayX<scalar> &eigenvalues) const {

    const scalar GOLDEN_RATIO = (1 + std::sqrt(5)) / 2;

    scalar lim_a = -std::sqrt((k - i + 1) * m);
    scalar lim_b = eigenvalues(0);

    do {
        double ratio = (lim_b - lim_a) / GOLDEN_RATIO;
        scalar c = lim_b - ratio;
        scalar d = lim_a + ratio;

        if (computeBound(m, i, k, n, c, eigenvalues) >
            computeBound(m, i, k, n, d, eigenvalues)) {
            // select left section
            lim_b = d;
        } else {
            // select right section
            lim_a = c;
        }
    } while (std::abs(lim_a - lim_b) > eps);

    return (lim_a + lim_b) / 2;
}

template <typename scalar>
scalar SpectralSelection2Selector<scalar>::minimizeL(
    uint m, uint i, uint k, uint n, scalar l_opt, scalar bound,
    const Eigen::ArrayX<scalar> &eigenvalues) const {

    scalar left = -std::sqrt((k - i + 1) * m);
    scalar right = l_opt;

    do {
        scalar mid = (left + right) / 2;

        if (computeBound(m, i, k, n, mid, eigenvalues) > bound) {
            // select left section
            right = mid;
        } else {
            // select right section
            left = mid;
        }
    } while (std::abs(left - right) > eps);

    return (left + right) / 2;
}

template <typename scalar>
std::vector<uint> SpectralSelection2Selector<scalar>::selectSubset(
    const Eigen::MatrixX<scalar> &X, uint k) {
    const uint m = X.rows();
    const uint n = X.cols();

    Eigen::ColPivHouseholderQR<Eigen::MatrixX<scalar>> qr(X.transpose());
    Eigen::MatrixX<scalar> Q_full = qr.matrixQ();
    Eigen::MatrixX<scalar> V = Q_full.leftCols(m).transpose();

    std::vector<uint> cols_remaining(n);
    for (uint j = 0; j < V.cols(); ++j) {
        cols_remaining[j] = j;
    }

    std::vector<uint> cols_selected;
    cols_selected.reserve(k);

    Eigen::MatrixX<scalar> Y = Eigen::MatrixX<scalar>::Zero(m, m);
    Eigen::MatrixX<scalar> U = Eigen::MatrixX<scalar>::Identity(m, m);
    Eigen::ArrayX<scalar> S = Eigen::ArrayX<scalar>::Zero(m);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(m);

    scalar bound = this->template bound<Norm::L2>(m, n, k);

    for (uint i = 0; i < k; ++i) {
        scalar l_opt = optimizeBound(m, i, k, n, S);
        scalar l;
        if (i < k - m) {
            l = minimizeL(m, i, k, n, l_opt, bound, S);
        } else {
            scalar l_min = minimizeL(m, i, k, n, l_opt, bound, S);
            l = l_min * (k - 1 - i) / m + l_opt * (i + m - k + 1) / m;
        }
        //l += (l_opt - l) * i / (k - 1);
        scalar epsilon = computeEpsilon(l, S);
        scalar delta = computeDelta(m, i, n, l, epsilon);

        //std::cout << epsilon << ' ';

        Eigen::ArrayX<scalar> D = (S - (l + delta)).inverse();
        Eigen::MatrixX<scalar> M_1 = D.matrix().asDiagonal();
        Eigen::MatrixX<scalar> M_2 = D.square().matrix().asDiagonal();
        Eigen::MatrixX<scalar> M_3 = U.transpose() * V;
        Eigen::ArrayX<scalar> Phi =
            -(M_3.transpose() * M_2 * M_3).diagonal().array() /
            (1 + (M_3.transpose() * M_1 * M_3).diagonal().array());

        uint j_min;
        Phi.minCoeff(&j_min);
        Y += V.col(j_min) * V.col(j_min).transpose();

        cols_selected.push_back(cols_remaining[j_min]);
        cols_remaining[j_min] = cols_remaining.back();
        cols_remaining.pop_back();
        V.col(j_min) = V.col(V.cols() - 1);
        V.conservativeResize(Eigen::NoChange, V.cols() - 1);

        decomposition.compute(Y);
        U = decomposition.eigenvectors();
        S = decomposition.eigenvalues().array();
    }

    //std::cout << std::endl;

    return cols_selected;
}

template <typename scalar>
scalar SpectralSelection2Selector<scalar>::boundInternal(uint m, uint n, uint k,
                                                         Norm norm) const {

    Eigen::ArrayX<scalar> eigenvalues = Eigen::ArrayX<scalar>::Zero(m);
    scalar l_opt = optimizeBound(m, 0, k, n, eigenvalues);

    return computeBound(m, 0, k, n, l_opt, eigenvalues);
}

} // namespace SubsetSelection