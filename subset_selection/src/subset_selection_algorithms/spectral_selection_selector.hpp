#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/QR>
#include <vector>

namespace SubsetSelection {

template <typename scalar>
SpectralSelectionSelector<scalar>::SpectralSelectionSelector(scalar eps)
    : eps(eps) {}

template <typename scalar>
std::string SpectralSelectionSelector<scalar>::getAlgorithmName() const {

    return "spectral selection";
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::calculateEpsilon(uint m, uint n,
                                                           uint k) const {
    scalar epsilon;
    if (m == 1) {
        epsilon = 0.5;
    } else {
        scalar alpha = std::sqrt((k - 1) * m + 1);
        epsilon =
            n *
            (2 * (alpha - 1) + m * (k * (alpha + m - 2) - 2 * alpha - m + 3)) /
            ((k - 1) * m * (k - m + 1));
    }
    return epsilon;
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::calculateDelta(
    uint m, uint n, uint k, scalar epsilon, scalar l,
    uint cols_remaining_size) const {

    scalar a = epsilon / m;
    scalar b = -1 - epsilon * (1 - l - m / epsilon) / cols_remaining_size;
    scalar c = (1 - l - m / epsilon) / cols_remaining_size;

    scalar D = b * b - 4 * a * c;
    return (-b - std::sqrt(D)) / (2 * a);
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::binarySearch(
    scalar l, scalar r, const std::function<scalar(scalar)> &f,
    scalar tol) const {

    scalar f_l = f(l);
    scalar f_r = f(r);

    while (r - l > tol) {
        scalar m = (r + l) / 2;
        scalar f_m = f(m);

        if (f_m > 0) {
            r = m;
            f_r = f_m;
        } else {
            l = m;
            f_l = f_m;
        }
    }

    return (r + l) / 2;
}

template <typename scalar>
std::vector<uint>
SpectralSelectionSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar> &X,
                                                uint k) {
    uint m = X.rows();
    uint n = X.cols();

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

    scalar epsilon = calculateEpsilon(m, n, k);
    scalar l_0 = -(m / epsilon);
    scalar l = l_0;

    while (cols_selected.size() < k) {
        scalar phi_l_Y = (S - l).inverse().sum();

        //std::cout << phi_l_Y << ' ';

        scalar delta =
            calculateDelta(m, n, k, phi_l_Y, l, cols_remaining.size());

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

        if (cols_selected.size() == k - 1) {

            auto f = [&S, &epsilon](scalar l) {
                return (S - l).inverse().sum() - epsilon;
            };

            l = binarySearch(l, S(0), f, delta * eps);
        } else {
            l += delta;
        }
    }

    //std::cout << std::endl;

    return cols_selected;
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::boundInternal(uint m, uint n, uint k,
                                                        Norm norm) const {
    scalar epsilon = calculateEpsilon(m, n, k);
    scalar l = -(m / epsilon);

    for (uint i = 0; i < k; ++i) {
        l += calculateDelta(m, n, k, epsilon, l, n - i);
    }

    return l + 1 / epsilon;
}

} // namespace SubsetSelection