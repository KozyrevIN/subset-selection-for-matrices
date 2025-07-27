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

    const scalar epsilon_0 = calculateEpsilon(m, n, k);
    scalar epsilon = epsilon_0;
    scalar l = -(m / epsilon);
    scalar delta = calculateDelta(m, n, k, epsilon, l, n);
    scalar bound = l + k * delta + 1 / epsilon;

    scalar epsilon_trial;
    scalar l_trial;
    scalar delta_trial;
    scalar bound_trial;

    while (cols_selected.size() < k) {
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
            epsilon_trial = epsilon_0;
            auto f = [&S, &epsilon_trial](scalar l) {
                return (S - l).inverse().sum() - epsilon_trial;
            };
            l_trial = binarySearch(l, S(0), f, eps / epsilon_trial);
        } else {
            l_trial = l + delta;
            epsilon_trial = (S - l_trial).inverse().sum();
        }

        delta_trial = calculateDelta(m, n, k, epsilon_trial, l_trial, cols_remaining.size());
        bound_trial = l_trial + (k - cols_selected.size()) * delta_trial + 1/epsilon_trial;

        if (bound_trial >= bound) {
            l = l_trial;
            epsilon = epsilon_trial;
            delta = delta_trial;
        } else {
            auto f = [&S, &epsilon](scalar l) {
                return (S - l).inverse().sum() - epsilon;
            };
            l = binarySearch(l, S(0), f, eps / epsilon);
            delta = calculateDelta(m, n, k, epsilon, l, cols_remaining.size());
        }

        // std::cout << epsilon << ' ';
    }

    // std::cout << std::endl;

    return cols_selected;
}

template <typename scalar>
scalar SpectralSelectionSelector<scalar>::boundInternal(uint m, uint n, uint k,
                                                        Norm norm) const {
    if (m == 1) {
        return std::pow(std::sqrt(k) - std::sqrt(m - 1), 2) / n;
    } else {
        scalar alpha = std::sqrt((k - 1) * m + 1);
        return static_cast<scalar>(m) / static_cast<scalar>(n) *
               std::pow((static_cast<scalar>(k) - alpha) /
                            (alpha - static_cast<scalar>(1)),
                        2);
    }
}

} // namespace SubsetSelection