#include <cassert>
#include <iostream>
#include <list>
#include <set>

namespace SubsetSelection {

// MatrixGenerator class

template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n) : matrixSize{m, n} {

    std::random_device rd;
    gen = std::mt19937(rd());
}

template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n, int seed)
    : matrixSize{m, n} {

    gen = std::mt19937(seed);
}

template <typename scalar>
std::pair<uint, uint> MatrixGenerator<scalar>::getMatrixSize() const {

    return matrixSize;
}

template <typename scalar>
std::string MatrixGenerator<scalar>::getMatrixType() const {

    return "not specified";
}

template <typename scalar>
Eigen::MatrixX<scalar> MatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = matrixSize;
    return Eigen::MatrixX<scalar>(m, n);
}

// GaussianMatrixGenerator class

template <typename scalar>
GaussianMatrixGenerator<scalar>::GaussianMatrixGenerator(uint m, uint n)
    : MatrixGenerator<scalar>(m, n) {}

template <typename scalar>
GaussianMatrixGenerator<scalar>::GaussianMatrixGenerator(uint m, uint n,
                                                         int seed)
    : MatrixGenerator<scalar>(m, n, seed) {}

template <typename scalar>
std::string GaussianMatrixGenerator<scalar>::getMatrixType() const {
    return "random matrix with i.i.d. gaussian entries";
}

template <typename scalar>
Eigen::MatrixX<scalar> GaussianMatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;

    std::normal_distribution<scalar> dis(0.0, 1.0);
    Eigen::MatrixX<scalar> X(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            X(i, j) = dis(MatrixGenerator<scalar>::gen);
        }
    }

    return X;
}

// OrthonormalVectorsMatrixGenerator class

template <typename scalar>
OrthonormalVectorsMatrixGenerator<scalar>::OrthonormalVectorsMatrixGenerator(
    uint m, uint n)
    : GaussianMatrixGenerator<scalar>(m, n) {}

template <typename scalar>
OrthonormalVectorsMatrixGenerator<scalar>::OrthonormalVectorsMatrixGenerator(
    uint m, uint n, int seed)
    : GaussianMatrixGenerator<scalar>(m, n, seed) {}

template <typename scalar>
std::string OrthonormalVectorsMatrixGenerator<scalar>::getMatrixType() const {
    return "matrix with orthonormal rows/columns";
}

template <typename scalar>
Eigen::MatrixX<scalar>
OrthonormalVectorsMatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;

    bool orthonormal_rows = false;
    if (m < n) {
        orthonormal_rows = true;
        std::swap(m, n);
    }

    Eigen::MatrixX<scalar> tmp(m, n);
    std::normal_distribution<scalar> dis(0.0, 1.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tmp(i, j) = dis(MatrixGenerator<scalar>::gen);
        }
    }

    Eigen::HouseholderQR<Eigen::MatrixX<scalar>> qr(tmp);
    Eigen::MatrixX<scalar> Qfull = qr.householderQ();
    Eigen::MatrixX<scalar> Q = Qfull.leftCols(n);
    Eigen::VectorX<scalar> Rdiag = qr.matrixQR().diagonal();

    for (int i = 0; i < n; ++i) {
        if (std::abs(Rdiag(i) != 0)) {
            Q.row(i) *= Rdiag(i) / std::abs(Rdiag(i));
        }
    }

    if (orthonormal_rows) {
        return Q.transpose();
    } else {
        return Q;
    }
}

// SigmaMatrixGenerator class

template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(
    uint m, uint n, const Eigen::VectorX<scalar> &sigma)
    : sigma(sigma), MatrixGenerator<scalar>(m, n) {

    assert(sigma.size() <= std::min(m, n) &&
           "Invalid number of singular values, must be less or equal to the "
           "smallest side of a matrix");
}

template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(
    uint m, uint n, int seed, const Eigen::VectorX<scalar> &sigma)
    : sigma(sigma), MatrixGenerator<scalar>(m, n, seed) {

    assert(sigma.size() <= std::min(m, n) &&
           "Invalid number of singular values, must be less or equal to the "
           "smallest side of a matrix");
}

template <typename scalar>
std::string SigmaMatrixGenerator<scalar>::getMatrixType() const {
    std::stringstream ss;
    ss << sigma.transpose();
    return "matrix with given $\\sigma$, $\\sigma = " + ss.str() + "$";
}

template <typename scalar>
Eigen::MatrixX<scalar> SigmaMatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;
    uint k = sigma.size();

    OrthonormalVectorsMatrixGenerator<scalar> u_gen(
        m, k, MatrixGenerator<scalar>::gen());
    OrthonormalVectorsMatrixGenerator<scalar> v_gen(
        n, k, MatrixGenerator<scalar>::gen());

    return u_gen.generateMatrix() * sigma.asDiagonal() *
           v_gen.generateMatrix().adjoint();
}

// NearRankOneMatrixGenerator class

template <typename scalar>
Eigen::VectorX<scalar>
NearRankOneMatrixGenerator<scalar>::getSigma(uint m, uint n, scalar eps) const {

    Eigen::VectorX<scalar> sigma =
        Eigen::VectorX<scalar>::Constant(std::min(m, n), eps);
    sigma(0) = 1;
    return sigma;
}

template <typename scalar>
NearRankOneMatrixGenerator<scalar>::NearRankOneMatrixGenerator(uint m, uint n,
                                                               scalar eps)
    : eps(eps), SigmaMatrixGenerator<scalar>(m, n, getSigma(m, n, eps)) {}

template <typename scalar>
NearRankOneMatrixGenerator<scalar>::NearRankOneMatrixGenerator(uint m, uint n,
                                                               scalar eps,
                                                               int seed)
    : eps(eps), SigmaMatrixGenerator<scalar>(m, n, seed, getSigma(m, n, eps)) {}

template <typename scalar>
std::string NearRankOneMatrixGenerator<scalar>::getMatrixType() const {
    return "near rank-one matrix, $\\varepsilon = " + std::to_string(eps) + "$";
}

// NearSingularMatrixGenerator class

template <typename scalar>
Eigen::VectorX<scalar>
NearSingularMatrixGenerator<scalar>::getSigma(uint m, uint n,
                                              scalar eps) const {

    Eigen::VectorX<scalar> sigma =
        Eigen::VectorX<scalar>::Constant(std::min(m, n), 1);
    sigma(std::min(m, n) - 1) = eps;
    return sigma;
}

template <typename scalar>
NearSingularMatrixGenerator<scalar>::NearSingularMatrixGenerator(uint m, uint n,
                                                                 scalar eps)
    : eps(eps), SigmaMatrixGenerator<scalar>(m, n, getSigma(m, n, eps)) {}

template <typename scalar>
NearSingularMatrixGenerator<scalar>::NearSingularMatrixGenerator(uint m, uint n,
                                                                 scalar eps,
                                                                 int seed)
    : eps(eps), SigmaMatrixGenerator<scalar>(m, n, seed, getSigma(m, n, eps)) {}

template <typename scalar>
std::string NearSingularMatrixGenerator<scalar>::getMatrixType() const {
    return "near singular matrix, $\\varepsilon = " + std::to_string(eps) + "$";
}

// GraphIncidenceMatrixGenerator class

template <typename scalar>
GraphIncidenceMatrixGenerator<scalar>::GraphIncidenceMatrixGenerator(uint m,
                                                                     uint n)
    : MatrixGenerator<scalar>(m, n) {

    assert(m - 1 <= n &&
           "n must be larger of equal to m since m + 1 is number of edges in "
           "constructed connected graph and n is a number of edges");
}

template <typename scalar>
GraphIncidenceMatrixGenerator<scalar>::GraphIncidenceMatrixGenerator(uint m,
                                                                     uint n,
                                                                     int seed)
    : MatrixGenerator<scalar>(m, n, seed) {

    assert(m - 1 <= n &&
           "n must be larger of equal to m since m + 1 is number of edges in "
           "constructed connected graph and n is a number of edges");
}

template <typename scalar>
std::string GraphIncidenceMatrixGenerator<scalar>::getMatrixType() const {
    return "graph incidence matrix";
}

template <typename scalar>
std::vector<std::pair<uint, uint>>
GraphIncidenceMatrixGenerator<scalar>::randomEdgeList() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;
    uint num_v = m + 1;
    uint num_e = n;

    std::vector<std::pair<uint, uint>> edge_list;
    edge_list.reserve((num_v * (num_v - 1) / 2));

    for (uint i = 0; i < num_v - 1; ++i) {
        for (uint j = i + 1; j < num_v; ++j) {
            edge_list.push_back(std::make_pair(i, j));
        }
    }

    std::shuffle(edge_list.begin(), edge_list.end(),
                 MatrixGenerator<scalar>::gen);
    edge_list.resize(num_e);

    return edge_list;
}

template <typename scalar>
bool GraphIncidenceMatrixGenerator<scalar>::checkConnectivity(
    const std::vector<std::pair<uint, uint>> &edge_list) const {

    auto [v_1, v_2] = edge_list[0];
    std::set<uint> seed{v_1, v_2};
    std::list<std::set<uint>> components{seed};

    for (uint i = 1; i < edge_list.size(); ++i) {
        auto [v_1, v_2] = edge_list[i];
        auto v_1_component = components.begin();
        auto v_2_component = components.begin();

        bool v_1_found = false;
        bool v_2_found = false;
        while (!((v_1_found && v_2_found) ||
                 v_1_component == components.end() ||
                 v_2_component == components.end())) {

            if (!v_1_found) {
                if (v_1_component->contains(v_1)) {
                    v_1_found = true;
                } else {
                    ++v_1_component;
                }
            }

            if (!v_2_found) {
                if (v_2_component->contains(v_2)) {
                    v_2_found = true;
                } else {
                    ++v_2_component;
                }
            }
        }

        if (v_1_found && v_2_found) {
            if (v_1_component != v_2_component) {
                v_1_component->merge(*v_2_component);
                components.erase(v_2_component);
            }
        } else if (v_1_found) {
            v_1_component->insert(v_2);
        } else if (v_2_found) {
            v_2_component->insert(v_1);
        } else {
            components.push_back(std::set<uint>{v_1, v_2});
        }
    }

    if (components.size() == 1) {
        return true;
    } else {
        return false;
    }
}

template <typename scalar>
Eigen::MatrixX<scalar>
GraphIncidenceMatrixGenerator<scalar>::incidenceMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;

    auto edge_list = randomEdgeList();
    while (!checkConnectivity(edge_list)) {
        std::cerr << "Generated graph isn't connected, trying again"
                  << std::endl;
        edge_list = randomEdgeList();
    }

    Eigen::MatrixX<scalar> M = Eigen::MatrixX<scalar>::Zero(m + 1, n);
    for (uint j = 0; j < n; ++j) {
        auto [v_1, v_2] = edge_list[j];
        M(v_1, j) = 1;
        M(v_2, j) = -1;
    }

    return M;
}

template <typename scalar>
Eigen::MatrixX<scalar> GraphIncidenceMatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;
    
    Eigen::MatrixX<scalar> M = incidenceMatrix();
    Eigen::BDCSVD svd(M, Eigen::ComputeThinV);
    return svd.matrixV().transpose().topRows(m);
}

// WeightedGraphIncidenceMatrixGenerator class

template <typename scalar>
WeightedGraphIncidenceMatrixGenerator<
    scalar>::WeightedGraphIncidenceMatrixGenerator(uint m, uint n)
    : GraphIncidenceMatrixGenerator<scalar>(m, n) {}

template <typename scalar>
WeightedGraphIncidenceMatrixGenerator<
    scalar>::WeightedGraphIncidenceMatrixGenerator(uint m, uint n, int seed)
    : GraphIncidenceMatrixGenerator<scalar>(m, n, seed) {}

template <typename scalar>
std::string
WeightedGraphIncidenceMatrixGenerator<scalar>::getMatrixType() const {
    return "weighted graph incidence matrix";
}

template <typename scalar>
Eigen::MatrixX<scalar>
WeightedGraphIncidenceMatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;

    std::uniform_real_distribution<scalar> dis(0, 1);
    Eigen::VectorX<scalar> weights(n);
    weights.setConstant(1);
    for (uint i = 0; i < n; ++i) {
        weights(i) -= dis(MatrixGenerator<scalar>::gen);
    }

    Eigen::MatrixX<scalar> M =
        GraphIncidenceMatrixGenerator<scalar>::incidenceMatrix() *
        weights.cwiseSqrt().asDiagonal();

    Eigen::BDCSVD svd(M, Eigen::ComputeThinV);
    return svd.matrixV().transpose().topRows(m);
}

// SmoluchowskiMatrixGenerator class

template <typename scalar>
SmoluchowskiMatrixGenerator<scalar>::SmoluchowskiMatrixGenerator(uint m, uint n)
    : MatrixGenerator<scalar>(m, n) {}

template <typename scalar>
SmoluchowskiMatrixGenerator<scalar>::SmoluchowskiMatrixGenerator(uint m, uint n,
                                                                 int seed)
    : MatrixGenerator<scalar>(m, n, seed) {}

template <typename scalar>
std::string SmoluchowskiMatrixGenerator<scalar>::getMatrixType() const {
    return "Smoluchowski equations matrix";
}

template <typename scalar>
Eigen::MatrixX<scalar> SmoluchowskiMatrixGenerator<scalar>::generateMatrix() {

    auto [m, n] = MatrixGenerator<scalar>::matrixSize;

    Eigen::MatrixX<scalar> M(n, n);
    for (uint i = 1; i <= n; ++i) {
        for (uint j = 1; j <= n; ++j) {
            M(i - 1, j - 1) =
                std::pow(std::pow(i, 1.d / 3) + std::pow(j, 1.d / 3), 2) *
                std::sqrt(1.d / i + 1.d / j);
        }
    }

    Eigen::BDCSVD svd(M, Eigen::ComputeFullV);
    return svd.matrixV().transpose().topRows(m);
}

} // namespace SubsetSelection
