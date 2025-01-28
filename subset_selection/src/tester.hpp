#include <chrono>
#include <filesystem>
#include <fstream>
#include <omp.h>

#include "../include/matrix_utilities.h"
#include "../include/tester.h"

namespace SubsetSelection {

template <typename scalar> Tester<scalar>::Tester() {
    // do nothing
}

template <typename scalar>
std::string
Tester<scalar>::testAlgorithmOnMatrix(const Eigen::MatrixX<scalar> &A,
                                      SubsetSelector<scalar> *algorithm,
                                      uint k) {
    std::string results;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto subset = algorithm->selectSubset(A, k);
    auto t2 = std::chrono::high_resolution_clock::now();

    scalar pinv_frobenius_norm_1 =
        pinv_norm<scalar, Norm::Frobenius>(A(Eigen::all, subset));
    scalar pinv_frobenius_norm_0 = pinv_norm<scalar, Norm::Frobenius>(A);

    scalar pinv_l2_norm_1 = pinv_norm<scalar, Norm::L2>(A(Eigen::all, subset));
    scalar pinv_l2_norm_0 = pinv_norm<scalar, Norm::L2>(A);

    scalar frobenius_volume_reduction =
        pinv_frobenius_norm_0 / pinv_frobenius_norm_1;
    scalar l2_volume_reduction = pinv_l2_norm_0 / pinv_l2_norm_1;

    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    results =
        algorithm->algorithmName +
        ":\n"
        "    frobenius volume reduction = " +
        std::to_string(frobenius_volume_reduction) + "\n" +
        "    l_2 volume reduction = " + std::to_string(l2_volume_reduction) +
        "\n" + "    time = " + std::to_string(time) + " ms \n";

    return results;
}

template <typename scalar>
std::string
Tester<scalar>::testAlgorithmOnMatrix(MatrixGenerator<scalar> *mat_gen,
                                      SubsetSelector<scalar> *algorithm,
                                      uint k) {
    auto A = mat_gen->generateMatrix();

    return testAlgorithmOnMatrix(A, algorithm, k);
}

template <typename scalar>
std::string
Tester<scalar>::testAlgorithmOnMatrix(MatrixGenerator<scalar> *mat_gen,
                                      SubsetSelector<scalar> *algorithm, uint k,
                                      uint cycles) {
    std::string results;
    Eigen::ArrayX<double> time(cycles);
    Eigen::ArrayX<scalar> frobeinus_volume_reduction(cycles);
    Eigen::ArrayX<scalar> l2_volume_reduction(cycles);

    for (uint i = 0; i < cycles; ++i) {
        auto A = mat_gen->generateMatrix();

        auto t1 = std::chrono::high_resolution_clock::now();
        auto subset = algorithm->selectSubset(A, k);
        auto t2 = std::chrono::high_resolution_clock::now();

        scalar pinv_frobenius_norm_1 =
            pinv_norm<scalar, Norm::Frobenius>(A(Eigen::all, subset));
        scalar pinv_frobenius_norm_0 = pinv_norm<scalar, Norm::Frobenius>(A);

        scalar pinv_l2_norm_1 =
            pinv_norm<scalar, Norm::L2>(A(Eigen::all, subset));
        scalar pinv_l2_norm_0 = pinv_norm<scalar, Norm::L2>(A);

        frobeinus_volume_reduction(i) =
            pinv_frobenius_norm_0 / pinv_frobenius_norm_1;
        l2_volume_reduction(i) = pinv_l2_norm_0 / pinv_l2_norm_1;

        time(i) = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                      .count();
    }
    results =
        algorithm->algorithmName +
        ":\n"
        "    frobenius volume reduction = " +
        std::to_string(frobeinus_volume_reduction.mean()) + " +- " +
        std::to_string(std::sqrt(
            (frobeinus_volume_reduction - frobeinus_volume_reduction.mean())
                .square()
                .sum() /
            (cycles - 1))) +
        "\n" + "    l_2 volume reduction = " +
        std::to_string(l2_volume_reduction.mean()) + " +- " +
        std::to_string(std::sqrt(
            (l2_volume_reduction - l2_volume_reduction.mean()).square().sum() /
            (cycles - 1))) +
        "\n" + "    time = " + std::to_string(time.mean()) + " +- " +
        std::to_string(
            std::sqrt((time - time.mean()).square().sum() / (cycles - 1))) +
        " ms \n";

    return results;
}

template <typename scalar>
std::string Tester<scalar>::testAlgorithmsOnMatrix(
    const Eigen::MatrixX<scalar> &A,
    std::vector<SubsetSelector<scalar> *> algorithms, uint k) {
    std::string results;
    for (auto algorithm : algorithms) {
        results += testAlgorithmOnMatrix(A, algorithm, k);
    }

    return results;
}

template <typename scalar>
std::string Tester<scalar>::testAlgorithmsOnMatrix(
    MatrixGenerator<scalar> *mat_gen,
    std::vector<SubsetSelector<scalar> *> algorithms, uint k, uint cycles) {
    std::string results;
    for (auto algorithm : algorithms) {
        results += testAlgorithmOnMatrix(mat_gen, algorithm, k, cycles);
    }

    return results;
}

template <typename scalar>
template <Norm norm>
void Tester<scalar>::scatterPoints(
    MatrixGenerator<scalar> *mat_gen,
    std::vector<SubsetSelector<scalar> *> algorithms, uint k_start,
    uint k_finish, uint points_per_k) {
    
    /*
    // initializing output to file
    std::filesystem::path absolutePath = std::filesystem::absolute(__FILE__);
    std::filesystem::path parentPath = absolutePath.parent_path();
    std::filesystem::current_path(parentPath);

    const auto now = std::chrono::system_clock::now();
    std::string path = "../../out/" + std::format("{:%d-%m-%Y_%H:%M:%OS}", now);
    std::filesystem::create_directory(path);

    // getting matrix parameters
    auto [m, n] = mat_gen->getMatrixSize();
    std::string matrix_type = mat_gen->matrix_type;

    std::ofstream output_points;
    output_points.open(path + "/points.csv");
    output_points << "k,value\n";

    std::ofstream output_bound;
    output_bound.open(path + "/bound.csv");
    output_bound << "k,value\n";

    

    // writing bound
    for (uint k = m; k <= n; ++k) {
        output_bound << k << ',' << algorithm->template bound<norm>(m, n, k)
                     << '\n';
    }

    // scattering points
    for (uint k = m; k <= n; ++k) {
#pragma omp parallel for
        for (uint i = 0; i < points_for_k; ++i) {
            auto A = mat_gen->generateMatrix();
            auto subset = algorithm->selectSubset(A, k);
            scalar pinv_norm_1 = pinv_norm<scalar, norm>(A(Eigen::all, subset));
            scalar pinv_norm_0 = pinv_norm<scalar, norm>(A);
#pragma omp critical
            { output_points << k << ',' << pinv_norm_0 / pinv_norm_1 << '\n'; }
        }
    }

    // closing files
    output_points.close();
    output_bound.close();
    */
}

} // namespace SubsetSelection