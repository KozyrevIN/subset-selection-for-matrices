#include <chrono>
#include <filesystem>
#include <fstream>
#include <json/json.h>
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

    scalar frobenius_norm_reduction =
        pinv_frobenius_norm_0 / pinv_frobenius_norm_1;
    scalar l2_norm_reduction = pinv_l2_norm_0 / pinv_l2_norm_1;

    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    results = algorithm->algorithmName +
              ":\n"
              "    frobenius volume reduction = " +
              std::to_string(std::pow(frobenius_norm_reduction, 2)) + "\n" +
              "    l_2 volume reduction = " +
              std::to_string(std::pow(l2_norm_reduction, 2)) + "\n" +
              "    time = " + std::to_string(time) + " ms \n";

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
    Eigen::ArrayX<scalar> frobeinus_norm_reduction_sq(cycles);
    Eigen::ArrayX<scalar> l2_norm_reduction_sq(cycles);

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

        frobeinus_norm_reduction_sq(i) =
            std::pow(pinv_frobenius_norm_0 / pinv_frobenius_norm_1, 2);
        l2_norm_reduction_sq(i) = std::pow(pinv_l2_norm_0 / pinv_l2_norm_1, 2);

        time(i) = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                      .count();
    }
    results = algorithm->getAlgorithmName() +
              ":\n"
              "    frobenius volume reduction = " +
              std::to_string(frobeinus_norm_reduction_sq.mean()) + " +- " +
              std::to_string(std::sqrt((frobeinus_norm_reduction_sq -
                                        frobeinus_norm_reduction_sq.mean())
                                           .square()
                                           .sum() /
                                       (cycles - 1))) +
              "\n" + "    l_2 volume reduction = " +
              std::to_string(l2_norm_reduction_sq.mean()) + " +- " +
              std::to_string(
                  std::sqrt((l2_norm_reduction_sq - l2_norm_reduction_sq.mean())
                                .square()
                                .sum() /
                            (cycles - 1))) +
              "\n" + "    time = " + std::to_string(time.mean()) + " +- " +
              std::to_string(std::sqrt((time - time.mean()).square().sum() /
                                       (cycles - 1))) +
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
    uint k_finish, uint k_step, uint points_per_k) {

    // initializing output to directory
    std::filesystem::path absolutePath = std::filesystem::absolute(__FILE__);
    std::filesystem::path parentPath = absolutePath.parent_path();
    std::filesystem::current_path(parentPath);

    const auto now = std::chrono::system_clock::now();
    const auto now_time_t = std::chrono::system_clock::to_time_t(now);
    const auto *now_local = std::localtime(&now_time_t);
    std::string path = "../../out/run_";
    path += std::format("{:02d}-{:02d}-{:04d}_{:02d}:{:02d}:{:02d}",
                        now_local->tm_mday, now_local->tm_mon + 1,
                        now_local->tm_year + 1900, now_local->tm_hour,
                        now_local->tm_min, now_local->tm_sec);
    std::filesystem::create_directory(path);

    // writing run info
    Json::Value run_info;

    auto [m, n] = mat_gen->getMatrixSize();
    std::string matrix_type = mat_gen->getMatrixType();
    run_info["matrix parameters"]["m"] = m;
    run_info["matrix parameters"]["n"] = n;
    run_info["matrix parameters"]["type"] = matrix_type;

    run_info["testing parameters"]["k start"] = k_start;
    run_info["testing parameters"]["k finish"] = k_finish;
    run_info["testing parameters"]["k step"] = k_step;
    run_info["testing parameters"]["points per k"] = points_per_k;

    for (auto algorithm : algorithms) {
        run_info["algorithms"].append(algorithm->getAlgorithmName());
    }

    Json::StyledWriter writer;
    std::string json_string = writer.write(run_info);
    std::ofstream run_info_file(path + "/run_info.json");
    run_info_file << json_string << std::endl;
    run_info_file.close();

    // creating algorithm names with underscores, which will be used as
    // filenames
    std::vector<std::string> underscored_names;

    for (auto algorithm : algorithms) {
        std::string name = algorithm->getAlgorithmName();
        std::replace(name.begin(), name.end(), ' ', '_');
        underscored_names.push_back(name);
    }

    // writing bounds to files
    for (uint i = 0; i < algorithms.size(); ++i) {
        std::ofstream bound_file(path + "/" + underscored_names[i] +
                                 "_bound.csv");
        bound_file << "k,value\n";
        for (uint k = k_start; k <= k_finish; k += k_step) {
            bound_file << k << ','
                       << algorithms[i]->template bound<norm>(m, n, k) << '\n';
        }
        bound_file.close();
    }

    // opening files to save testing results
    std::vector<std::ofstream> points_files;
    for (uint i = 0; i < algorithms.size(); ++i) {
        points_files.push_back(
            std::ofstream(path + "/" + underscored_names[i] + "_points.csv"));
        points_files[i] << "k,value\n";
    }

    // testing algorithms and outputting results
#pragma omp parallel for schedule(dynamic, 1)
    for (uint k = k_start; k <= k_finish; k += k_step) {
        for (uint point = 0; point < points_per_k; ++point) {
            auto A = mat_gen->generateMatrix();
            for (uint i = 0; i < algorithms.size(); ++i) {
                auto subset = algorithms[i]->selectSubset(A, k);
                scalar pinv_norm_1 =
                    pinv_norm<scalar, norm>(A(Eigen::all, subset));
                scalar pinv_norm_0 = pinv_norm<scalar, norm>(A);
#pragma omp critical
                {
                    points_files[i] << k << ','
                                    << std::pow(pinv_norm_0 / pinv_norm_1, 2)
                                    << '\n';
                }
            }
        }
    }

    // closing files
    for (uint i = 0; i < algorithms.size(); ++i) {
        points_files[i].close();
    }
}

} // namespace SubsetSelection