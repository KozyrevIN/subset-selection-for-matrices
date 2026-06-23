#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <MatSubset/DerandomizedVolumeSelector.h>
#include "OptimizedDerandomizedVolumeSelector.h"

using Scalar = double;

// Returns sorted indices so order doesn't matter for comparison
static std::vector<Eigen::Index>
sorted(std::vector<Eigen::Index> v) {
    std::sort(v.begin(), v.end());
    return v;
}

struct BenchResult {
    std::vector<Eigen::Index> indices;
    double elapsed_ms;
};

template <typename Selector>
BenchResult run(Selector &sel, const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                int reps) {
    std::vector<Eigen::Index> result;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) {
        result = sel.selectSubset(X, k);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    return {result, ms};
}

// Frobenius norm of the Moore-Penrose pseudoinverse of X(:, indices).
// Equals sqrt(sum 1/sigma_i^2) over nonzero singular values.
static Scalar pinvFrobNorm(const Eigen::MatrixX<Scalar> &X,
                           const std::vector<Eigen::Index> &indices) {
    Eigen::MatrixX<Scalar> S(X.rows(), indices.size());
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(indices.size()); ++j) {
        S.col(j) = X.col(indices[static_cast<size_t>(j)]);
    }
    Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(S);
    const auto &sv = svd.singularValues();
    Scalar tol = std::numeric_limits<Scalar>::epsilon() * sv(0) *
                 std::max(S.rows(), S.cols());
    Scalar acc = 0;
    for (Eigen::Index i = 0; i < sv.size(); ++i) {
        if (sv(i) > tol) acc += Scalar(1) / (sv(i) * sv(i));
    }
    return std::sqrt(acc);
}

static Eigen::MatrixX<Scalar> randomMatrix(Eigen::Index m, Eigen::Index n,
                                           unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<Scalar> dist(0.0, 1.0);
    Eigen::MatrixX<Scalar> X(m, n);
    for (Eigen::Index j = 0; j < n; ++j)
        for (Eigen::Index i = 0; i < m; ++i)
            X(i, j) = dist(rng);
    return X;
}

// Load a CSV file of numeric values into a matrix. Returns rows x cols matrix.
static Eigen::MatrixX<Scalar> loadCSV(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + path);
    }

    std::vector<std::vector<Scalar>> data;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<Scalar> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            if (!value.empty()) {
                row.push_back(static_cast<Scalar>(std::stod(value)));
            }
        }
        if (!row.empty()) data.push_back(row);
    }
    if (data.empty()) {
        throw std::runtime_error("No data found in CSV file: " + path);
    }

    const Eigen::Index rows = static_cast<Eigen::Index>(data.size());
    const Eigen::Index cols = static_cast<Eigen::Index>(data[0].size());
    Eigen::MatrixX<Scalar> M(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            M(i, j) = data[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    return M;
}

static void benchmarkOne(const std::string &label,
                         const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                         int reps, bool &all_match) {
    MatSubset::DerandomizedVolumeSelector<Scalar> orig;
    MatSubset::OptimizedDerandomizedVolumeSelector<Scalar> opt;

    auto r_orig = run(orig, X, k, reps);
    auto r_opt  = run(opt,  X, k, reps);

    bool match = (sorted(r_orig.indices) == sorted(r_opt.indices));
    if (!match) all_match = false;

    double speedup  = r_orig.elapsed_ms / r_opt.elapsed_ms;
    Scalar fro_orig = pinvFrobNorm(X, r_orig.indices);
    Scalar fro_opt  = pinvFrobNorm(X, r_opt.indices);

    std::cout << std::setw(11) << label                << "  "
              << std::setw(3)  << X.rows()             << " "
              << std::setw(5)  << X.cols()             << " "
              << std::setw(4)  << k                    << "  "
              << std::setw(9)  << r_orig.elapsed_ms    << " "
              << std::setw(9)  << r_opt.elapsed_ms     << "  "
              << std::setw(7)  << speedup              << "  "
              << std::setw(15) << fro_orig             << "  "
              << std::setw(14) << fro_opt              << "   "
              << (match ? "OK" : "MISMATCH") << "\n";
}

struct TestCase {
    Eigen::Index m, n, k;
    unsigned seed;
    int reps;
};

int main() {
    Eigen::setNbThreads(1);

    const std::vector<TestCase> cases = {
        // Edge: k = m (square submatrix)
        {4,  20,  4,  40,  200},
        // Edge: k = m + 1
        {4,  20,  5,  41,  200},
        // Regular cases
        {4,  20,  6,  42,  200},
        {8,  40,  12, 43,  100},
        {16, 100, 24, 44,  30},
        {32, 200, 48, 45,  10},
        {64, 500, 96, 46,  3},
        // Edge: k = n - 1 (one column left out)
        {4,  20,  19, 47,  200},
    };

    bool all_match = true;

    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << "\n";
    std::cout << "     label      m     n    k   orig(ms)   opt(ms)  speedup   ||S+||F (orig)   ||S+||F (opt)   match\n";
    std::cout << "------------------------------------------------------------------------------------------------------\n";

    for (auto &tc : cases) {
        auto X = randomMatrix(tc.m, tc.n, tc.seed);
        benchmarkOne("synthetic", X, tc.k, tc.reps, all_match);
    }

    // Abalone dataset benchmark: 4177 samples x 10 features -> transpose to
    // 10 x 4177 (m=10, n=4177).
    const std::string abalone_path =
        "../subset_selection_for_matrices_in_spectral_norm/data/abalone.csv";
    try {
        Eigen::MatrixX<Scalar> A = loadCSV(abalone_path);
        if (A.rows() > A.cols()) A.transposeInPlace();

        const std::vector<Eigen::Index> abalone_ks = {
            A.rows(),         // k = m
            A.rows() + 1,     // k = m + 1
            2 * A.rows(),     // k = 2m
            5 * A.rows(),     // k = 5m
            A.cols() / 10,    // k ~ n/10
        };

        std::cout << "------------------------------------------------------------------------------------------------------\n";
        for (Eigen::Index k : abalone_ks) {
            if (k < A.rows() || k > A.cols()) continue;
            benchmarkOne("abalone", A, k, 1, all_match);
        }
    } catch (const std::exception &e) {
        std::cerr << "Skipping abalone benchmark: " << e.what() << "\n";
    }

    std::cout << "------------------------------------------------------------------------------------------------------\n";
    std::cout << "\nResult: " << (all_match ? "ALL MATCH" : "SOME RESULTS DIFFER") << "\n\n";

    return all_match ? 0 : 1;
}
