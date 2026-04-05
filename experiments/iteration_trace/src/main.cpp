#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <nlohmann/json.hpp>

#include <MatSubset/MatSubset.h>
#include <MatrixGenerators/GraphIncidenceMatrixGenerator.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Frobenius norm of the Moore-Penrose pseudoinverse.
// Returns infinity for rank-deficient matrices.
template <typename Scalar>
Scalar pinv_frobenius_norm(const Eigen::MatrixX<Scalar> &A) {
    Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(A);
    const auto &sv = svd.singularValues();
    if (sv.size() == 0)
        return static_cast<Scalar>(0);
    const Scalar tol = sv(0) * std::numeric_limits<Scalar>::epsilon();
    if (sv(sv.size() - 1) <= tol)
        return std::numeric_limits<Scalar>::infinity();
    return std::sqrt(sv.array().inverse().square().sum());
}

// ---------------------------------------------------------------------------
// Dominant swap loop that records metrics at every swap.
// Starts from the subset given in `indices` (size k).
// Returns a vector of rows: {swap_index, frob_ratio_sq, vol_sq_relative}.
//
// vol_sq_relative[t] = vol²(X_S_t) / vol²(X_S_0).
// Each dominant swap multiplies vol² by max_val (the B matrix maximum),
// so this is tracked exactly from the algorithm internals — no SVD needed.
// ---------------------------------------------------------------------------
template <typename Scalar> struct TraceRow {
    int swap_index;
    Scalar frob_ratio_sq;   // ||X†||²_F / ||X_S†||²_F
    Scalar vol_sq_relative; // vol²(X_S_t) / vol²(X_S_0)
};

template <typename Scalar>
std::vector<TraceRow<Scalar>>
dominant_trace(const Eigen::MatrixX<Scalar> &X,
               std::vector<Eigen::Index> indices, // starting subset (size k)
               Eigen::Index k) {
    const Eigen::Index m = X.rows();
    const Eigen::Index n = X.cols();

    // Pre-compute ||X†||_F (fixed throughout)
    const Scalar X_frob_pinv = pinv_frobenius_norm(X);

    // Build permuted working matrix R: selected cols first, rest after
    Eigen::MatrixX<Scalar> R(m, n);
    std::vector<bool> selected(n, false);
    for (Eigen::Index i = 0; i < k; ++i)
        selected[indices[i]] = true;
    for (Eigen::Index j = 0; j < n; ++j)
        if (!selected[j])
            indices.push_back(j);
    for (Eigen::Index j = 0; j < n; ++j)
        R.col(j) = X.col(indices[j]);

    // Compute frob ratio from current R.leftCols(k)
    auto frob_ratio_sq = [&]() -> Scalar {
        Scalar fs = pinv_frobenius_norm(Eigen::MatrixX<Scalar>(R.leftCols(k)));
        return (X_frob_pinv / fs) * (X_frob_pinv / fs);
    };

    std::vector<TraceRow<Scalar>> trace;
    trace.push_back({0, frob_ratio_sq(), static_cast<Scalar>(0)});

    // Dominant loop
    Eigen::MatrixX<Scalar> R_S_dag =
        R.leftCols(k).completeOrthogonalDecomposition().pseudoInverse();
    Eigen::MatrixX<Scalar> C = R_S_dag * R;
    Eigen::ArrayX<Scalar> l = C.colwise().squaredNorm();

    auto C_sel = C.leftCols(k);
    auto C_rem = C.rightCols(n - k);
    auto l_sel = l.head(k);
    auto l_rem = l.tail(n - k);

    Eigen::Index i_max, j_max;
    // Compute B - 1 directly to avoid cancellation when B ≈ 1.
    // B[i,j] = (1 - l_sel[i]) * (1 + l_rem[j]) + C_rem[i,j]²
    // B[i,j] - 1 = l_rem[j] - l_sel[i] - l_sel[i]*l_rem[j] + C_rem[i,j]²
    auto compute_B_m1 = [&]() {
        return (Eigen::VectorX<Scalar>::Ones(k) * l_rem.matrix().transpose() -
                l_sel.matrix() *
                    Eigen::VectorX<Scalar>::Ones(n - k).transpose() -
                l_sel.matrix() * l_rem.matrix().transpose() +
                C_rem.array().abs2().matrix())
            .eval();
    };

    Eigen::MatrixX<Scalar> B_m1 = compute_B_m1();
    Scalar max_val_m1 = B_m1.maxCoeff(&i_max, &j_max);

    const Scalar eps = 100 * std::numeric_limits<Scalar>::epsilon();
    int swap_idx = 0;
    Scalar vol_sq_rel = static_cast<Scalar>(0); // cumulative vol
    Eigen::MatrixX<Scalar> last_row(1, n);

    while (max_val_m1 > eps) {
        // Each swap multiplies vol² by (1 + max_val_m1).
        // log1p is numerically stable when max_val_m1 is tiny.
        vol_sq_rel += max_val_m1;
        Scalar max_val = static_cast<Scalar>(1) + max_val_m1;

        // Add column j_max
        last_row = C_rem.col(j_max).transpose() * C / (1 + l_rem(j_max));
        C -= C_rem.col(j_max) * last_row;
        l -= last_row.transpose().array().abs2() * (1 + l_rem(j_max));

        // Swap into position i_max
        std::swap(indices[static_cast<size_t>(i_max)],
                  indices[static_cast<size_t>(k + j_max)]);
        std::swap(l_sel(i_max), l_rem(j_max));
        C_sel.col(i_max).swap(C_rem.col(j_max));
        std::swap(last_row(i_max), last_row(k + j_max));
        C.row(i_max).swap(last_row);

        // Remove old column i_max
        l += last_row.transpose().array().abs2() / (1 - l_rem(j_max));
        C += C_rem.col(j_max) * last_row * (1 + l_rem(j_max));

        // Update R for frob norm measurement
        R.col(i_max) = X.col(indices[i_max]);
        R.col(k + j_max) = X.col(indices[k + j_max]);

        ++swap_idx;
        trace.push_back({swap_idx, frob_ratio_sq(), vol_sq_rel});

        // Next candidate
        B_m1 = compute_B_m1();
        max_val_m1 = B_m1.maxCoeff(&i_max, &j_max);
    }

    return trace;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    Eigen::setNbThreads(1);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>\n";
        return 1;
    }

    std::filesystem::path config_path(argv[1]);
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Cannot open config: " << config_path << "\n";
        return 1;
    }
    nlohmann::json cfg;
    config_file >> cfg;

    std::filesystem::path base = config_path.parent_path();
    std::filesystem::path outdir = base / cfg.value("output_path", "results");
    std::filesystem::create_directories(outdir);

    const Eigen::Index m = cfg.at("matrix").at("rows").get<Eigen::Index>();
    const Eigen::Index n = cfg.at("matrix").at("cols").get<Eigen::Index>();
    const uint32_t mat_seed = cfg.at("matrix").value("seed", 42u);
    const Eigen::Index k = cfg.at("k").get<Eigen::Index>();
    const int trials = cfg.value("trials", 20);

    using Scalar = double;

    MatSubset::VolumeAddRemoveSelector<Scalar> var_selector(
        static_cast<Scalar>(1.01), MatSubset::Initialization::Greedy);

    std::filesystem::path out_file = outdir / "trace.csv";
    std::ofstream out(out_file);
    out << "trial,swap_index,frob_ratio_sq,vol_sq_relative\n";

    for (int trial = 0; trial < trials; ++trial) {
        // Each trial uses a freshly generated graph incidence matrix with a
        // per-trial seed so the matrix varies across trials.
        MatSubset::Experiments::GraphIncidenceMatrixGenerator<Scalar> gen(
            m, n, mat_seed + static_cast<uint32_t>(trial));
        Eigen::MatrixX<Scalar> X = gen.generateMatrix();

        // Phase 1: volume-add-remove gives us the starting subset
        std::vector<Eigen::Index> var_indices = var_selector.selectSubset(X, k);

        // Phase 2: run dominant from that starting point, recording each swap
        auto trace = dominant_trace(X, var_indices, k);

        for (const auto &row : trace) {
            out << trial << "," << row.swap_index << "," << row.frob_ratio_sq
                << "," << row.vol_sq_relative << "\n";
        }

        std::cout << "Trial " << trial + 1 << "/" << trials << " — "
                  << trace.size() - 1 << " dominant swaps\n";
    }

    out.close();
    std::cout << "Trace written to " << out_file << "\n";
    return 0;
}
