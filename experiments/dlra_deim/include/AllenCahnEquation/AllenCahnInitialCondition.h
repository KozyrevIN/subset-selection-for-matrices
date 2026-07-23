#ifndef MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_INITIAL_CONDITION_H
#define MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_INITIAL_CONDITION_H

#include <cassert> // For assert
#include <cmath>   // For std::exp, std::tan, std::sin, std::abs
#include <cstddef> // For std::size_t
#include <utility> // For std::move
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::VectorX, Eigen::Index
#include <Eigen/SVD>  // For Eigen::BDCSVD

#include "TTCrossSolver/SnapshotSaver.h"   // For Grid
#include "TTCrossSolver/TensorTrain.h"     // For TensorTrain
#include "TTCrossSolver/TensorTrainCore.h" // For TensorTrainCore

namespace MatSubset::Experiments {

/*!
 * @brief The building block \f$ g \f$ of the 3D Allen-Cahn initial condition
 * (report eq. for \f$ g \f$):
 * \f[ g(x_1, x_2, x_3) =
 *   \frac{\bigl(e^{-\tan(x_1)^2} + e^{-\tan(x_2)^2} + e^{-\tan(x_3)^2}\bigr)
 *         \sin(x_1 + x_2 + x_3)}
 *        {1 + e^{|\csc(-x_1/2)|} + e^{|\csc(-x_2/2)|} + e^{|\csc(-x_3/2)|}} . \f]
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Not separable — the numerator's \f$ \sin(x_1 + x_2 + x_3) \f$ and the coupled
 * denominator entangle all three axes, which is exactly why the initial state
 * has to be compressed by TT-SVD (`ttFromDenseTensor`) rather than built as a
 * rank-1 train. On the periodic interior grid \f$ x_k = 2\pi k / n \f$ the
 * endpoints \f$ 0, 2\pi \f$ (where \f$ \csc(-x/2) \f$ blows up) are excluded;
 * \f$ \tan \f$ still spikes near \f$ \pi/2, 3\pi/2 \f$ but stays finite on-grid,
 * where its Gaussian \f$ e^{-\tan^2} \f$ simply underflows to 0.
 */
template <typename Scalar>
[[nodiscard]] Scalar allenCahnG(Scalar x1, Scalar x2, Scalar x3) {
    const auto gaussTan = [](Scalar x) {
        const Scalar t = std::tan(x);
        return std::exp(-t * t);
    };
    // csc(y) = 1 / sin(y); here y = -x/2.
    const auto expAbsCsc = [](Scalar x) {
        const Scalar s = std::sin(-x / Scalar(2));
        return std::exp(std::abs(Scalar(1) / s));
    };

    const Scalar numerator =
        (gaussTan(x1) + gaussTan(x2) + gaussTan(x3)) * std::sin(x1 + x2 + x3);
    const Scalar denominator =
        Scalar(1) + expAbsCsc(x1) + expAbsCsc(x2) + expAbsCsc(x3);
    return numerator / denominator;
}

/*!
 * @brief The 3D Allen-Cahn initial condition \f$ f_0 \f$ (report eq. (for
 * \f$ f_0 \f$)):
 * \f[ f_0(x_1, x_2, x_3) = g(x_1, x_2, x_3) - g(2x_1, x_2, x_3)
 *     + g(x_1, 2x_2, x_3) - g(x_1, x_2, 2x_3), \f]
 * with \f$ g \f$ as in `allenCahnG`.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 */
template <typename Scalar>
[[nodiscard]] Scalar allenCahnInitial(Scalar x1, Scalar x2, Scalar x3) {
    return allenCahnG<Scalar>(x1, x2, x3) -
           allenCahnG<Scalar>(Scalar(2) * x1, x2, x3) +
           allenCahnG<Scalar>(x1, Scalar(2) * x2, x3) -
           allenCahnG<Scalar>(x1, x2, Scalar(2) * x3);
}

/*!
 * @brief Samples a scalar field \f$ f_0 \f$ on every node of a 3D grid into a
 * flat, first-mode-fastest field.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 * @param grid The grid (must be 3D).
 * @param func The field \f$ f_0(x_1, x_2, x_3) \f$.
 * @return The flat field of length \f$ n_1 n_2 n_3 \f$ laid out column-major
 * (first-mode-fastest, entry \f$ i + n_1(j + n_2 k) \f$), matching
 * `TensorTrain::toDense()` and `AllenCahnDenseSolver::field()`.
 */
template <typename Scalar, typename Func>
[[nodiscard]] Eigen::VectorX<Scalar>
denseFieldFromFunction(const Grid<Scalar> &grid, const Func &func) {
    assert(grid.dim() == 3 && "denseFieldFromFunction: expects a 3D grid.");
    const Eigen::Index n0 = grid.size(0);
    const Eigen::Index n1 = grid.size(1);
    const Eigen::Index n2 = grid.size(2);
    Eigen::VectorX<Scalar> field(n0 * n1 * n2);
    for (Eigen::Index k = 0; k < n2; ++k) {
        const Scalar x3 = grid.coordinate(2, k);
        for (Eigen::Index j = 0; j < n1; ++j) {
            const Scalar x2 = grid.coordinate(1, j);
            for (Eigen::Index i = 0; i < n0; ++i) {
                const Scalar x1 = grid.coordinate(0, i);
                field(i + n0 * (j + n1 * k)) = func(x1, x2, x3);
            }
        }
    }
    return field;
}

/*!
 * @brief Exact TT-SVD of a dense tensor into a left-orthogonal `TensorTrain`,
 * truncated at the relative Frobenius tolerance `rtol`.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 * @param dense The flat tensor in column-major, first-mode-fastest order (as
 * produced by `denseFieldFromFunction` / `TensorTrain::toDense`), length
 * \f$ \prod_k n_k \f$.
 * @param sizes The mode sizes \f$ (n_1, \dots, n_d) \f$.
 * @param rtol Relative Frobenius tolerance: at each bond the singular values
 * whose trailing energy stays within `rtol` of the slab norm are dropped.
 * @return The compressed train; its `toDense()` matches `dense` to within
 * roughly \f$ \sqrt{d - 1}\,\text{rtol} \f$ relative error.
 *
 * The classic left-to-right sequential SVD. Because the storage is column-major
 * with the left index fastest inside each core's unfolding
 * (`CoreBase::modeSlice`), the reshape from the running carry to the bond-`k`
 * unfolding is a plain `Eigen::Map` — no explicit index juggling. This is only
 * used once, to seed the initial state from the non-separable initial
 * condition; the point of a TT everywhere else is to avoid forming `dense`.
 */
template <typename Scalar>
[[nodiscard]] TensorTrain<Scalar>
ttFromDenseTensor(const Eigen::VectorX<Scalar> &dense,
                  const std::vector<Eigen::Index> &sizes,
                  Scalar rtol = Scalar(1e-10)) {
    const std::size_t d = sizes.size();
    assert(d >= 1 && "ttFromDenseTensor: at least one mode.");

    Eigen::Index total = 1;
    for (const Eigen::Index n : sizes) {
        assert(n >= 1 && "ttFromDenseTensor: each mode needs >= 1 point.");
        total *= n;
    }
    assert(dense.size() == total &&
           "ttFromDenseTensor: dense size must equal prod(sizes).");

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(d);

    // carry: r_{k-1} x (n_k ... n_{d-1}), column-major with the first remaining
    // mode fastest. Starts as 1 x total (the whole tensor as one row).
    Eigen::Index left_rank = 1;
    Eigen::Index remaining = total;
    Eigen::MatrixX<Scalar> carry =
        Eigen::Map<const Eigen::MatrixX<Scalar>>(dense.data(), 1, total);

    for (std::size_t k = 0; k < d; ++k) {
        const Eigen::Index n = sizes[k];
        remaining /= n;

        // Reshape carry (left_rank x (n * remaining)) into the bond-k left
        // unfolding M of shape (left_rank * n) x remaining. Column-major with
        // the left index fastest: row = a + left_rank * i_k, exactly the
        // CoreBase unfolding convention, so this is a pure reinterpretation.
        Eigen::Map<Eigen::MatrixX<Scalar>> M(carry.data(), left_rank * n,
                                             remaining);

        if (k + 1 == d) {
            // Last core: the carry is the core itself (right rank 1).
            cores.emplace_back(Eigen::MatrixX<Scalar>(M), n);
            break;
        }

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(
            M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorX<Scalar> &sv = svd.singularValues();

        // Truncate: keep the leading singular values whose *trailing* energy
        // (the discarded tail) stays within rtol of the slab's Frobenius norm.
        const Scalar total_energy = sv.squaredNorm();
        const Scalar threshold = rtol * rtol * total_energy;
        Eigen::Index rank = sv.size();
        Scalar tail = Scalar(0);
        while (rank > 1) {
            const Scalar next_tail =
                tail + sv(rank - 1) * sv(rank - 1);
            if (next_tail > threshold) {
                break;
            }
            tail = next_tail;
            --rank;
        }

        // Core k: the leading `rank` left singular vectors, shape
        // (left_rank * n) x rank.
        cores.emplace_back(
            Eigen::MatrixX<Scalar>(svd.matrixU().leftCols(rank)), n);

        // New carry: Sigma * V^T restricted to the kept directions, shape
        // rank x remaining.
        carry = sv.head(rank).asDiagonal() *
                svd.matrixV().leftCols(rank).transpose();
        left_rank = rank;
    }

    return TensorTrain<Scalar>(std::move(cores));
}

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_INITIAL_CONDITION_H
