#ifndef MAT_SUBSET_EXPERIMENTS_ACOUSTIC_RHS_H
#define MAT_SUBSET_EXPERIMENTS_ACOUSTIC_RHS_H

#include <algorithm>  // For std::min
#include <cassert>    // For assert
#include <cmath>      // For std::exp
#include <cstddef>    // For std::size_t
#include <functional> // For std::function
#include <utility>    // For std::move
#include <vector>     // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include <TTCrossSolver/Solver.h>          // For RhsBase
#include <TTCrossSolver/TensorFibers.h>    // For TensorFibers
#include <TTCrossSolver/TensorTrain.h>     // For TensorTrain
#include <TTCrossSolver/TensorTrainCore.h> // For TensorTrainCore

namespace MatSubset::Experiments {

/*!
 * @brief Builds a TT-operator core from a block matrix of \f$ n \times n \f$
 * mode operators.
 * @param blocks `blocks[a][b]` is the operator sitting at bond indices
 * \f$ (a, b) \f$; all blocks must be square with the same size \f$ n \f$.
 * @return The core of shape \f$ r_0 \times n^2 \times r_1 \f$ with the
 * operator mode folded row-major as \f$ \text{out} \cdot n + \text{in} \f$ —
 * the `TensorTrain::zip` convention.
 */
template <typename Scalar>
TensorTrainCore<Scalar> makeOperatorCore(
    const std::vector<std::vector<Eigen::MatrixX<Scalar>>> &blocks) {
    assert(!blocks.empty() && !blocks.front().empty() &&
           "makeOperatorCore: the block matrix must be non-empty.");
    const auto r0 = static_cast<Eigen::Index>(blocks.size());
    const auto r1 = static_cast<Eigen::Index>(blocks.front().size());
    const Eigen::Index n = blocks.front().front().rows();

    Eigen::MatrixX<Scalar> unfolding =
        Eigen::MatrixX<Scalar>::Zero(r0 * n * n, r1);
    for (std::size_t a = 0; a < blocks.size(); ++a) {
        assert(static_cast<Eigen::Index>(blocks[a].size()) == r1 &&
               "makeOperatorCore: ragged block matrix.");
        for (std::size_t b = 0; b < blocks[a].size(); ++b) {
            const Eigen::MatrixX<Scalar> &block = blocks[a][b];
            assert(block.rows() == n && block.cols() == n &&
                   "makeOperatorCore: all blocks must be n x n.");
            for (Eigen::Index out = 0; out < n; ++out) {
                for (Eigen::Index in = 0; in < n; ++in) {
                    unfolding(static_cast<Eigen::Index>(a) +
                                  r0 * (out * n + in),
                              static_cast<Eigen::Index>(b)) = block(out, in);
                }
            }
        }
    }
    return TensorTrainCore<Scalar>(unfolding, n * n);
}

/*!
 * @brief The d-dimensional second-order finite-difference Laplacian
 * \f$ \Delta = \sum_k I \otimes \dots \otimes L_k \otimes \dots \otimes I \f$
 * as a rank-2 TT operator.
 * @param sizes The grid points per dimension \f$ (n_1, \dots, n_d) \f$.
 * @param spacings The grid spacings \f$ (h_1, \dots, h_d) \f$.
 * @return A `TensorTrain` read as a TT operator by `zip` (mode \f$ k \f$
 * folded as \f$ \text{out} \cdot n_k + \text{in} \f$); interior bond ranks
 * are 2 (1 for d = 1).
 *
 * \f$ L_k = \mathrm{tridiag}(1, -2, 1) / h_k^2 \f$ with the stencil truncated
 * at the ends — homogeneous Dirichlet boundaries. The classic construction:
 * the first core carries \f$ [L_1 \; I] \f$, middle cores
 * \f$ \bigl[\begin{smallmatrix} I & 0 \\ L_k & I \end{smallmatrix}\bigr] \f$,
 * the last core \f$ [I \; L_d]^\top \f$, so the bond product telescopes into
 * the sum of one-site terms.
 */
template <typename Scalar>
TensorTrain<Scalar>
makeLaplacianOperator(const std::vector<Eigen::Index> &sizes,
                      const std::vector<Scalar> &spacings) {
    const std::size_t d = sizes.size();
    assert(d >= 1 && "makeLaplacianOperator: at least one dimension.");
    assert(spacings.size() == d &&
           "makeLaplacianOperator: one spacing per dimension.");

    // The 1D Dirichlet stencil tridiag(1, -2, 1) / h^2.
    const auto stencil = [](Eigen::Index n, Scalar h) {
        Eigen::MatrixX<Scalar> L = Eigen::MatrixX<Scalar>::Zero(n, n);
        const Scalar w = Scalar(1) / (h * h);
        for (Eigen::Index i = 0; i < n; ++i) {
            L(i, i) = Scalar(-2) * w;
            if (i > 0) {
                L(i, i - 1) = w;
            }
            if (i + 1 < n) {
                L(i, i + 1) = w;
            }
        }
        return L;
    };

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(d);
    if (d == 1) {
        cores.push_back(
            makeOperatorCore<Scalar>({{stencil(sizes[0], spacings[0])}}));
    } else {
        for (std::size_t k = 0; k < d; ++k) {
            const Eigen::Index n = sizes[k];
            const Eigen::MatrixX<Scalar> L = stencil(n, spacings[k]);
            const Eigen::MatrixX<Scalar> I =
                Eigen::MatrixX<Scalar>::Identity(n, n);
            const Eigen::MatrixX<Scalar> Z = Eigen::MatrixX<Scalar>::Zero(n, n);
            if (k == 0) {
                cores.push_back(makeOperatorCore<Scalar>({{L, I}}));
            } else if (k + 1 < d) {
                cores.push_back(makeOperatorCore<Scalar>({{I, Z}, {L, I}}));
            } else {
                cores.push_back(makeOperatorCore<Scalar>({{I}, {L}}));
            }
        }
    }
    return TensorTrain<Scalar>(std::move(cores));
}

/*!
 * @brief Right-hand side of the acoustic wave equation
 * \f$ \frac{1}{c(x)^2} \partial_t^2 p = \Delta p + s(x) f(t) \f$, i.e.
 * \f$ F(p, t) = c^2 \odot (\Delta p + s(x) f(t)) \f$, in the fiber format —
 * the TT-leapfrog rhs of Liu & Sacchi (GeoConvention 2025, eq. 3) for the
 * second-order leapfrog scheme.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * All three ingredients are evaluated entry-wise on the step's skeleton
 * (collocation in the sense of Dektor, arXiv:2402.18721), so no low-rank
 * arithmetic on the product terms is ever performed:
 * - \f$ \Delta p \f$: the rank-2 Laplacian TT operator applied by `zip`, the
 *   result sampled with `atFibers` — the only term with rank structure;
 * - \f$ s(x) f(t) \f$: the separable source, its spatial factor sampled on
 *   the skeleton and scaled by the time envelope;
 * - \f$ c^2 \odot \f$: the speed train sampled on the skeleton, squared and
 *   applied via `hadamardProduct` — a pointwise coefficient, however large
 *   the rank of \f$ c^2 \odot (\Delta p + s f) \f$ would be as a train.
 */
template <typename Scalar> class AcousticRhs : public RhsBase<Scalar> {
  public:
    /*!
     * @brief Constructs the rhs from the medium, the source and the grid.
     * @param speed The sound speed \f$ c(x) \f$ compressed to a TT; mode
     * sizes must equal `sizes`.
     * @param source_spatial The spatial source factor \f$ s(x) \f$ as a TT;
     * mode sizes must equal `sizes`.
     * @param source_time The time envelope \f$ f(t) \f$.
     * @param sizes The grid points per dimension.
     * @param spacings The grid spacings per dimension (for the Laplacian
     * stencil).
     */
    AcousticRhs(TensorTrain<Scalar> speed, TensorTrain<Scalar> source_spatial,
                std::function<Scalar(Scalar)> source_time,
                std::vector<Eigen::Index> sizes, std::vector<Scalar> spacings)
        : speed(std::move(speed)), source_spatial(std::move(source_spatial)),
          source_time(std::move(source_time)), sizes(std::move(sizes)),
          laplacian(makeLaplacianOperator<Scalar>(this->sizes, spacings)) {
        assert(this->source_time && "AcousticRhs: null time envelope.");
        assert(this->speed.modeSizes() == this->sizes &&
               "AcousticRhs: speed mode sizes must match the grid.");
        assert(this->source_spatial.modeSizes() == this->sizes &&
               "AcousticRhs: source mode sizes must match the grid.");
    }

    [[nodiscard]] TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &state,
             const TensorFibers<Scalar> &state_fibers,
             Scalar t) const override {
        assert(state.modeSizes() == sizes &&
               "AcousticRhs: state mode sizes must match the grid.");
        const auto &skeleton = state_fibers.skeleton();

        // Δp + s(x) f(t) on the skeleton: the operator through zip, the
        // separable source through its sampled spatial factor.
        TensorFibers<Scalar> forced =
            laplacian.zip(state, sizes, sizes).atFibers(skeleton) +
            source_time(t) * source_spatial.atFibers(skeleton);

        // The pointwise c^2 through the Hadamard product of fibers.
        TensorFibers<Scalar> c = speed.atFibers(skeleton);
        return hadamardProduct(hadamardProduct(c, c), forced);
    }

  private:
    TensorTrain<Scalar> speed;
    TensorTrain<Scalar> source_spatial;
    std::function<Scalar(Scalar)> source_time;
    std::vector<Eigen::Index> sizes;
    TensorTrain<Scalar> laplacian;
};

/*!
 * @brief A pointwise masking boundary condition: the new state is multiplied
 * entry-wise by a fixed mask tensor after every step,
 * \f$ p^{n+1} \leftarrow D \odot p^{n+1} \f$ (Liu & Sacchi, Algorithm 1,
 * line 5).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * In the fiber format the mask costs one `atFibers` and a Hadamard product
 * per step, whatever its rank; for the classic separable (rank-1) absorbing
 * sponge see `makeCerjanMask`.
 */
template <typename Scalar>
class MaskBoundaryCondition : public BoundaryConditionBase<Scalar> {
  public:
    /*! @brief Stores the mask \f$ D \f$; its mode sizes must match the
     * states it will be applied to. */
    explicit MaskBoundaryCondition(TensorTrain<Scalar> mask)
        : mask(std::move(mask)) {}

    [[nodiscard]] TensorFibers<Scalar>
    apply(const TensorFibers<Scalar> &state_fibers, Scalar) const override {
        return hadamardProduct(mask.atFibers(state_fibers.skeleton()),
                               state_fibers);
    }

  private:
    TensorTrain<Scalar> mask;
};

/*!
 * @brief The Cerjan absorbing sponge as a rank-1 TT mask:
 * \f$ D = d_1 \otimes \dots \otimes d_d \f$ with the per-axis taper
 * \f$ d(i) = \exp\bigl(-(\alpha (w - \mathrm{dist}_i))^2\bigr) \f$ for nodes
 * within \f$ w \f$ cells of a wall (\f$ \mathrm{dist}_i = \min(i, n-1-i) \f$)
 * and \f$ d(i) = 1 \f$ in the interior.
 * @param sizes The grid points per dimension.
 * @param width The absorbing strip width \f$ w \f$ in cells (classically
 * 20-40); 0 disables damping (all-ones mask).
 * @param strength The taper rate \f$ \alpha \f$ (Cerjan's classic value is
 * 0.015 for a 20-cell strip).
 * @return The rank-1 mask train, ready for `MaskBoundaryCondition`.
 *
 * Separability is what makes the sponge free in TT arithmetic: a Hadamard
 * product with a rank-1 tensor only rescales core slices and cannot increase
 * ranks. Corners are damped by the product of both axes' tapers, as they
 * should be. The taper must stay gentle (small \f$ \alpha \f$, wide strip) or
 * its inner edge itself reflects.
 */
template <typename Scalar>
TensorTrain<Scalar> makeCerjanMask(const std::vector<Eigen::Index> &sizes,
                                   Eigen::Index width,
                                   Scalar strength = Scalar(0.015)) {
    assert(!sizes.empty() && "makeCerjanMask: at least one dimension.");
    assert(width >= 0 && "makeCerjanMask: width must be non-negative.");

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(sizes.size());
    for (const Eigen::Index n : sizes) {
        Eigen::MatrixX<Scalar> taper(n, 1);
        for (Eigen::Index i = 0; i < n; ++i) {
            const Eigen::Index dist = std::min(i, n - 1 - i);
            if (dist < width) {
                const Scalar x = strength * static_cast<Scalar>(width - dist);
                taper(i, 0) = std::exp(-x * x);
            } else {
                taper(i, 0) = Scalar(1);
            }
        }
        cores.emplace_back(taper, n);
    }
    return TensorTrain<Scalar>(std::move(cores));
}

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_ACOUSTIC_RHS_H
