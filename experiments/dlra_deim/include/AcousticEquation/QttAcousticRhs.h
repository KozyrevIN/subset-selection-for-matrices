#ifndef MAT_SUBSET_EXPERIMENTS_QTT_ACOUSTIC_RHS_H
#define MAT_SUBSET_EXPERIMENTS_QTT_ACOUSTIC_RHS_H

#include <cassert> // For assert
#include <cstddef> // For std::size_t
#include <utility> // For std::move
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index, Eigen::JacobiSVD

#include <AcousticEquation/AcousticRhs.h> // For AcousticRhs, makeOperatorCore
#include <TTCrossSolver/TensorTrain.h>     // For TensorTrain, operator+
#include <TTCrossSolver/TensorTrainCore.h> // For TensorTrainCore

namespace MatSubset::Experiments {

// ---------------------------------------------------------------------------
// QTT layout convention (grouped by axis, big-endian within an axis)
//
// A physical d-D grid of sizes (2^{L_0}, ..., 2^{L_{d-1}}) is represented with
// sum_k L_k binary cores, laid out most-significant bit first per axis. Two
// core orderings are supported (QttLayout):
//
//   Grouped     [ x_{L-1}..x_0 ] [ y_{L-1}..y_0 ] [ z_{L-1}..z_0 ]
//   Interleaved [ x_{L-1} y_{L-1} z_{L-1} ] .. [ x_0 y_0 z_0 ]
//
// so a physical index i_k along axis k is i_k = sum_j b_{k,j} 2^j (MSB first).
// Grouped keeps each axis contiguous ("untangled"); Interleaved groups the bits
// by scale, which tends to lower the ranks for isotropic 3D fields (a radial
// wavefront couples x/y/z at the same scale, which sit adjacent there). Both
// require every axis to share the same number of levels L (true for a cubic
// grid); the interleaving is a pure permutation of the core order, applied
// identically to reshaped tensors, the Laplacian operator and the field mapper.
// ---------------------------------------------------------------------------

/*!
 * @brief The QTT core ordering across axes (see the layout note above).
 */
enum class QttLayout {
    Grouped,    //!< all of axis 0's bit-cores, then axis 1's, ...
    Interleaved //!< bits grouped by scale: (x,y,z) at level L-1, then L-2, ...
};

/*!
 * @brief Returns the number of bit levels \f$ L \f$ for a QTT axis of
 * \f$ 2^L \f$ points, asserting the size is a power of two.
 */
inline Eigen::Index qttLevels(Eigen::Index n) {
    assert(n >= 2 && "qttLevels: a QTT axis needs at least 2 points.");
    Eigen::Index levels = 0;
    Eigen::Index m = n;
    while (m > 1) {
        assert((m & 1) == 0 && "qttLevels: QTT axis size must be a power of 2.");
        m >>= 1;
        ++levels;
    }
    return levels;
}

/*!
 * @brief Reshapes a length-\f$ 2^L \f$ vector into a QTT of \f$ L \f$ binary
 * cores, big-endian (most-significant bit first), by successive thin SVDs.
 * @param v The dense samples \f$ v(i) \f$, \f$ i = 0 \dots 2^L - 1 \f$.
 * @return A `TensorTrain` of \f$ L \f$ size-2 cores whose entry at the bit
 * multi-index \f$ (b_{L-1}, \dots, b_0) \f$ equals \f$ v(i) \f$ with
 * \f$ i = \sum_j b_j 2^j \f$.
 *
 * Exact for any vector (this is TT-SVD of the bit reshape); the separable ICs
 * of this experiment come out rank 1, but nothing here assumes that. The MSB is
 * peeled first: the current remainder's top half is bit 0, its bottom half is
 * bit 1 (i.e. the MSB selects which half of the index range).
 */
template <typename Scalar>
TensorTrain<Scalar> vectorToQtt(const Eigen::VectorX<Scalar> &v) {
    const Eigen::Index n = v.size();
    const Eigen::Index levels = qttLevels(n);

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(levels));

    // "remainder" is the not-yet-factored tail as a (left_rank) x (tail) block;
    // row a is the length-(2*tail) segment behind left-bond index a.
    Eigen::MatrixX<Scalar> remainder = v.transpose(); // 1 x n
    Eigen::Index left_rank = 1;
    Eigen::Index tail = n;

    for (Eigen::Index level = 0; level < levels; ++level) {
        tail /= 2;
        // Left unfolding of this core: (2 * left_rank) x tail. The core storage
        // convention puts the mode index (the bit) as the *outer* row block
        // (row = bit * left_rank + a), so bit 0 occupies the top left_rank rows
        // and bit 1 the bottom. Big-endian: bit 0 is the first `tail` entries
        // of each remainder row (top half of the index range), bit 1 the next.
        Eigen::MatrixX<Scalar> block(2 * left_rank, tail);
        for (Eigen::Index a = 0; a < left_rank; ++a) {
            block.row(0 * left_rank + a) = remainder.row(a).segment(0, tail);
            block.row(1 * left_rank + a) = remainder.row(a).segment(tail, tail);
        }

        if (level + 1 == levels) {
            // Last bit: tail == 1, the block is the final core (lr x 2 x 1).
            cores.emplace_back(block, /*n=*/2);
            break;
        }

        // Split the peeled bit (rows) from the tail (cols) by a thin SVD; keep
        // the numerical rank, carry S V^T as the next remainder.
        Eigen::JacobiSVD<Eigen::MatrixX<Scalar>> svd(
            block, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorX<Scalar> sv = svd.singularValues();
        const Scalar tol = sv(0) * Scalar(1e-12) + Scalar(1e-300);
        Eigen::Index rank = 0;
        for (Eigen::Index r = 0; r < sv.size(); ++r) {
            if (sv(r) > tol) {
                ++rank;
            }
        }
        rank = (rank == 0) ? 1 : rank; // an all-zero block is rank 1

        cores.emplace_back(svd.matrixU().leftCols(rank), /*n=*/2);
        remainder = sv.head(rank).asDiagonal() *
                    svd.matrixV().leftCols(rank).transpose();
        left_rank = rank;
    }

    return TensorTrain<Scalar>(std::move(cores));
}

/*!
 * @brief Concatenates \f$ d \f$ per-axis bit-core chains into a single core list
 * in the requested `QttLayout` order.
 * @param axis_chains One chain per axis; for `Interleaved` every chain must have
 * the same length \f$ L \f$ (a cubic grid).
 * @return `Grouped`: axis 0's chain, then axis 1's, ... `Interleaved`: level
 * \f$ L-1 \f$ of every axis (x, y, z, ...), then level \f$ L-2 \f$, ..., down to
 * level 0 — the bit-cores are consumed MSB-first within each chain.
 *
 * The interleaving is a pure permutation, applied identically to reshaped
 * tensors and to the Laplacian's per-axis operator terms, so both stay in the
 * same core order and remain compatible for `zip`.
 */
template <typename Scalar>
std::vector<TensorTrainCore<Scalar>>
concatAxisChains(std::vector<std::vector<TensorTrainCore<Scalar>>> axis_chains,
                 QttLayout layout) {
    std::vector<TensorTrainCore<Scalar>> cores;
    if (layout == QttLayout::Grouped) {
        for (auto &chain : axis_chains) {
            for (auto &core : chain) {
                cores.push_back(std::move(core));
            }
        }
        return cores;
    }
    // Interleaved: scale-major. All chains share length L.
    assert(!axis_chains.empty() && "concatAxisChains: no axes.");
    const std::size_t levels = axis_chains.front().size();
    for (const auto &chain : axis_chains) {
        assert(chain.size() == levels &&
               "concatAxisChains: interleaved layout needs equal levels per "
               "axis (a cubic grid).");
    }
    for (std::size_t level = 0; level < levels; ++level) {
        for (auto &chain : axis_chains) {
            cores.push_back(std::move(chain[level]));
        }
    }
    return cores;
}

/*!
 * @brief Lifts a per-physical-axis TT (one core per axis) to a QTT in the given
 * layout by bit-splitting every axis and concatenating per `concatAxisChains`.
 * @param axis_train A `TensorTrain` with one core per physical axis, each of
 * size \f$ 2^{L_k} \f$; assumed rank 1 (the layered speed and separable source
 * of this experiment).
 * @param layout The QTT core ordering (default `Grouped`).
 * @return The same tensor as a QTT with \f$ \sum_k L_k \f$ size-2 cores.
 *
 * Rank-1 assumption: for a separable tensor \f$ T = f_0 \otimes \dots \otimes
 * f_{d-1} \f$ each axis profile is a slice with the other axes pinned. Pinning
 * at the *corner* (all zeros) is numerically unsafe when \f$ T \f$ is peaked far
 * from the corner (e.g. a Gaussian source): the corner slice underflows and its
 * reciprocal explodes, destroying precision. Instead we pin at a pivot near the
 * tensor's peak: sweep each axis once from the zero corner to locate its argmax
 * \f$ p_k \f$ (a scale-invariant choice — the shared tiny constant does not move
 * the argmax), then re-sweep axis \f$ k \f$ with the others pinned at
 * \f$ p = (p_0, \dots) \f$. The re-sweep gives \f$ g_k(i) = f_k(i)
 * \prod_{m \ne k} f_m(p_m) \f$; normalizing each \f$ g_k \f$ to unit norm and
 * folding the product of norms (times the pivot-value correction) into axis 0
 * as a single scalar reproduces \f$ T \f$ with every factor well-scaled. For a
 * genuinely low-rank (not rank-1) train this needs a full TT-to-QTT reshape;
 * not needed in this experiment.
 */
template <typename Scalar>
TensorTrain<Scalar> trainToQtt(const TensorTrain<Scalar> &axis_train,
                               QttLayout layout = QttLayout::Grouped) {
    const std::vector<Eigen::Index> sizes = axis_train.modeSizes();
    const std::size_t d = sizes.size();
    std::vector<Eigen::Index> idx(d, Eigen::Index(0));

    // Pass 1: locate a per-axis pivot at the argmax of |slice from the corner|.
    // The corner slice is badly scaled but its argmax is still the true peak of
    // f_k (a positive shared constant does not move it).
    std::vector<Eigen::Index> pivot(d, Eigen::Index(0));
    for (std::size_t k = 0; k < d; ++k) {
        Scalar best = Scalar(-1);
        for (Eigen::Index i = 0; i < sizes[k]; ++i) {
            idx[k] = i;
            const Scalar v = std::abs(axis_train(idx));
            if (v > best) {
                best = v;
                pivot[k] = i;
            }
        }
        idx[k] = pivot[k]; // pin this axis at its pivot for later sweeps
    }

    // Pass 2: sweep each axis with the others pinned at the pivot (well scaled)
    // and normalize each factor to unit norm, accumulating the scale.
    Scalar scale = Scalar(1);
    std::vector<Eigen::VectorX<Scalar>> factors(d);
    idx.assign(pivot.begin(), pivot.end());
    for (std::size_t k = 0; k < d; ++k) {
        Eigen::VectorX<Scalar> factor(sizes[k]);
        for (Eigen::Index i = 0; i < sizes[k]; ++i) {
            idx[k] = i;
            factor(i) = axis_train(idx);
        }
        idx[k] = pivot[k];

        const Scalar norm = factor.norm();
        assert(norm > Scalar(0) &&
               "trainToQtt: a rank-1 axis factor is identically zero; use "
               "vectorToQtt on a zero vector for an all-zero tensor.");
        // g_k = f_k * prod_{m!=k} f_m(p_m); the unit factors multiply to
        // T / (scale we accumulate). Each axis contributes its norm; the
        // (d-1)-fold over-count of the pivot values (g_k(p_k) = T(p)) is
        // corrected below.
        factors[k] = factor / norm;
        scale *= norm;
    }

    // Correct the pivot over-count: the unit factors reproduce T scaled by
    // scale / T(p)^{d-1}. Fold that scalar into axis 0's factor.
    const Scalar pivot_value = axis_train(pivot); // T(p) = prod_m f_m(p_m)
    assert(pivot_value != Scalar(0) &&
           "trainToQtt: pivot value is zero; tensor is not rank-1 positive.");
    Scalar overall = scale;
    for (std::size_t m = 0; m + 1 < d; ++m) {
        overall /= pivot_value;
    }
    factors[0] *= overall;

    if (layout == QttLayout::Grouped) {
        // Each normalized factor bit-splits to a rank-1 (all-bond-1) chain, so
        // concatenating the axes is a valid QTT with matching unit bonds.
        std::vector<std::vector<TensorTrainCore<Scalar>>> axis_chains(d);
        for (std::size_t k = 0; k < d; ++k) {
            TensorTrain<Scalar> axis_qtt = vectorToQtt<Scalar>(factors[k]);
            for (std::size_t level = 0; level < axis_qtt.modeSizes().size();
                 ++level) {
                axis_chains[k].push_back(axis_qtt.core(level));
            }
        }
        return TensorTrain<Scalar>(
            concatAxisChains<Scalar>(std::move(axis_chains), layout));
    }

    // Interleaved: the per-axis bit-chains cannot simply be permuted — each
    // factor's bit reshape carries its own internal ranks, so interleaving their
    // cores would break the bonds. Instead assemble the full separable tensor in
    // interleaved index order and TT-SVD it with `vectorToQtt`, which re-threads
    // the bonds correctly. Interleaved index (big-endian across the whole train):
    // bit level L-1 of every axis (x, y, z, ...), then L-2, ..., down to 0.
    // Cost is O(prod_k n_k) for the flat build — paid once for ICs, fine at the
    // grid sizes here.
    for (std::size_t k = 1; k < d; ++k) {
        assert(sizes[k] == sizes[0] &&
               "trainToQtt: interleaved layout needs equal axis sizes (a cubic "
               "grid).");
    }
    const Eigen::Index L = qttLevels(sizes[0]);
    Eigen::Index total = 1;
    for (const Eigen::Index nk : sizes) {
        total *= nk;
    }
    Eigen::VectorX<Scalar> flat(total);
    // Enumerate the interleaved bit multi-index as a big-endian counter: the
    // most-significant train position is (level L-1, axis 0), then (L-1, axis 1),
    // ..., matching QttFieldMapper's interleaved emission.
    for (Eigen::Index flat_i = 0; flat_i < total; ++flat_i) {
        // Decode flat_i's bits (big-endian) into per-axis physical indices.
        std::vector<Eigen::Index> axis_idx(d, 0);
        Eigen::Index bitpos = d * L - 1; // position of the MSB in flat_i
        for (Eigen::Index level = L - 1; level >= 0; --level) {
            for (std::size_t k = 0; k < d; ++k) {
                const Eigen::Index b = (flat_i >> bitpos) & Eigen::Index(1);
                axis_idx[k] |= (b << level);
                --bitpos;
            }
        }
        Scalar value = Scalar(1);
        for (std::size_t k = 0; k < d; ++k) {
            value *= factors[k](axis_idx[k]);
        }
        flat(flat_i) = value;
    }
    return vectorToQtt<Scalar>(flat);
}

/*!
 * @brief The size-2 identity operator chain on a single QTT axis of \f$ L \f$
 * levels: a length-\f$ L \f$ list of rank-1 operator cores each equal to the
 * \f$ 2 \times 2 \f$ identity (`zip` convention, mode folded out*2 + in).
 */
template <typename Scalar>
std::vector<TensorTrainCore<Scalar>> qttIdentityAxis(Eigen::Index levels) {
    const Eigen::MatrixX<Scalar> I2 = Eigen::MatrixX<Scalar>::Identity(2, 2);
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(levels));
    for (Eigen::Index level = 0; level < levels; ++level) {
        cores.push_back(makeOperatorCore<Scalar>({{I2}}));
    }
    return cores;
}

/*!
 * @brief The single-step shift operator \f$ S_{\pm} \f$ on \f$ 2^L \f$
 * Dirichlet points, as an \f$ L \f$-core rank-2 QTT operator.
 * @param levels The number of bit levels \f$ L \f$.
 * @param plus `true` for the \f$ +1 \f$ shift \f$ (S_+ x)_i = x_{i+1} \f$
 * (superdiagonal), `false` for the \f$ -1 \f$ shift \f$ (S_- x)_i = x_{i-1} \f$
 * (subdiagonal); overflow off either end is dropped (homogeneous Dirichlet).
 * @return A `TensorTrain` of \f$ L \f$ size-4 operator cores (`zip` convention).
 *
 * Big-endian carry construction (verified against the dense shift for
 * \f$ L \le 6 \f$): with \f$ J = \bigl[\begin{smallmatrix}0&1\\0&0
 * \end{smallmatrix}\bigr] \f$ (the mode block mapping input bit 1 to output bit
 * 0, i.e. `block(out=0, in=1)`) and \f$ J^\top \f$, the \f$ +1 \f$ shift's MSB
 * core is the row \f$ [\, I \;\; J \,] \f$, interior cores are
 * \f$ \bigl[\begin{smallmatrix} I & J \\ 0 & J^\top \end{smallmatrix}\bigr] \f$
 * and the LSB core is \f$ [\, J \;\; J^\top \,]^\top \f$; the rank-2 bond
 * carries "a +1 is pending from the lower bits", consumed when an input bit is
 * 0 and propagated when it is 1. The \f$ -1 \f$ shift is the transpose, obtained
 * by swapping \f$ J \leftrightarrow J^\top \f$.
 */
template <typename Scalar>
TensorTrain<Scalar> qttShiftAxis(Eigen::Index levels, bool plus) {
    assert(levels >= 1 && "qttShiftAxis: need at least one level.");
    using Mat = Eigen::MatrixX<Scalar>;

    const Mat I = Mat::Identity(2, 2);
    Mat A = Mat::Zero(2, 2);
    A(0, 1) = Scalar(1);   // J: block(out=0, in=1)
    Mat B = A.transpose(); // J^T
    if (!plus) {
        std::swap(A, B); // S_- is the transpose of S_+
    }
    const Mat Z = Mat::Zero(2, 2);

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(levels));
    if (levels == 1) {
        cores.push_back(makeOperatorCore<Scalar>({{A}}));
        return TensorTrain<Scalar>(std::move(cores));
    }
    cores.push_back(makeOperatorCore<Scalar>({{I, A}}));       // MSB
    for (Eigen::Index level = 1; level + 1 < levels; ++level) {
        cores.push_back(makeOperatorCore<Scalar>({{I, A}, {Z, B}}));
    }
    cores.push_back(makeOperatorCore<Scalar>({{A}, {B}}));     // LSB
    return TensorTrain<Scalar>(std::move(cores));
}

/*!
 * @brief The size-2 identity operator train on a single QTT axis of \f$ L \f$
 * levels (a `TensorTrain` of \f$ L \f$ rank-1 identity cores).
 */
template <typename Scalar>
TensorTrain<Scalar> qttIdentityAxisTrain(Eigen::Index levels) {
    return TensorTrain<Scalar>(qttIdentityAxis<Scalar>(levels));
}

/*!
 * @brief The 1-D second-difference operator \f$ \mathrm{tridiag}(1,-2,1)/h^2 \f$
 * on \f$ 2^L \f$ Dirichlet points, as an \f$ L \f$-core QTT operator.
 * @param levels The number of bit levels \f$ L \f$ (axis size \f$ 2^L \f$).
 * @param h The grid spacing.
 * @return The axis' operator cores (`zip` convention, per-core in/out size 2),
 * as a plain list so it can be embedded into a multi-axis operator train.
 *
 * Built as the scaled shift sum \f$ (S_+ + S_- - 2I)/h^2 \f$ from the verified
 * `qttShiftAxis` cores and the identity chain, added with the train
 * `operator+`. This yields the exact tridiagonal (matching the physical
 * `makeLaplacianOperator`) at a pre-truncation bond rank of ~5 that the
 * downstream `zip` + truncation collapses to the true rank 3; it is assembled
 * once, so the small slack is immaterial and buys an obviously-correct
 * construction over a hand-written carry core.
 */
template <typename Scalar>
std::vector<TensorTrainCore<Scalar>>
qttSecondDifferenceAxis(Eigen::Index levels, Scalar h) {
    assert(levels >= 1 && "qttSecondDifferenceAxis: need at least one level.");
    const Scalar w = Scalar(1) / (h * h);
    TensorTrain<Scalar> op =
        w * (qttShiftAxis<Scalar>(levels, /*plus=*/true) +
             qttShiftAxis<Scalar>(levels, /*plus=*/false) +
             Scalar(-2) * qttIdentityAxisTrain<Scalar>(levels));

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(static_cast<std::size_t>(levels));
    for (Eigen::Index level = 0; level < levels; ++level) {
        cores.push_back(op.core(static_cast<std::size_t>(level)));
    }
    return cores;
}

/*!
 * @brief The d-dimensional Laplacian on a QTT grid, in the given layout.
 * @param sizes The physical grid points per axis \f$ (2^{L_1}, \dots) \f$, each
 * a power of two.
 * @param spacings The grid spacings per axis.
 * @param layout The QTT core ordering (default `Grouped`); `Interleaved`
 * requires all axes to share the same level count.
 * @return A QTT operator of \f$ \sum_k L_k \f$ size-4 cores (`zip` convention).
 *
 * Assembled as the sum of \f$ d \f$ one-axis terms,
 * \f$ \Delta = \sum_k I \otimes \dots \otimes \Delta_k \otimes \dots \otimes I
 * \f$: term \f$ k \f$ is the identity chain on every axis except \f$ k \f$,
 * where it is the `qttSecondDifferenceAxis` chain, and the axis chains are
 * concatenated in the same `layout` order (`concatAxisChains`) as the state so
 * the operator and state cores align for `zip`. The terms are added with the
 * train `operator+`, giving an un-truncated inter-axis rank \f$ d \f$ that the
 * solver's `zip` + truncation collapses. In the interleaved layout the sum's
 * pre-truncation ranks are higher (each axis' carry threads past the others),
 * but the operator is built once and truncation restores it; the point of
 * interleaving is the *state's* rank, not the operator's.
 */
template <typename Scalar>
TensorTrain<Scalar>
makeQttLaplacianOperator(const std::vector<Eigen::Index> &sizes,
                         const std::vector<Scalar> &spacings,
                         QttLayout layout = QttLayout::Grouped) {
    const std::size_t d = sizes.size();
    assert(d >= 1 && "makeQttLaplacianOperator: at least one dimension.");
    assert(spacings.size() == d &&
           "makeQttLaplacianOperator: one spacing per dimension.");

    std::vector<Eigen::Index> levels(d);
    for (std::size_t k = 0; k < d; ++k) {
        levels[k] = qttLevels(sizes[k]);
    }

    if (layout == QttLayout::Grouped) {
        // Grouped term k = (I on axes != k) x (Delta_k on axis k): the axes are
        // contiguous, so the chains concatenate directly (only Delta_k's bond is
        // nontrivial and it stays within its own contiguous block).
        const auto axisTerm = [&](std::size_t term) {
            std::vector<std::vector<TensorTrainCore<Scalar>>> axis_chains(d);
            for (std::size_t k = 0; k < d; ++k) {
                axis_chains[k] =
                    (k == term)
                        ? qttSecondDifferenceAxis<Scalar>(levels[k], spacings[k])
                        : qttIdentityAxis<Scalar>(levels[k]);
            }
            return TensorTrain<Scalar>(
                concatAxisChains<Scalar>(std::move(axis_chains), layout));
        };
        TensorTrain<Scalar> laplacian = axisTerm(0);
        for (std::size_t term = 1; term < d; ++term) {
            laplacian = laplacian + axisTerm(term);
        }
        return laplacian;
    }

    // Interleaved: within a term, Delta_k's bond (rank up to ~5) must thread
    // *past* the identity cores of the other axes that sit between k's cores.
    // We emit the d*L cores in scale-major order and pad each identity core with
    // the currently active Delta bond as I_2 (x) I_bond so the inter-core rank
    // stays continuous; k's own cores are used verbatim (their left/right ranks
    // are the bond in/out at that point). All axes share L (asserted).
    for (std::size_t k = 1; k < d; ++k) {
        assert(levels[k] == levels[0] &&
               "makeQttLaplacianOperator: interleaved layout needs equal levels "
               "per axis (a cubic grid).");
    }
    const Eigen::Index L = levels[0];
    // A bond-padded size-2 identity operator core: mode I_2, bond I_w, of shape
    // w x 4 x w (zip convention mode = out*2 + in).
    const auto paddedIdentity = [](Eigen::Index w) {
        std::vector<std::vector<Eigen::MatrixX<Scalar>>> blocks(
            static_cast<std::size_t>(w),
            std::vector<Eigen::MatrixX<Scalar>>(static_cast<std::size_t>(w)));
        const Eigen::MatrixX<Scalar> I2 =
            Eigen::MatrixX<Scalar>::Identity(2, 2);
        const Eigen::MatrixX<Scalar> Z2 = Eigen::MatrixX<Scalar>::Zero(2, 2);
        for (Eigen::Index a = 0; a < w; ++a) {
            for (Eigen::Index b = 0; b < w; ++b) {
                blocks[a][b] = (a == b) ? I2 : Z2; // I_w on the bond, I_2 on mode
            }
        }
        return makeOperatorCore<Scalar>(blocks);
    };

    const auto axisTermInterleaved = [&](std::size_t term) {
        const std::vector<TensorTrainCore<Scalar>> delta =
            qttSecondDifferenceAxis<Scalar>(levels[term], spacings[term]);
        std::vector<TensorTrainCore<Scalar>> cores;
        cores.reserve(static_cast<std::size_t>(d * L));
        Eigen::Index bond = 1; // active inter-core bond width
        for (Eigen::Index level = L - 1; level >= 0; --level) {
            const std::size_t di =
                static_cast<std::size_t>(L - 1 - level); // Delta core index
            for (std::size_t a = 0; a < d; ++a) {
                if (a == term) {
                    cores.push_back(delta[di]); // left=bond in, right=bond out
                    bond = delta[di].rightRank();
                } else {
                    cores.push_back(paddedIdentity(bond)); // thread the bond
                }
            }
        }
        return TensorTrain<Scalar>(std::move(cores));
    };

    TensorTrain<Scalar> laplacian = axisTermInterleaved(0);
    for (std::size_t term = 1; term < d; ++term) {
        laplacian = laplacian + axisTermInterleaved(term);
    }
    return laplacian;
}

/*!
 * @brief Right-hand side of the acoustic wave equation in the grouped QTT
 * layout: identical physics to `AcousticRhs`, but the Laplacian operator is the
 * QTT `makeQttLaplacianOperator` and every ingredient train (speed, source) is
 * expected in QTT (size-2) modes.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * `AcousticRhs` is dimension-agnostic apart from the Laplacian it builds from
 * `sizes` in its constructor. Rather than parameterize that, this subclass
 * simply is `AcousticRhs` over the QTT mode sizes (all 2s) — but the base
 * builds the *physical* Laplacian, which is wrong for QTT. So we do not reuse
 * the base's Laplacian: `QttAcousticRhs` is a thin standalone RHS mirroring
 * `AcousticRhs::evaluate / evaluateTrain / makeEvaluator` with the QTT
 * Laplacian. Kept in one place so the QTT and TT RHS stay in lockstep.
 */
template <typename Scalar> class QttAcousticRhs : public RhsBase<Scalar> {
  public:
    /*!
     * @brief Constructs the QTT RHS.
     * @param speed The sound speed \f$ c(x) \f$ as a QTT (size-2 modes,
     * \f$ \sum_k L_k \f$ of them).
     * @param source_spatial The spatial source factor \f$ s(x) \f$ as a QTT.
     * @param source_time The time envelope \f$ f(t) \f$.
     * @param sizes The *physical* grid points per axis (powers of two); used to
     * build the QTT Laplacian and its per-core in/out sizes (all 2).
     * @param spacings The physical grid spacings per axis.
     * @param layout The QTT core ordering; must match the layout the `speed`,
     * `source_spatial` and state trains were built with (default `Grouped`).
     */
    QttAcousticRhs(TensorTrain<Scalar> speed,
                   TensorTrain<Scalar> source_spatial,
                   std::function<Scalar(Scalar)> source_time,
                   std::vector<Eigen::Index> sizes,
                   std::vector<Scalar> spacings,
                   QttLayout layout = QttLayout::Grouped)
        : speed(std::move(speed)), source_spatial(std::move(source_spatial)),
          source_time(std::move(source_time)),
          laplacian(makeQttLaplacianOperator<Scalar>(sizes, spacings, layout)),
          speed_squared(hadamardProduct(this->speed, this->speed)) {
        // QTT mode sizes: sum_k L_k twos.
        qtt_sizes = this->speed.modeSizes();
        assert(this->source_time && "QttAcousticRhs: null time envelope.");
        assert(this->source_spatial.modeSizes() == qtt_sizes &&
               "QttAcousticRhs: source QTT mode sizes must match the speed.");
        assert(laplacian.modeSizes().size() == qtt_sizes.size() &&
               "QttAcousticRhs: Laplacian order must match the QTT state.");
        for (const Eigen::Index m : qtt_sizes) {
            assert(m == 2 && "QttAcousticRhs: QTT modes must be size 2.");
        }
    }

    [[nodiscard]] TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &state,
             const TensorFibers<Scalar> &state_fibers,
             Scalar t) const override {
        assert(state.modeSizes() == qtt_sizes &&
               "QttAcousticRhs: state QTT mode sizes must match.");
        const auto &skeleton = state_fibers.skeleton();
        TensorFibers<Scalar> forced =
            laplacian.zip(state, qtt_sizes, qtt_sizes).atFibers(skeleton) +
            source_time(t) * source_spatial.atFibers(skeleton);
        TensorFibers<Scalar> c = speed.atFibers(skeleton);
        return hadamardProduct(hadamardProduct(c, c), forced);
    }

    [[nodiscard]] TensorTrain<Scalar>
    evaluateTrain(const TensorTrain<Scalar> &state, Scalar t) const override {
        assert(state.modeSizes() == qtt_sizes &&
               "QttAcousticRhs: state QTT mode sizes must match.");
        return hadamardProduct(speed_squared,
                               laplacian.zip(state, qtt_sizes, qtt_sizes) +
                                   source_time(t) * source_spatial);
    }

    [[nodiscard]] std::unique_ptr<FiberEvaluatorBase<Scalar>>
    makeEvaluator(const TensorTrain<Scalar> &state, Scalar t) const override {
        assert(state.modeSizes() == qtt_sizes &&
               "QttAcousticRhs: state QTT mode sizes must match.");
        return std::make_unique<AcousticRhsFiberEvaluator<Scalar>>(
            laplacian.zip(state, qtt_sizes, qtt_sizes), speed, source_spatial,
            source_time(t));
    }

  private:
    TensorTrain<Scalar> speed;
    TensorTrain<Scalar> source_spatial;
    std::function<Scalar(Scalar)> source_time;
    std::vector<Eigen::Index> qtt_sizes;
    TensorTrain<Scalar> laplacian;
    TensorTrain<Scalar> speed_squared;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_QTT_ACOUSTIC_RHS_H
