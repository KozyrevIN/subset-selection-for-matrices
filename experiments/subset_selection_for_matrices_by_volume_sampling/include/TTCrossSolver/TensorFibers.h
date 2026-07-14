#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_H

#include <cassert> // For assert
#include <cstddef> // For std::size_t
#include <memory>  // For std::shared_ptr
#include <utility> // For std::move
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include "TTCrossSolver/TensorFibersCore.h" // For TensorFibersCore

namespace MatSubset::Experiments {

/*!
 * @brief Nested left/right cross index sets for a tensor / tensor-train, the
 * "skeleton" on which a tensor is sampled.
 *
 * At bond \f$ k \f$ the left index set is a collection of multi-indices
 * \f$ (i_1, \dots, i_k) \f$ and the right index set a collection of
 * \f$ (i_{k+1}, \dots, i_d) \f$. The sets are *nested*: every left multi-index
 * at bond \f$ k \f$ extends exactly one left multi-index at bond \f$ k-1 \f$
 * by appending a mode value \f$ i_k \f$, and every right multi-index at bond
 * \f$ k \f$ extends exactly one right multi-index at bond \f$ k+1 \f$ by
 * prepending a mode value \f$ i_{k+1} \f$. Rather than store the full tuples,
 * each node records only its added mode value and a `parent` pointer into the
 * level it extends, so a multi-index is recovered by walking parents to the
 * root.
 *
 * Both sides are indexed by bond: `left[k]` and `right[k]` (k = 0 .. d-1) are
 * the bond-k index sets, so `left[k]` parents point into `left[k-1]` and
 * `right[k]` parents point into `right[k+1]`. At the boundaries, `left[0]`
 * nodes carry `parent == -1` (each extends the empty left boundary index),
 * `right[d-1]` is the right root — a single node with `parent == -1` standing
 * for the empty right boundary index (its mode value is unused) — and
 * `left[d-1]` is conventionally empty (a left set at the last bond would
 * enumerate full multi-indices).
 */
class FiberIndices {
  public:
    /*! @brief One nesting level: the multi-indices selected at a single bond.
     */
    class Level {
      public:
        Level() = default;

        /*!
         * @brief Constructs a level from appended mode values and parent links.
         * @param mode_idx `i_k` appended by each node.
         * @param parent For each node, the index into the previous level it
         * extends (-1 at the root level). Must match `mode_idx` in length.
         */
        Level(std::vector<Eigen::Index> mode_idx,
              std::vector<Eigen::Index> parent)
            : mode_idx(std::move(mode_idx)), parent(std::move(parent)) {
            assert(this->mode_idx.size() == this->parent.size() &&
                   "FiberIndices::Level: mode_idx and parent size mismatch.");
        }

        /*! @brief Number of multi-indices (nodes) at this level. */
        [[nodiscard]] std::size_t size() const { return mode_idx.size(); }

        /*! @brief Appended mode value i_k of node `j`. */
        [[nodiscard]] Eigen::Index mode(std::size_t j) const {
            assert(j < mode_idx.size() &&
                   "FiberIndices::Level: node index OOR.");
            return mode_idx[j];
        }

        /*! @brief Parent of node `j` in the previous level (-1 at the root). */
        [[nodiscard]] Eigen::Index parentOf(std::size_t j) const {
            assert(j < parent.size() && "FiberIndices::Level: node index OOR.");
            return parent[j];
        }

        /*!
         * @brief Decodes selected columns of a right unfolding into a right
         * level.
         * @param indices Selected flat column indices, each folded as
         * `mode * rho + parent` (mode slower, matching the absorbed right
         * unfolding).
         * @param rho The number of right fibers (columns) the parent runs over.
         * @return The level whose node `j` prepends mode `indices[j] / rho` to
         * parent `indices[j] % rho`.
         */
        [[nodiscard]] static Level
        fromRightIndices(const std::vector<Eigen::Index> &indices,
                         Eigen::Index rho) {
            std::vector<Eigen::Index> modes(indices.size());
            std::vector<Eigen::Index> parents(indices.size());
            for (std::size_t j = 0; j < indices.size(); ++j) {
                modes[j] = indices[j] / rho;
                parents[j] = indices[j] % rho;
            }
            return Level(std::move(modes), std::move(parents));
        }

        /*!
         * @brief Decodes selected rows of a left unfolding into a left level.
         * @param indices Selected flat row indices, each folded as
         * `parent + l_prev * mode` (mode slower, matching the absorbed left
         * unfolding).
         * @param l_prev The number of left fibers (rows) the parent runs over.
         * @param is_root True at bond 0, whose nodes extend the empty boundary
         * index (parent -1) rather than a previous level.
         * @return The level whose node `j` appends mode `indices[j] / l_prev` to
         * parent `indices[j] % l_prev` (or -1 at the root).
         */
        [[nodiscard]] static Level
        fromLeftIndices(const std::vector<Eigen::Index> &indices,
                        Eigen::Index l_prev, bool is_root) {
            std::vector<Eigen::Index> modes(indices.size());
            std::vector<Eigen::Index> parents(indices.size());
            for (std::size_t j = 0; j < indices.size(); ++j) {
                modes[j] = indices[j] / l_prev;
                parents[j] = is_root ? -1 : indices[j] % l_prev;
            }
            return Level(std::move(modes), std::move(parents));
        }

      private:
        std::vector<Eigen::Index> mode_idx;
        std::vector<Eigen::Index> parent;
    };

    FiberIndices() = default;

    /*!
     * @brief Constructs a skeleton from its left and right level lists.
     * @param left `left[k]`, k = 0 .. d-1: the bond-k left sets (parent -1 at
     * `left[0]`).
     * @param right `right[k]`, k = 0 .. d-1: the bond-k right sets (root at
     * `right[d-1]`).
     */
    FiberIndices(std::vector<Level> left, std::vector<Level> right)
        : left(std::move(left)), right(std::move(right)) {
        assert(this->left.size() == this->right.size() &&
               "FiberIndices: left and right must have the same order.");
    }

    /*! @brief The tensor order d (number of cores). */
    [[nodiscard]] std::size_t order() const { return left.size(); }

    /*! @brief The left level at bond `k`. */
    [[nodiscard]] const Level &leftLevel(std::size_t k) const {
        assert(k < left.size() && "FiberIndices: left level index OOR.");
        return left[k];
    }

    /*! @brief The right level at bond `k` (root at `right[d-1]`). */
    [[nodiscard]] const Level &rightLevel(std::size_t k) const {
        assert(k < right.size() && "FiberIndices: right level index OOR.");
        return right[k];
    }

    /*!
     * @brief Replaces the left level at bond `k`.
     *
     * For adaptive cross sweeps, which re-select the levels one bond at a
     * time. The caller is responsible for nestedness: the new level's parents
     * must point into the current `left[k-1]`, and any existing `left[k+1]`
     * becomes stale until it is re-selected in turn (sweep order takes care
     * of both).
     */
    void setLeftLevel(std::size_t k, Level level) {
        assert(k < left.size() && "FiberIndices: left level index OOR.");
        left[k] = std::move(level);
    }

    /*!
     * @brief Replaces the right level at bond `k`; mirror of `setLeftLevel`
     * (parents of the new level must point into the current `right[k+1]`).
     */
    void setRightLevel(std::size_t k, Level level) {
        assert(k < right.size() && "FiberIndices: right level index OOR.");
        right[k] = std::move(level);
    }

    /*!
     * @brief Number of left fibers feeding into core `k` (its slab's left
     * rank).
     *
     * \f$ |\text{left}[k-1]| \f$, with the boundary convention that core 0 has
     * a single (empty) left multi-index.
     */
    [[nodiscard]] std::size_t leftFiberCount(std::size_t k) const {
        return (k == 0) ? 1 : left[k - 1].size();
    }

    /*!
     * @brief Number of right fibers feeding into core `k` (its slab's right
     * rank).
     *
     * \f$ |\text{right}[k]| \f$, with the boundary convention that the last
     * core has a single (empty) right multi-index.
     */
    [[nodiscard]] std::size_t rightFiberCount(std::size_t k) const {
        return (k + 1 == order()) ? 1 : right[k].size();
    }

  private:
    std::vector<Level> left;  // Bond-k left sets; parents into left[k-1].
    std::vector<Level> right; // Bond-k right sets; parents into right[k+1].
};

/*!
 * @brief The tensor evaluated on the fibers of a `FiberIndices` skeleton: one
 * slab per core.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Slab \f$ k \f$ is the 3D tensor of entries of the sampled tensor obtained by
 * fixing the left multi-index to one of `indices->left[k-1]`, the right
 * multi-index to one of `indices->right[k]`, and letting mode \f$ i_k \f$ run
 * over \f$ 0 \dots n_k - 1 \f$. It is stored as a `TensorFibersCore` whose
 * left unfolding has shape \f$ (\ell_{k-1} \, n_k) \times \rho_k \f$, where
 * \f$ \ell_{k-1} = \text{leftFiberCount}(k) \f$ and
 * \f$ \rho_k = \text{rightFiberCount}(k) \f$. These are exactly the shapes
 * consumed by `TensorTrainCore`.
 *
 * Two `TensorFibers` sharing the same `FiberIndices` combine slab-wise via `+`
 * and the Hadamard product, provided as friend free functions delegating to
 * the entry-wise `TensorFibersCore` algebra.
 */
template <typename Scalar> class TensorFibers {
  public:
    TensorFibers() = default;

    /*!
     * @brief Constructs from per-core slab unfoldings and their shared
     * skeleton.
     * @param slabs Slab `k` as a left unfolding, shape
     * `(leftFiberCount(k) * n_k) x rightFiberCount(k)`.
     * @param indices The skeleton the slabs were evaluated on.
     */
    TensorFibers(std::vector<Eigen::MatrixX<Scalar>> slabs,
                 std::shared_ptr<const FiberIndices> indices)
        : indices(std::move(indices)) {
        assert(this->indices && "TensorFibers: null skeleton.");
        assert(slabs.size() == this->indices->order() &&
               "TensorFibers: slab count must equal the tensor order.");
        cores.reserve(slabs.size());
        for (std::size_t k = 0; k < slabs.size(); ++k) {
            const auto l_prev =
                static_cast<Eigen::Index>(this->indices->leftFiberCount(k));
            assert(slabs[k].rows() % l_prev == 0 &&
                   "TensorFibers: slab rows must be divisible by the left "
                   "fiber count.");
            assert(slabs[k].cols() ==
                       static_cast<Eigen::Index>(
                           this->indices->rightFiberCount(k)) &&
                   "TensorFibers: slab columns must equal the right fiber "
                   "count.");
            const Eigen::Index n = slabs[k].rows() / l_prev;
            cores.emplace_back(std::move(slabs[k]), n);
        }
    }

    /*!
     * @brief Constructs from per-core slabs and their shared skeleton.
     * @param cores Slab `k` as a `TensorFibersCore` of shape
     * `leftFiberCount(k) x n_k x rightFiberCount(k)`.
     * @param indices The skeleton the slabs were evaluated on.
     */
    TensorFibers(std::vector<TensorFibersCore<Scalar>> cores,
                 std::shared_ptr<const FiberIndices> indices)
        : cores(std::move(cores)), indices(std::move(indices)) {
        assert(this->indices && "TensorFibers: null skeleton.");
        assert(this->cores.size() == this->indices->order() &&
               "TensorFibers: slab count must equal the tensor order.");
    }

    /*! @brief The tensor order d (number of fiber cores). */
    [[nodiscard]] std::size_t order() const { return cores.size(); }

    /*! @brief Fiber core `k`; its left unfolding (`leftUnfolding()`) is the
     * shape consumed by `TensorTrainCore`. */
    [[nodiscard]] const TensorFibersCore<Scalar> &core(std::size_t k) const {
        assert(k < cores.size() && "TensorFibers: core index OOR.");
        return cores[k];
    }

    /*! @brief The shared skeleton the slabs were sampled on. */
    [[nodiscard]] const std::shared_ptr<const FiberIndices> &skeleton() const {
        return indices;
    }

    /*!
     * @brief Slab-wise sum of two `TensorFibers` sampled on the same skeleton.
     *
     * The pointwise sum of the two sampled fiber tensors, entry by entry on the
     * shared fiber set.
     */
    friend TensorFibers operator+(const TensorFibers &a,
                                  const TensorFibers &b) {
        assertCombinable(a, b);
        std::vector<TensorFibersCore<Scalar>> out;
        out.reserve(a.cores.size());
        for (std::size_t k = 0; k < a.cores.size(); ++k) {
            out.push_back(a.cores[k] + b.cores[k]);
        }
        return TensorFibers(std::move(out), a.indices);
    }

    /*!
     * @brief Scales every slab of a `TensorFibers` by a scalar.
     *
     * The pointwise product of the sampled fiber tensor with `scalar`, entry by
     * entry on the fiber set; the skeleton is shared unchanged.
     */
    friend TensorFibers operator*(Scalar scalar, const TensorFibers &a) {
        std::vector<TensorFibersCore<Scalar>> out;
        out.reserve(a.cores.size());
        for (const auto &core : a.cores) {
            out.push_back(scalar * core);
        }
        return TensorFibers(std::move(out), a.indices);
    }

    /*!
     * @brief Slab-wise elementwise (Hadamard) product of two `TensorFibers`
     * sampled on the same skeleton.
     */
    friend TensorFibers hadamardProduct(const TensorFibers &a,
                                        const TensorFibers &b) {
        assertCombinable(a, b);
        std::vector<TensorFibersCore<Scalar>> out;
        out.reserve(a.cores.size());
        for (std::size_t k = 0; k < a.cores.size(); ++k) {
            out.push_back(hadamardProduct(a.cores[k], b.cores[k]));
        }
        return TensorFibers(std::move(out), a.indices);
    }

  private:
    // Slab k of shape l_{k-1} x n_k x rho_k.
    std::vector<TensorFibersCore<Scalar>> cores;

    // The skeleton the slabs were evaluated on. Shared because a whole family
    // of TensorFibers is typically sampled on one fixed set of indices.
    std::shared_ptr<const FiberIndices> indices;

    /*!
     * @brief Asserts two operands are slab-wise combinable: same order and
     * same shared skeleton (the per-slab shape checks live in the
     * `TensorFibersCore` algebra).
     */
    static void assertCombinable(const TensorFibers &a, const TensorFibers &b) {
        static_cast<void>(a);
        static_cast<void>(b);
        assert(a.cores.size() == b.cores.size() &&
               "TensorFibers combine: mismatched tensor order.");
        assert(a.indices == b.indices &&
               "TensorFibers combine: operands must share the same skeleton.");
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_H
