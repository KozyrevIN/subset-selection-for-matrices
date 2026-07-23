#ifndef MAT_SUBSET_EXPERIMENTS_FIBER_INDICES_H
#define MAT_SUBSET_EXPERIMENTS_FIBER_INDICES_H

#include <cassert> // For assert
#include <cstddef> // For std::size_t
#include <utility> // For std::move
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::Index

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

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_FIBER_INDICES_H
