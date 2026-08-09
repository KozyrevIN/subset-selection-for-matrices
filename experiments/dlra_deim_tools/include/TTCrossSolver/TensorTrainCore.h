#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_CORE_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_CORE_H

#include <cassert> // For assert
#include <utility> // For std::move

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include "TTCrossSolver/CoreBase.h" // For CoreBase

namespace MatSubset::Experiments {

/*!
 * @brief A single TT-core of a tensor-train, logically of shape
 * \f$ r_0 \times n \times r_1 \f$ (left-rank, mode, right-rank).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Storage and the per-core sweep steps (orthogonalization, truncated SVD,
 * factor absorption, index selection) live in `CoreBase`; this class adds the
 * TT-specific combining operation `zip`, the per-core step of applying a TT
 * operator to a TT tensor.
 */
template <typename Scalar> class TensorTrainCore : public CoreBase<Scalar> {

  public:
    /*!
     * @brief Constructs a zero-initialized core of shape r0 x n x r1.
     */
    TensorTrainCore(Eigen::Index r0, Eigen::Index n, Eigen::Index r1)
        : CoreBase<Scalar>(r0, n, r1) {}

    /*!
     * @brief Constructs a core from an existing left unfolding.
     * @param unfolding Left unfolding of shape (r0 * n) x r1.
     * @param n The mode size, needed to recover r0 from unfolding.rows().
     */
    TensorTrainCore(Eigen::MatrixX<Scalar> unfolding, Eigen::Index n)
        : CoreBase<Scalar>(std::move(unfolding), n) {}

    /*!
     * @brief Contracts this operator core with a tensor core over a shared
     * mode ("zips" them).
     * @param tensor The tensor core \f$ B \f$, of shape
     * \f$ s_0 \times n \times s_1 \f$; its mode size \f$ n \f$ is the
     * operator's input size and is summed over.
     * @param out_size The operator's output mode size \f$ m \f$.
     * @param in_size The operator's input mode size \f$ n \f$; must equal
     * `tensor.modeSize()`. `out_size * in_size` must equal this core's mode
     * size, which is read row-major as \f$ \text{mode} = \text{out} \cdot n +
     * \text{in} \f$ (output index slower).
     * @return The contracted core of shape
     * \f$ (r_0 s_0) \times m \times (r_1 s_1) \f$: at output value
     * \f$ o \f$ its slice is \f$ \sum_{\text{in}} A[:, o, \text{in}, :] \otimes
     * B[:, \text{in}, :] \f$, with the operator rank the outer (slower) factor
     * of each Kronecker bond (bond index \f$ = a \, s + b \f$).
     *
     * This is the per-core step of a TT operator applied to a TT tensor: the
     * bond ranks multiply and the shared physical mode is contracted.
     */
    [[nodiscard]] TensorTrainCore zip(const TensorTrainCore &tensor,
                                      Eigen::Index out_size,
                                      Eigen::Index in_size) const {
        assert(out_size * in_size == this->mode &&
               "zip: out_size * in_size must equal the operator mode size.");
        assert(in_size == tensor.mode &&
               "zip: in_size must equal the tensor core's mode size.");

        const Eigen::Index s0 = tensor.left_rank;
        const Eigen::Index s1 = tensor.right_rank;
        const Eigen::Index out_r0 = this->left_rank * s0;
        const Eigen::Index out_r1 = this->right_rank * s1;

        TensorTrainCore result(out_r0, out_size, out_r1);
        for (Eigen::Index o = 0; o < out_size; ++o) {
            // Slice of the result core at output mode o: (r0*s0) x (r1*s1).
            Eigen::MatrixX<Scalar> slice =
                Eigen::MatrixX<Scalar>::Zero(out_r0, out_r1);
            for (Eigen::Index in = 0; in < in_size; ++in) {
                const Eigen::MatrixX<Scalar> A =
                    this->modeSlice(o * in_size + in); // r0 x r1
                const Eigen::MatrixX<Scalar> B =
                    tensor.modeSlice(in); // s0 x s1
                // kron(A, B): block (a, c) is A(a, c) * B, operator rank outer.
                for (Eigen::Index a = 0; a < this->left_rank; ++a) {
                    for (Eigen::Index c = 0; c < this->right_rank; ++c) {
                        slice.block(a * s0, c * s1, s0, s1) += A(a, c) * B;
                    }
                }
            }
            result.setModeSlice(o, slice);
        }
        return result;
    }

    /*!
     * @brief Block core for the sum of two trains at this position (the
     * per-core step of `operator+`).
     * @param other The other train's core at the same position.
     * @param is_first True at the first core (left boundary bond stays 1).
     * @param is_last True at the last core (right boundary bond stays 1).
     * @return The block core: \f$ [A(i) \; B(i)] \f$ at the first core,
     * block-diagonal \f$ \operatorname{diag}(A(i), B(i)) \f$ in the interior,
     * \f$ [A(i); B(i)] \f$ at the last core (boundary bonds shared, so the
     * blocks overlap and add there).
     */
    [[nodiscard]] TensorTrainCore add(const TensorTrainCore &other,
                                      bool is_first, bool is_last) const {
        const Eigen::Index n = this->mode;
        const Eigen::Index r0 =
            is_first ? 1 : this->left_rank + other.left_rank;
        const Eigen::Index r1 =
            is_last ? 1 : this->right_rank + other.right_rank;

        TensorTrainCore result(r0, n, r1);
        for (Eigen::Index i = 0; i < n; ++i) {
            // This core occupies the top-left block, other the bottom-right;
            // at the boundaries the singleton bond dimension is shared (`+=`
            // adds the overlapping contributions there, e.g. for order 1).
            Eigen::MatrixX<Scalar> slice = Eigen::MatrixX<Scalar>::Zero(r0, r1);
            const Eigen::Index other_row = is_first ? 0 : this->left_rank;
            const Eigen::Index other_col = is_last ? 0 : this->right_rank;
            slice.block(0, 0, this->left_rank, this->right_rank) =
                this->modeSlice(i);
            slice.block(other_row, other_col, other.left_rank,
                        other.right_rank) += other.modeSlice(i);
            result.setModeSlice(i, slice);
        }
        return result;
    }

    /*!
     * @brief Scales this core by a constant, folded into the left unfolding.
     */
    [[nodiscard]] TensorTrainCore scale(Scalar factor) const {
        TensorTrainCore result(*this);
        result.setLeftUnfolding(factor * result.leftUnfolding());
        return result;
    }

    /*!
     * @brief Entry-wise (Hadamard) product core with `other` (the per-core
     * step of `hadamardProduct`).
     * @param other The other train's core at the same position.
     * @return The Kronecker-product core \f$ C(i) = A(i) \otimes B(i) \f$;
     * its bond ranks are the products of the operands' ranks.
     */
    [[nodiscard]] TensorTrainCore
    hadamardProduct(const TensorTrainCore &other) const {
        const Eigen::Index n = this->mode;
        TensorTrainCore result(this->left_rank * other.left_rank, n,
                               this->right_rank * other.right_rank);
        for (Eigen::Index i = 0; i < n; ++i) {
            const Eigen::MatrixX<Scalar> A = this->modeSlice(i);
            const Eigen::MatrixX<Scalar> B = other.modeSlice(i);
            Eigen::MatrixX<Scalar> slice(A.rows() * B.rows(),
                                         A.cols() * B.cols());
            for (Eigen::Index p = 0; p < A.rows(); ++p) {
                for (Eigen::Index q = 0; q < A.cols(); ++q) {
                    slice.block(p * B.rows(), q * B.cols(), B.rows(),
                                B.cols()) = A(p, q) * B;
                }
            }
            result.setModeSlice(i, slice);
        }
        return result;
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_CORE_H
