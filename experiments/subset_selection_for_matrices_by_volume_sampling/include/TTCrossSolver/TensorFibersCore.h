#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_CORE_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_CORE_H

#include <cassert> // For assert
#include <utility> // For std::move

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include "TTCrossSolver/CoreBase.h" // For CoreBase

namespace MatSubset::Experiments {

/*!
 * @brief One sampled slab of a tensor: the fiber counterpart of a TT-core,
 * logically of shape \f$ \ell_0 \times n \times \rho_1 \f$ (left fiber count,
 * mode, right fiber count).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Storage and the per-core sweep steps live in `CoreBase`; this class adds
 * the slab-specific combining operations. Because the entries of a slab are
 * *tensor values sampled on a fixed fiber set* (not bond coefficients),
 * slabs combine entry-wise: `+` is the pointwise sum, `*` the pointwise
 * scaling and `hadamardProduct` the pointwise product of the sampled
 * tensors — in contrast to TT-cores, whose sum concatenates blocks and whose
 * Hadamard product is a Kronecker product of slices.
 */
template <typename Scalar> class TensorFibersCore : public CoreBase<Scalar> {

  public:
    /*!
     * @brief Constructs a zero-initialized slab of shape r0 x n x r1.
     */
    TensorFibersCore(Eigen::Index r0, Eigen::Index n, Eigen::Index r1)
        : CoreBase<Scalar>(r0, n, r1) {}

    /*!
     * @brief Constructs a slab from an existing left unfolding.
     * @param unfolding Left unfolding of shape (l0 * n) x rho1.
     * @param n The mode size, needed to recover l0 from unfolding.rows().
     */
    TensorFibersCore(Eigen::MatrixX<Scalar> unfolding, Eigen::Index n)
        : CoreBase<Scalar>(std::move(unfolding), n) {}

    /*! @brief Entry-wise sum of two slabs sampled on the same fibers. */
    friend TensorFibersCore operator+(const TensorFibersCore &a,
                                      const TensorFibersCore &b) {
        assertSameShape(a, b, "TensorFibersCore operator+");
        return TensorFibersCore(a.left_unfolding + b.left_unfolding, a.mode);
    }

    /*! @brief Entry-wise scaling of a slab. */
    friend TensorFibersCore operator*(Scalar scalar,
                                      const TensorFibersCore &a) {
        return TensorFibersCore(scalar * a.left_unfolding, a.mode);
    }

    /*! @brief Entry-wise (Hadamard) product of two slabs sampled on the same
     * fibers. */
    friend TensorFibersCore hadamardProduct(const TensorFibersCore &a,
                                            const TensorFibersCore &b) {
        assertSameShape(a, b, "TensorFibersCore hadamardProduct");
        return TensorFibersCore(a.left_unfolding.cwiseProduct(b.left_unfolding),
                                a.mode);
    }

  private:
    static void assertSameShape(const TensorFibersCore &a,
                                const TensorFibersCore &b, const char *what) {
        static_cast<void>(a);
        static_cast<void>(b);
        static_cast<void>(what);
        assert(a.left_rank == b.left_rank && a.mode == b.mode &&
               a.right_rank == b.right_rank &&
               "TensorFibersCore: operands must have the same shape.");
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_CORE_H
