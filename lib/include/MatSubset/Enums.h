#ifndef MAT_SUBSET_ENUMS_H
#define MAT_SUBSET_ENUMS_H

namespace MatSubset {

/*!
 * @brief Specifies the type of matrix norm to be used in calculations.
 */
enum class Norm {
    Frobenius, /*!< The Frobenius norm. */
    Spectral   /*!< The Spectral norm (or Schatten 2-norm). */
};

} // namespace MatSubset

#endif // MAT_SUBSET_ENUMS_H