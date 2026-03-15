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

/*!
 * @brief Specifies the initialization strategy for volume-based iterative
 * selectors (DominantSelector, VolumeAddRemoveSelector).
 */
enum class Initialization {
    CPQR,     /*!< Initialize m columns via column-pivoted QR only. */
    Greedy,   /*!< Initialize k columns via CPQR + greedy addition. */
    Advanced  /*!< Initialization involving oversampling, exchanges and
                   downsampling. Provides lower theoretical complexity but can
                   be suboptimal in practice. */
};

} // namespace MatSubset

#endif // MAT_SUBSET_ENUMS_H