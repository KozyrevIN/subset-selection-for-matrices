#ifndef MAT_SUBSET_EXPERIMENTS_UTILS_H
#define MAT_SUBSET_EXPERIMENTS_UTILS_H

#include <algorithm> // For std::replace
#include <string>    // For std::string

namespace MatSubset::Experiments::Utils {

/**
 * @brief Replaces all spaces in a string with underscores.
 *
 * This function takes a string, creates a copy of it, and replaces every
 * space character ' ' with an underscore character '_'.
 *
 * @param text The input string.
 * @return A new string with spaces replaced by underscores.
 */
std::string add_underscores(const std::string &text) {
    std::string result = text;
    std::replace(result.begin(), result.end(), ' ', '_');
    return result;
}

/**
 * @brief Replaces all underscores in a string with spaces.
 *
 * This function takes a string, creates a copy of it, and replaces every
 * underscore character '_' with a space character ' '.
 *
 * @param text The input string.
 * @return A new string with underscores replaced by spaces.
 */
std::string remove_underscores(const std::string &text) {
    std::string result = text;
    std::replace(result.begin(), result.end(), '_', ' ');
    return result;
}

} // namespace MatSubset::Experiments::Utils

#endif // MAT_SUBSET_EXPERIMENTS_UTILS_H