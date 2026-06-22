#ifndef MAT_SUBSET_EXPERIMENTS_SELECTOR_FACTORY_H
#define MAT_SUBSET_EXPERIMENTS_SELECTOR_FACTORY_H

#include <functional> // For std::function
#include <map>        // For std::map
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string

#include <MatSubset/MatSubset.h> // For Selector classes
#include <nlohmann/json.hpp>     // For nlohmann::json

namespace MatSubset::Experiments {

/*!
 * @brief Factory for creating selector algorithm instances from JSON
 * configuration.
 * @tparam Scalar The underlying scalar type (e.g., float, double).
 *
 * This is a flexible base factory class that manages a registry of selector
 * creation functions. Selectors are identified by their algorithm name and
 * can be instantiated from JSON configuration objects.
 *
 * The factory uses a registration pattern where each selector type is
 * associated with a creation function (Creator) that takes a JSON config
 * and returns a unique_ptr to the selector instance.
 *
 * This class is designed to be extended (see DefaultSelectorFactory) or
 * used directly with custom registrations for testing or experimentation.
 */
template <typename Scalar> class SelectorFactory {
  public:
    /*!
     * @brief Default constructor.
     *
     * Creates an empty factory with no registered selectors. Use
     * registerSelector() to add selectors, or use DefaultSelectorFactory
     * which comes pre-populated with all standard selectors.
     */
    SelectorFactory() = default;

    /*!
     * @brief Type alias for selector creation functions.
     *
     * A Creator is a callable that takes a JSON configuration object and
     * returns a unique_ptr to a newly created selector instance. The JSON
     * config may contain algorithm-specific parameters.
     */
    using Creator = std::function<std::unique_ptr<SelectorBase<Scalar>>(
        const nlohmann::json &)>;

    /*!
     * @brief Creates a selector instance from JSON configuration.
     * @param config JSON object containing at least a "name" field
     *        with the selector's algorithm name. May contain additional
     *        algorithm-specific parameters (e.g., "eps", "c").
     * @return A unique_ptr to the newly created selector instance.
     * @throws std::runtime_error if the algorithm name is not registered.
     * @throws nlohmann::json::exception if "name" field is missing or invalid.
     *
     * Example JSON configurations:
     * @code
     * {"name": "dual set"}                          // No-args selector
     * {"name": "spectral selection", "eps": 0.01}   // Selector with eps
     * {"name": "dominant", "c": 1.01}               // DominantSelector with c
     * @endcode
     */
    std::unique_ptr<SelectorBase<Scalar>>
    create(const nlohmann::json &config) const {
        const std::string &name = config.at("name").get<std::string>();
        auto it = creators.find(name);
        if (it == creators.end()) {
            throw std::runtime_error("Unknown selector: " + name);
        }
        return it->second(config);
    }

    /*!
     * @brief Registers a new selector type with the factory.
     * @param name The algorithm name (as returned by getAlgorithmName()).
     * @param creator A callable that creates selector instances from config.
     * @return true if registration succeeded, false if name already registered.
     *
     * This allows custom selectors or alternative configurations to be
     * registered at runtime.
     *
     * Example:
     * @code
     * factory.registerSelector("my custom", [](const nlohmann::json& config) {
     *     return std::make_unique<MyCustomSelector<double>>();
     * });
     * @endcode
     */
    bool registerSelector(const std::string &name, Creator creator) {
        return creators.emplace(name, creator).second;
    }

  private:
    std::map<std::string, Creator> creators; ///< Registry of selector creators.
};

/*!
 * @brief Pre-configured factory with all standard MatSubset selectors
 * registered.
 * @tparam Scalar The underlying scalar type (e.g., float, double).
 *
 * This factory comes pre-populated with all standard selector algorithms
 * from the MatSubset library, ready to use. It handles different constructor
 * signatures automatically:
 * - No-args selectors (DualSetSelector, etc.)
 * - Selectors with optional eps parameter (SpectralSelectionSelector, etc.)
 * - Selectors with required c parameter (RectMaxvolSelector)
 * - Selectors with required c and optional initialization strategy
 *   (DominantSelector, VolumeAddRemoveSelector)
 *
 * Registered selectors:
 * - "derandomized volume" - DerandomizedVolumeSelector (optional "eps")
 * - "dominant" - DominantSelector (requires "c" parameter)
 * - "dual set" - DualSetSelector
 * - "forward iterative volume sampling" - ForwardIterativeVolumeSamplingSelector (optional "seed")
 * - "frobenius removal" - FrobeniusRemovalSelector (optional "eps")
 * - "frobenius selection" - FrobeniusSelectionSelector
 * - "interlacing families" - InterlacingFamiliesSelector (optional "eps")
 * - "leverage scores" - LeverageScoresSelector (optional "seed")
 * - "random columns" - RandomColumnsSelector (optional "seed")
 * - "rect-maxvol" - RectMaxvolSelector (requires "c" parameter)
 * - "reverse iterative volume sampling" - ReverseIterativeVolumeSamplingSelector (optional "seed")
 * - "spectral removal" - SpectralRemovalSelector (optional "eps")
 * - "spectral selection" - SpectralSelectionSelector (optional "eps")
 * - "volume add-remove" - VolumeAddRemoveSelector (requires "c" parameter)
 * - "volume removal" - VolumeRemovalSelector
 *
 * Usage example:
 * @code
 * DefaultSelectorFactory<double> factory;
 * nlohmann::json config = {{"name", "spectral selection"}, {"eps", 0.01}};
 * auto selector = factory.create(config);
 * auto indices = selector->selectSubset(X, k);
 * @endcode
 */
template <typename Scalar>
class DefaultSelectorFactory : public SelectorFactory<Scalar> {
  public:
    /*!
     * @brief Constructor that registers all standard selectors.
     *
     * Automatically populates the factory with creation functions for all
     * standard MatSubset selector algorithms. After construction, the
     * factory is immediately ready to create selectors via create().
     */
    DefaultSelectorFactory() {
        registerEpsArgSelector<DerandomizedVolumeSelector>();
        registerCArgWithInitSelector<DominantSelector>();
        registerNoArgsSelector<DualSetSelector>();
        registerSeedArgSelector<ForwardIterativeVolumeSamplingSelector>();
        registerEpsArgSelector<FrobeniusRemovalSelector>();
        registerNoArgsSelector<FrobeniusSelectionSelector>();
        registerEpsArgSelector<InterlacingFamiliesSelector>();
        registerSeedArgSelector<LeverageScoresSelector>();
        registerSeedArgSelector<RandomColumnsSelector>();
        registerCArgSelector<RectMaxvolSelector>();
        registerSeedArgSelector<ReverseIterativeVolumeSamplingSelector>();
        registerEpsArgSelector<SpectralRemovalSelector>();
        registerEpsArgSelector<SpectralSelectionSelector>();
        registerCArgWithInitSelector<VolumeAddRemoveSelector>();
        registerNoArgsSelector<VolumeRemovalSelector>();
    }

  private:
    /*!
     * @brief Registers a selector with no constructor arguments.
     * @tparam Selector Template template parameter for the selector class.
     *
     * Creates a registration for selectors that only need default construction.
     * The selector's algorithm name is automatically retrieved via
     * getAlgorithmName().
     */
    template <template <typename> class Selector>
    void registerNoArgsSelector() {
        auto dummy = std::make_unique<Selector<Scalar>>();
        std::string name = dummy->getAlgorithmName();
        typename SelectorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                return std::make_unique<Selector<Scalar>>();
            };
        this->registerSelector(name, creator);
    }

    /*!
     * @brief Registers a selector with an optional eps constructor argument.
     * @tparam Selector Template template parameter for the selector class.
     *
     * Creates a registration for selectors that accept an optional eps
     * (epsilon) parameter. If "eps" is present in the JSON config, it's
     * passed to the constructor; otherwise, default construction is used.
     *
     * Applies to: SpectralSelectionSelector, FrobeniusRemovalSelector,
     * SpectralRemovalSelector, InterlacingFamiliesSelector.
     */
    template <template <typename> class Selector>
    void registerEpsArgSelector() {
        auto dummy = std::make_unique<Selector<Scalar>>();
        std::string name = dummy->getAlgorithmName();
        typename SelectorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                if (config.contains("eps")) {
                    return std::make_unique<Selector<Scalar>>(
                        config.at("eps").get<Scalar>());
                } else {
                    return std::make_unique<Selector<Scalar>>();
                }
            };
        this->registerSelector(name, creator);
    }

    /*!
     * @brief Registers a selector with a required c constructor argument.
     * @tparam Selector Template template parameter for the selector class.
     *
     * Creates a registration for selectors that require a "c" parameter
     * (c > 1). The JSON config must contain {"c": value}.
     *
     * Applies to: RectMaxvolSelector.
     *
     * @note Uses dummy value of 1.01 for registration only; actual instances
     * use the c value from config.
     */
    template <template <typename> class Selector> void registerCArgSelector() {
        auto dummy = std::make_unique<Selector<Scalar>>(1.01);
        std::string name = dummy->getAlgorithmName();
        typename SelectorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                return std::make_unique<Selector<Scalar>>(
                    config.at("c").get<Scalar>());
            };
        this->registerSelector(name, creator);
    }

    /*!
     * @brief Registers a selector with required c and optional initialization
     * strategy argument.
     * @tparam Selector Template template parameter for the selector class.
     *
     * Creates a registration for selectors that require a "c" parameter
     * and optionally accept an "initialization" string ("CPQR", "greedy",
     * or "advanced"). Defaults to "greedy".
     *
     * Applies to: DominantSelector, VolumeAddRemoveSelector.
     *
     * @note Uses dummy value of 1.01 for registration only; actual instances
     * use parameters from config.
     */
    template <template <typename> class Selector>
    void registerCArgWithInitSelector() {
        auto dummy = std::make_unique<Selector<Scalar>>(1.01);
        std::string name = dummy->getAlgorithmName();
        typename SelectorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                Scalar c = config.at("c").get<Scalar>();
                std::string init_str = config.value("initialization", "greedy");
                MatSubset::Initialization init;
                if (init_str == "CPQR") {
                    init = MatSubset::Initialization::CPQR;
                } else if (init_str == "advanced") {
                    init = MatSubset::Initialization::Advanced;
                } else {
                    init = MatSubset::Initialization::Greedy;
                }
                return std::make_unique<Selector<Scalar>>(c, init);
            };
        this->registerSelector(name, creator);
    }

    /*!
     * @brief Registers a selector with an optional seed constructor argument.
     * @tparam Selector Template template parameter for the selector class.
     *
     * Creates a registration for selectors that accept an optional seed
     * parameter for reproducible random number generation. If "seed" is
     * present in the JSON config, it's passed to the constructor; otherwise,
     * default construction is used (which will use random seeding).
     *
     * Applies to: RandomColumnsSelector.
     */
    template <template <typename> class Selector>
    void registerSeedArgSelector() {
        auto dummy = std::make_unique<Selector<Scalar>>();
        std::string name = dummy->getAlgorithmName();
        typename SelectorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                if (config.contains("seed")) {
                    return std::make_unique<Selector<Scalar>>(
                        config.at("seed").get<std::mt19937::result_type>());
                } else {
                    return std::make_unique<Selector<Scalar>>();
                }
            };
        this->registerSelector(name, creator);
    }


};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_SELECTOR_FACTORY_H
