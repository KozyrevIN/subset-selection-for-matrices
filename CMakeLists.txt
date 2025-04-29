cmake_minimum_required(VERSION 3.5)

project(
    MatSubset
    VERSION 0.0.0
    LANGUAGES CXX)

# Include guard
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
message(
    FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
)
endif()

# Creating header only library target
add_library(${PROJECT_NAME} INTERFACE)

# Being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# Resolving dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

# Configuring project directories
include(GNUInstallDirs)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Install targets and export them to Targets.cmake
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)

# Configure and install the Config.cmake and ConfigVersion.cmake files
include(CMakePackageConfigHelpers)

# Generate the ConfigVersion.cmake file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# Configure the Config.cmake file from the .in template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Install the generated Config.cmake, ConfigVersion.cmake, and Targets.cmake
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Install headers
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")

# Add uninstall target
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Uninstall.cmake")

        

