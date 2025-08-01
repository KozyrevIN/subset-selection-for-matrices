# This script installs/uninstalls MatSubset library

cmake_minimum_required(VERSION VERSION 3.28)

project(
    MatSubset
    VERSION 0.0
    LANGUAGES CXX)

# In-source build guard
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
message(
    FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
)
endif()

# Guard against bad build-type strings
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Creating header only library target
add_library(${PROJECT_NAME} INTERFACE)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# Being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# Managing dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

# Configuring project directories
include(GNUInstallDirs)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Installing targets and exporting them to Targets.cmake
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)

# Configuring and installing the Config.cmake and ConfigVersion.cmake files
include(CMakePackageConfigHelpers)

# Generateing the ConfigVersion.cmake file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# Configuring the Config.cmake file from the .in template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Installing the generated Config.cmake, ConfigVersion.cmake, and Targets.cmake
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Installing headers
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")

# Exporting targets for use in build tree
export(
    TARGETS MatSubset
    NAMESPACE ${PROJECT_NAME}::
    FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

# Adding uninstall target
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Uninstall.cmake")

