# This script builds the test executable for the TT-cross experiment
cmake_minimum_required(VERSION 3.28)

project(TTCrossSolverTests
        LANGUAGES CXX)

# Required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MatSubset 0.0 REQUIRED)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOXML)

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)

# Create binaries
file(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "Tests")

# The TT-cross headers live in the experiment's include/ directory; the
# acoustic-equation rhs in its own subproject's include/.
target_include_directories(${PROJECT_NAME} PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/../include")

target_link_libraries(${PROJECT_NAME} doctest::doctest
                                      Eigen3::Eigen
                                      MatSubset::MatSubset
                                      VTK::CommonCore
                                      VTK::CommonDataModel
                                      VTK::IOXML)
vtk_module_autoinit(TARGETS ${PROJECT_NAME}
                    MODULES VTK::CommonCore VTK::CommonDataModel VTK::IOXML)

# Enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wpedantic -Wextra -Werror)
    elseif(MSVC)
        target_compile_options(${PROJECT_NAME} PUBLIC /W4 /WX)
        target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
    endif()
endif()

# Enable tests
enable_testing()

include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")
doctest_discover_tests("${PROJECT_NAME}")
