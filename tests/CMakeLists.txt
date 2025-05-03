cmake_minimum_required(VERSION 3.5)

project(MatSubsetTests 
        LANGUAGES CXX)

# Options
option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)

# Required packages
find_package(MatSubset REQUIRED)
find_package(doctest REQUIRED)

# Create binaries
file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} doctest::doctest MatSubset::MatSubset)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# Enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(Greeter PUBLIC -Wall -Wpedantic -Wextra -Werror)
  elseif(MSVC)
    target_compile_options(Greeter PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()

# Enable tests
enable_testing()

# Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND
# ${PROJECT_NAME})

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(${PROJECT_NAME})

# Code coverage test
if(ENABLE_TEST_COVERAGE)
  target_compile_options(MatSubset PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(MatSubset PUBLIC -fprofile-arcs -ftest-coverage)
endif()