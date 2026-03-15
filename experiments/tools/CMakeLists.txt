cmake_minimum_required(VERSION 3.28)

project(MatSubsetExperiments LANGUAGES CXX)

# Fetch nlohmann_json using FetchContent
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# Creating executable target
add_executable(${PROJECT_NAME} src/main.cpp)

# Add include directory for headers
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Setting c++ standard
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "Tester"
)

# Add conditional compile flags for Debug and Release
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -g -march=native -DNDEBUG>
)

# Managing dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MatSubset 0.0 REQUIRED)
find_package(TBB REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    MatSubset::MatSubset
    TBB::tbb
)