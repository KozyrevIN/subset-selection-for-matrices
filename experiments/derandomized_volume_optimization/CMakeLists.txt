cmake_minimum_required(VERSION 3.28)

project(DerandomizedVolumeOptimization LANGUAGES CXX)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "bench"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -g -march=native -DNDEBUG>
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MatSubset 0.0 REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    MatSubset::MatSubset
)
