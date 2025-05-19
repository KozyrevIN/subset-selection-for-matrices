cmake_minimum_required(VERSION 3.28)

project(MatSubsetTheoreticalBoundsDemo LANGUAGES CXX)

# Creating executable target
add_executable(${PROJECT_NAME} src/main.cpp) 

# Setting c++ standard
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "theoretical_bounds"
)

# Managing dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MatSubset 0.0 REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    MatSubset::MatSubset
)