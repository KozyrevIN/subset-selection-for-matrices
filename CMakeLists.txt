cmake_minimum_required(VERSION 3.27)

project(Subset_selection)

set(SOURCES 
    src/main.cpp
    src/matrix_utilities.cpp 
    src/matrix_generator.cpp
    src/tester.cpp 
    src/subset_selection_algorithms/subset_selector.cpp
    src/subset_selection_algorithms/rank_revealing_QR_selector.cpp
    src/subset_selection_algorithms/greedy_removal_1_selector.cpp
    src/subset_selection_algorithms/greedy_removal_2_selector.cpp
    src/subset_selection_algorithms/greedy_selection_1_selector.cpp
)

set(CMAKE_CXX_FLAGS "-O3")

add_executable(main ${SOURCES})