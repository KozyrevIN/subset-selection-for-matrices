cmake_minimum_required(VERSION 3.27)

project(Subset_selection)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(jsoncpp REQUIRED)

set(SOURCES 
    main.cpp
)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp")

add_executable(main ${SOURCES})

target_link_libraries(main Eigen3::Eigen jsoncpp)
target_compile_features(main PRIVATE cxx_std_20)