cmake_minimum_required(VERSION 3.27)

project(Subset_selection)

set(SOURCES 
    main.cpp
)

set(CMAKE_CXX_FLAGS "-fopenmp")

add_executable(main ${SOURCES})

target_compile_features(main PRIVATE cxx_std_20)