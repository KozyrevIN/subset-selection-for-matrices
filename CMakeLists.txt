cmake_minimum_required(VERSION 3.27)

project(Subset_selection)

set(SOURCES 
    main.cpp
)

set(CMAKE_CXX_FLAGS "-O3")

add_executable(main ${SOURCES})