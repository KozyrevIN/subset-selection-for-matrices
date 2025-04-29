# This script builds all examples 

cmake_minimum_required(VERSION 3.5)

project(MatSubsetExamples LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/basic_usage ${CMAKE_BINARY_DIR}/examples/basic_usage)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/theoretical_bounds ${CMAKE_BINARY_DIR}/examples/theoretical_bounds)