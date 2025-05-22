# This script builds all demo executables

cmake_minimum_required(VERSION 3.28)

project(MatSubsetExamples LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/basic_usage ${CMAKE_CURRENT_BINARY_DIR}/basic_usage)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/theoretical_bounds ${CMAKE_CURRENT_BINARY_DIR}/theoretical_bounds)