cmake_minimum_required(VERSION 3.5)

project(MatSubsetBasicUsage LANGUAGES CXX)

# Creating executable target
add_executable(${PROJECT_NAME} src/main.cpp) 

# Setting c++ standard
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "basic_usage"
)

# Managing dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# You can remove if clause when using MatSubset in your project. 
# It is here to allow simoltaneous installation of library and building examples.
if(NOT TARGET MatSubset::MatSubset)
    find_package(MatSubset 0.0.0 REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    MatSubset::MatSubset
)