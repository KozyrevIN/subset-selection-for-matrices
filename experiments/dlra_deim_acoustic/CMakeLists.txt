# This script builds the acoustic-wave demo executable for the TT-cross
# experiment; the tests have their own standalone project in tests/.
cmake_minimum_required(VERSION 3.28)

project(TTCrossAcoustic
        LANGUAGES CXX)

# Required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MatSubset 0.0 REQUIRED)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOXML)

# Create binaries
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
set_target_properties(${PROJECT_NAME} PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AcousticWave")

target_include_directories(${PROJECT_NAME} PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen
                                      MatSubset::MatSubset
                                      VTK::CommonCore
                                      VTK::CommonDataModel
                                      VTK::IOXML)
vtk_module_autoinit(TARGETS ${PROJECT_NAME}
                    MODULES VTK::CommonCore VTK::CommonDataModel VTK::IOXML)

# Enable compiler warnings. No -Werror here: the solver headers guard several
# locals behind assert(), so a Release (NDEBUG) build legitimately reports them
# as unused. That is a property of the shared headers, not this demo.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wpedantic -Wextra)
    # Aggressive optimization tuned to the build host in Release.
    target_compile_options(${PROJECT_NAME} PUBLIC
                           $<$<CONFIG:Release>:-O3 -march=native>)
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC /W4)
endif()
