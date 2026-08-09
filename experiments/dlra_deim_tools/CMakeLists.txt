# This script builds the acoustic-wave demo executable for the TT-cross
# experiment; the tests have their own standalone project in tests/.
cmake_minimum_required(VERSION 3.28)

project(TTCrossAcoustic
        LANGUAGES CXX)

# Required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MatSubset 0.0 REQUIRED)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOXML)

# Create binaries: the TT demo (AcousticWave) and its QTT variant
# (AcousticWaveQtt), which share the include/ headers.
add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
set_target_properties(${PROJECT_NAME} PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AcousticWave")

add_executable(AcousticWaveQtt "${CMAKE_CURRENT_SOURCE_DIR}/src/main_qtt.cpp")
set_target_properties(AcousticWaveQtt PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AcousticWaveQtt")

# The 3D Allen-Cahn demo (AllenCahn), sharing the same include/ headers as the
# acoustic demos.
add_executable(AllenCahn "${CMAKE_CURRENT_SOURCE_DIR}/src/main_allen_cahn.cpp")
set_target_properties(AllenCahn PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AllenCahn")

foreach(target ${PROJECT_NAME} AcousticWaveQtt AllenCahn)
    target_include_directories(${target} PRIVATE
                               "${CMAKE_CURRENT_SOURCE_DIR}/include")

    target_link_libraries(${target} Eigen3::Eigen
                                    MatSubset::MatSubset
                                    VTK::CommonCore
                                    VTK::CommonDataModel
                                    VTK::IOXML)
    vtk_module_autoinit(TARGETS ${target}
                        MODULES VTK::CommonCore VTK::CommonDataModel VTK::IOXML)

    # Enable compiler warnings. No -Werror here: the solver headers guard
    # several locals behind assert(), so a Release (NDEBUG) build legitimately
    # reports them as unused. That is a property of the shared headers.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
       CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${target} PUBLIC -Wall -Wpedantic -Wextra)
        # Aggressive optimization tuned to the build host in Release.
        target_compile_options(${target} PUBLIC
                               $<$<CONFIG:Release>:-O3 -march=native>)
    elseif(MSVC)
        target_compile_options(${target} PUBLIC /W4)
    endif()
endforeach()
