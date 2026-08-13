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
add_executable(${PROJECT_NAME}
               "${CMAKE_CURRENT_SOURCE_DIR}/src/main_acoustic.cpp")
set_target_properties(${PROJECT_NAME} PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AcousticWave")

# The 3D Allen-Cahn demo (AllenCahn), sharing the same include/ headers as the
# acoustic demos.
add_executable(AllenCahn "${CMAKE_CURRENT_SOURCE_DIR}/src/main_allen_cahn.cpp")
set_target_properties(AllenCahn PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AllenCahn")

# The selector-comparison driver of the volume-sampling experiment: the same
# Allen-Cahn discretization, run once per column-subset-selection algorithm and
# scored against a dense reference. Config-driven (see
# src/main_allen_cahn_tester.cpp), and it builds its selectors with the
# Tester's own factory, hence the matrix_tools include directory below.
add_executable(AllenCahnTester
               "${CMAKE_CURRENT_SOURCE_DIR}/src/main_allen_cahn_tester.cpp")
set_target_properties(AllenCahnTester PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AllenCahnTester")

# The same comparison on the acoustic wave equation: tolerance-driven rather
# than fixed-rank, since a wavefront's rank grows as it propagates, so it
# reports the rank alongside the error (see src/main_acoustic_tester.cpp).
add_executable(AcousticTester
               "${CMAKE_CURRENT_SOURCE_DIR}/src/main_acoustic_tester.cpp")
set_target_properties(AcousticTester PROPERTIES
                      CXX_STANDARD 20
                      OUTPUT_NAME "AcousticTester")

# nlohmann_json, for the config and the shared SelectorFactory.
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

foreach(tester AllenCahnTester AcousticTester)
    target_include_directories(${tester} PRIVATE
                               "${CMAKE_CURRENT_SOURCE_DIR}/../matrix_tools/include")
    target_link_libraries(${tester} nlohmann_json::nlohmann_json)
endforeach()

foreach(target ${PROJECT_NAME} AllenCahn AllenCahnTester AcousticTester)
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
