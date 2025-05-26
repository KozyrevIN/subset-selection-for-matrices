# This is a master script that can install MatSubset library and build examples, tests, documentation and benchmarking code
# Each component is a separate, self-sustained project with unique dependencies

cmake_minimum_required(VERSION 3.28)

project(SuperBuild)
include(ExternalProject)

# Options
option(BUILD_LIB "Build the MatSubset library" ON)
option(INSTALL_LIB "Install the MatSubset library" ON)
option(LINK_AGAINST_BUILD_TREE "Link components against build tree" OFF)
option(BUILD_EXAMPLES "Build MatSubset demo executables" OFF)
option(BUILD_TESTS "Build MatSubset test executables" OFF)
option(BUILD_DOCS "Build MatSubset documentation" OFF)
option(BUILD_BENCH "Build MatSubset benchmarking code" OFF)

# Dependency checks
if(INSTALL_LIB AND NOT BUILD_LIB)
    message(FATAL_ERROR "INSTALL_LIB requires BUILD_LIB=ON")
endif()
if(LINK_AGAINST_BUILD_TREE AND NOT BUILD_LIB)
    message(FATAL_ERROR "LINK_AGAINST_BUILD_TREE requires BUILD_LIB=ON")
endif()

# Build/install the MatSubset library
if (BUILD_LIB)
    # Set install_command conditionally
    if (INSTALL_LIB)
        set(install_command cmake --install .)
    else()
        set(install_command cmake -E echo "INSTALL_LIB option is disabled, skipping install step")
    endif()

    ExternalProject_Add(
        MatSubset
        SOURCE_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        BINARY_DIR       "${CMAKE_CURRENT_BINARY_DIR}/lib"
        CMAKE_ARGS       ${CMAKE_ARGS}
        BUILD_COMMAND    cmake --build .
        INSTALL_COMMAND  ${install_command}
    )
    set(components_dependencies MatSubset)
else()
    set(components_dependencies "")
endif()

# Helper function to add components (examples, tests, etc.)
function(add_component name source_dir build_dir)
    # --- Parse optional arguments ---
    set(options "")
    set(oneValueArgs COMPONENT_TARGET) # Changed from SUB_TARGET_NAME
    set(multiValueArgs "")
    cmake_parse_arguments(AC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # AC_COMPONENT_TARGET will be set if COMPONENT_TARGET was passed

    string(TOUPPER "${name}" name_upper)
    if (${BUILD_${name_upper}})
        set(user_cmake_prefix_path "")
        foreach(arg IN LISTS CMAKE_ARGS)
            if (arg MATCHES "-DCMAKE_PREFIX_PATH=(.*)")
                set(user_cmake_prefix_path "${CMAKE_MATCH_1}")
            endif()
        endforeach()

        set(component_specific_cmake_args "")
        if (LINK_AGAINST_BUILD_TREE)
            set(lib_build_dir "${CMAKE_CURRENT_BINARY_DIR}/lib")
            set(effective_prefix_path "")
            if (user_cmake_prefix_path)
                set(effective_prefix_path "${lib_build_dir};${user_cmake_prefix_path}")
            else()
                set(effective_prefix_path "${lib_build_dir}")
            endif()
            list(APPEND component_specific_cmake_args "-DCMAKE_PREFIX_PATH=${effective_prefix_path}")
        elseif(user_cmake_prefix_path)
            list(APPEND component_specific_cmake_args "-DCMAKE_PREFIX_PATH=${user_cmake_prefix_path}")
        endif()

        set(ep_build_command "${CMAKE_COMMAND}" --build .)
        if (AC_COMPONENT_TARGET AND NOT AC_COMPONENT_TARGET STREQUAL "" AND NOT AC_COMPONENT_TARGET STREQUAL "all")
            list(APPEND ep_build_command --target "${AC_COMPONENT_TARGET}")
        endif()
        
        set(final_cmake_args ${CMAKE_ARGS} ${component_specific_cmake_args})

        ExternalProject_Add(
            MatSubset${name}
            SOURCE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}"
            BINARY_DIR      "${CMAKE_CURRENT_BINARY_DIR}/${build_dir}"
            CMAKE_ARGS      ${final_cmake_args}
            BUILD_COMMAND   ${ep_build_command}
            INSTALL_COMMAND ""
            DEPENDS         ${components_dependencies}
        )
    endif()
endfunction()

# Add components (examples, tests, etc.)
add_component(Examples examples examples)
add_component(Tests tests tests)
add_component(Docs docs docs COMPONENT_TARGET GenerateDocs)
#add_component(Bench bench bench)