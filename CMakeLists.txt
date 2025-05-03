# This is a master script that can install MatSubset library and build demos, tests, documentation and benchmarking code
# Each component is a separate, self-sustained project with unique dependencies

cmake_minimum_required(VERSION 3.28)

project(SuperBuild)
include(ExternalProject)

# Options
option(BUILD_LIB "Build the MatSubset library" ON)
option(INSTALL_LIB "Install the MatSubset library" ON)
option(LINK_AGAINST_BUILD_TREE "Link components against build tree" OFF)
option(BUILD_DEMOS "Build MatSubset demo executables" OFF)
option(BUILD_TESTS "Build MatSubset test executables" OFF)
option(BUILD_DOCS "Build MatSubset documentation" OFF)
option(BUILD_BENCH "Build MatSubset benchmarks" OFF)

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

# Helper function to add components (demos, tests, etc.)
function(add_component name source_dir build_dir)
    string(TOUPPER "${name}" name_upper)
    if (${BUILD_${name_upper}})
        # Extract user's CMAKE_PREFIX_PATH (if any)
        set(user_cmake_prefix_path "")
        foreach(arg IN LISTS CMAKE_ARGS)
            if (arg MATCHES "-DCMAKE_PREFIX_PATH=(.*)")
                set(user_cmake_prefix_path "${CMAKE_MATCH_1}")
            endif()
        endforeach()

        # If linking against build tree, prepend library's build dir
        set(cmake_prefix_path_args "")
        if (LINK_AGAINST_BUILD_TREE)
            set(lib_build_dir "${CMAKE_CURRENT_BINARY_DIR}/lib")
            if (user_cmake_prefix_path)
                set(cmake_prefix_path "${lib_build_dir};${user_cmake_prefix_path}")
            else()
                set(cmake_prefix_path "${lib_build_dir}")
            endif()
            list(APPEND cmake_prefix_path_args -DCMAKE_PREFIX_PATH=${cmake_prefix_path})
        endif()

        ExternalProject_Add(
        MatSubset${name}
        SOURCE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}"
        BINARY_DIR      "${CMAKE_CURRENT_BINARY_DIR}/${build_dir}"
        CMAKE_ARGS
            ${CMAKE_ARGS}
            ${cmake_prefix_path_args}  # Includes build dir + user paths
        BUILD_COMMAND   cmake --build .
        INSTALL_COMMAND ""
        DEPENDS         ${components_dependencies}
        )
    endif()
endfunction()

# Add components (demos, tests, etc.)
add_component(Demos demos demos)
#add_component(Tests tests tests)
#add_component(Docs docs docs)
#add_component(Bench bench bench)