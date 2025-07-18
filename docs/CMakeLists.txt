# This file creates html documentation for the project
cmake_minimum_required(VERSION 3.28)

project(MatSubsetDocs)

# Dependencies
include(FetchContent)
FetchContent_Declare(
    m.css
    GIT_REPOSITORY https://github.com/mosra/m.css.git
)
FetchContent_MakeAvailable(m.css)

# Doxygen variables
set(DOXYGEN_PROJECT_NAME MatSubset)
set(DOXYGEN_PROJECT_VERSION ${MatSubset_VERSION})
set(DOXYGEN_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../lib")
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

configure_file("${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

configure_file("${CMAKE_CURRENT_LIST_DIR}/conf.py.in" "${CMAKE_CURRENT_BINARY_DIR}/conf.py")

add_custom_target(
    GenerateDocs
    ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIRECTORY}"
    COMMAND "${m.css_SOURCE_DIR}/documentation/doxygen.py" "${CMAKE_CURRENT_BINARY_DIR}/conf.py"
    COMMAND echo "Docs written to: ${DOXYGEN_OUTPUT_DIRECTORY}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)