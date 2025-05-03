# Check existence of install manifest
if(NOT EXISTS ${CMAKE_BINARY_DIR}/install_manifest.txt)
    message(FATAL_ERROR "Cannot find install manifest: ${CMAKE_BINARY_DIR}/install_manifest.txt")
endif()

# Read all filenames from install manifest and count them
file(READ "${CMAKE_BINARY_DIR}/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
list(LENGTH files total_files)  # Get total file count
set(current_file 0)

# Delete all files listed in install manifest
foreach(file ${files})
    if(EXISTS "$ENV{DESTDIR}${file}")
        math(EXPR current_file "${current_file} + 1")
        message(STATUS "[${current_file}/${total_files}] Removing: $ENV{DESTDIR}${file}")
        file(REMOVE "$ENV{DESTDIR}${file}")
        if(EXISTS "$ENV{DESTDIR}${file}")
            message(FATAL_ERROR "Failed to remove: $ENV{DESTDIR}${file}")
        endif()
    else()
        message(STATUS "File does not exist: $ENV{DESTDIR}${file}")
    endif()
endforeach()