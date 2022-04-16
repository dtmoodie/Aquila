cmake_minimum_required(VERSION 3.5)
CMAKE_POLICY(SET CMP0048 NEW)
project(Aquila
    VERSION 0.1
    LANGUAGES CXX
)
CMAKE_POLICY(SET CMP0020 NEW)
CMAKE_POLICY(SET CMP0048 NEW)
cmake_policy(SET CMP0012 NEW)

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

include(cmake/utilities.cmake)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(REVERSE CMAKE_MODULE_PATH)

IF(WIN32)
  add_definitions(-DWIN32)
  add_definitions(-D_WIN32)
  ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)
  ADD_DEFINITIONS(-D_WIN32_WINNT=0x0601)
  ADD_DEFINITIONS(-D_ENABLE_ATOMIC_ALIGNMENT_FIX)
ELSE(WIN32)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-reorder -Wno-sign-compare -Wl,--no-as-needed -O0")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -ftemplate-backtrace-limit=0")
ENDIF(WIN32)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_subdirectory("dependencies/MetaObject")
add_subdirectory("dependencies/cache-engine")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/dependencies/MetaObject/cmake")

find_package(ctext QUIET)
if(NOT TARGET ctext)
    add_subdirectory("dependencies/ctext")
else()
    # since we found a system install of ctext, we also need to look for a system install of its dependency minitensor
    find_package(minitensor REQUIRED)
endif()

find_package(OpenCV QUIET COMPONENTS
    core
)

find_package(CUDA QUIET)

IF(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib)
    SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/)
ELSE(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib64/cudadevrt)
    SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
ENDIF(WIN32)

SET(LINK_LIBS "")
ADD_DEFINITIONS(-DAquila_EXPORTS)

set(BOOST_REQUIRED_MODULES
    system
    filesystem
    thread
    date_time
    iostreams
    chrono
    regex
    log
    log_setup
    atomic
    unit_test_framework
)

set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)
find_package(Boost 1.47.0 QUIET COMPONENTS ${BOOST_REQUIRED_MODULES})

foreach(lib ${MetaObject_LIBRARIES})
    set_target_properties(${lib} PROPERTIES FOLDER MetaObject)
endforeach()

IF(NOT WIN32)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
ENDIF()

# =====================
# modules
set(aquila_export_targets "Aquila" CACHE INTERNAL "" FORCE)
set(aquila_modules "" CACHE INTERNAL "" FORCE)
set(aquila_module_includes "" CACHE INTERNAL "" FORCE)

include(cmake/aquila_declare_module.cmake)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/modules/gui)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/modules/rcc)
SUBDIRLIST(modules "${CMAKE_CURRENT_LIST_DIR}/modules")
list(REMOVE_ITEM modules gui)
list(REMOVE_ITEM modules rcc)
foreach( module ${modules})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/modules/${module})
endforeach()

# print config
message("--- Aquila config ---")
message(" Aquila modules: ${ENABLED_AQUILA_MODULES}")


# ----------------- Install
configure_file(AquilaConfig.cmake.in
    "${PROJECT_BINARY_DIR}/AquilaConfig.cmake" @ONLY
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

if(WIN32)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/Debug/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.pdb")
    install(DIRECTORY ${CMAKE_BINARY_DIR}/RelWithDebInfo/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.pdb")
endif(WIN32)

install(DIRECTORY cmake DESTINATION share/Aquila FILES_MATCHING PATTERN "*.cmake")
install(DIRECTORY cmake DESTINATION ${PROJECT_BINARY_DIR} FILES_MATCHING PATTERN "*.cmake")

install(FILES
  "${PROJECT_BINARY_DIR}/AquilaConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Aquila" COMPONENT dev
)

