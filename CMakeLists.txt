cmake_minimum_required(VERSION 2.8)
CMAKE_POLICY(SET CMP0048 NEW)
project(Aquila
    VERSION 0.1
    LANGUAGES CXX)
CMAKE_POLICY(SET CMP0020 NEW)
CMAKE_POLICY(SET CMP0048 NEW)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_DEBUG_POSTFIX d)
include(cmake/utilities.cmake)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(REVERSE CMAKE_MODULE_PATH)

option(BUILD_DEPENDENCIES "Build dependencies within the AquilOS family" ON)

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
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 11)  
ENDIF(WIN32)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT BUILD_DEPENDENCIES)
    find_package(MetaObject REQUIRED)
    include_directories(${MetaObject_INCLUDE_DIRS})
else()
    add_subdirectory("dependencies/MetaObject")
    add_subdirectory("dependencies/cache-engine")
    LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/dependencies/MetaObject/cmake")
    set_target_properties(RuntimeCompiler
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_target_properties(RuntimeObjectSystem
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

set(RCC_PERMODULE_INTERFACE_SOURCE
    ${RCC_PERMODULE_INTERFACE_SOURCE}
    CACHE PATH "" FORCE
)

include(cmake/configure_bin_dirs.cmake)
find_package(OpenCV 3.0 QUIET COMPONENTS
    core
)

find_package(Gstreamer QUIET)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
find_package(CUDA QUIET)
if(CUDA_FOUND)
  set(CUDA_ARCH_NAME "Manual")
  set(CUDA_ARCH_BIN "30 35 50 60")
  set(CUDA_ARCH_PTX "30 35 50 60")
  get_filename_component(_cuda_lib_dir "${CUDA_CUDART_LIBRARY}" DIRECTORY)
  link_directories(${_cuda_lib_dir})
endif(CUDA_FOUND)

find_package(Qt5 QUIET COMPONENTS Core Gui Widgets Network OpenGL)
cmake_policy(SET CMP0012 NEW)

IF(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib)
    SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/)
ELSE(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib64/cudadevrt)
    SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
ENDIF(WIN32)

SET(LINK_LIBS "")
ADD_DEFINITIONS(-DAquila_EXPORTS)
IF(${NV_TOOLS_EXT_FOUND})
  ADD_DEFINITIONS(-DUSE_NVTX)
  list(APPEND LINK_LIBS ${NV_TOOLS_EXT_LIB})
  include_directories(${NV_TOOLS_EXT_INCLUDE_DIR})
  SET(NVTX_BIN_DIR_OPT "${NV_TOOLS_EXT_BIN_DIR}" CACHE PATH "")
  SET(NVTX_BIN_DIR_DBG "${NV_TOOLS_EXT_BIN_DIR}" CACHE PATH "")
  SET(BIN_DIRS "${BIN_DIRS};NVTX" CACHE STRING "" FORCE)
endif()

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

# Link directories
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR_RELEASE})

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

# This is needed becaus cuda + cmake does not properly support target_include_directories https://gitlab.kitware.com/cmake/cmake/issues/16571
# in the version of cmake that is include with ubuntu 16.04.  Thus to make everything happy we also propagate up the needed build include dirs
# using the same variables that would be set if included from an external package, and we set them here so that cuda can be happy
set(Aquila_INCLUDE_DIRS "${aquila_module_includes};${CMAKE_CURRENT_LIST_DIR}/include;${MetaObject_INCLUDE_DIRS};${EIGEN_INCLUDE_DIRS}" CACHE PATH "" FORCE)
set(Aquila_LIBRARIES "${aquila_modules};${MetaObject_LIBRARIES}" CACHE INTERNAL "" FORCE)

# ----------------- Install
if(Qt5_FOUND)
    set(AQ_QT ON)
else()
    set(AQ_QT OFF)
endif()

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

