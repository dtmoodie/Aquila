cmake_minimum_required(VERSION 2.8)
project(Aquila)
CMAKE_POLICY(SET CMP0020 NEW)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_DEBUG_POSTFIX d)
include(cmake/utilities.cmake)
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
option(BUILD_DEPENDENCIES "Build dependencies within the AquilOS family" ON)
IF(WIN32)
  add_definitions(-DWIN32)
  add_definitions(-D_WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)
  ADD_DEFINITIONS(-D_WIN32_WINNT=0x0601)
  ADD_DEFINITIONS(-D_ENABLE_ATOMIC_ALIGNMENT_FIX)
ELSE(WIN32)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-reorder -Wno-sign-compare -Wl,--no-as-needed -O0")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -g")
  SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}" CACHE PATH "Library output path" FORCE)
  SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Executable output path" FORCE)
ENDIF(WIN32)

if(NOT BUILD_DEPENDENCIES)
    find_package(MetaObject REQUIRED)
    include_directories(${MetaObject_INCLUDE_DIRS})
    find_package(pplx REQUIRED)
else()
    ADD_SUBDIRECTORY("dependencies/MetaObject")
    ADD_SUBDIRECTORY("dependencies/pplx")
    set(MetaObject_LIBRARIES MetaObject RuntimeObjectSystem RuntimeCompiler)
    set_target_properties(pplx
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
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
    imgproc
    highgui
    cudaimgproc
    cudawarping
    cudafeatures2d
    cudaoptflow
    cudacodec
    cudabgsegm
)

find_package(Gstreamer QUIET)

find_package(CUDA REQUIRED)
  set(CUDA_ARCH_NAME "Manual")
  set(CUDA_ARCH_BIN "30 35 50 60")
  set(CUDA_ARCH_PTX "30 35 50 60")
get_filename_component(_cuda_lib_dir "${CUDA_CUDART_LIBRARY}" DIRECTORY)
link_directories(${_cuda_lib_dir})
find_package(Qt5 QUIET COMPONENTS Core Gui Widgets Network OpenGL)
cmake_policy(SET CMP0012 NEW)

find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})

IF(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib)
    SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/)
ELSE(WIN32)
    SET(CUDA_DEVRT ${CUDA_TOOLKIT_ROOT_DIR}/lib64/cudadevrt)
    SET(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
ENDIF(WIN32)

SET(LINK_LIBS "")
ADD_DEFINITIONS(-DOPENCV_FOUND)
ADD_DEFINITIONS(-DAquila_EXPORTS)
ADD_DEFINITIONS(-DRMT_DLL)
ADD_DEFINITIONS(-DBUILD_DIR=${CMAKE_BINARY_DIR})
IF(${NV_TOOLS_EXT_FOUND})
  ADD_DEFINITIONS(-DUSE_NVTX)
  list(APPEND LINK_LIBS ${NV_TOOLS_EXT_LIB})
  include_directories(${NV_TOOLS_EXT_INCLUDE_DIR})
  SET(NVTX_BIN_DIR_OPT "${NV_TOOLS_EXT_BIN_DIR}" CACHE PATH "")
  SET(NVTX_BIN_DIR_DBG "${NV_TOOLS_EXT_BIN_DIR}" CACHE PATH "")
  SET(BIN_DIRS "${BIN_DIRS};NVTX" CACHE STRING "" FORCE)
endif()
LIST(APPEND defs_ -DRMT_USE_CUDA)
LIST(APPEND defs_ -DRMT_USE_OPENGL)
IF(${Aquila_JETSON})
    LIST(APPEND defs_ "-DJETSON")
ENDIF(${Aquila_JETSON})

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

FILE(GLOB_RECURSE src "src/*.cpp" "remotery/lib/*.c")
FILE(GLOB_RECURSE hdr "remotery/lib/*.h" "include/*.hpp" "include/*.h" "src/*.hpp")
FILE(GLOB_RECURSE ext_inc "include/Aquila/rcc/external_includes")
FILE(GLOB_RECURSE knl "src/*.cu")

find_package(mxnet QUIET)
if(${MXNET_FOUND})
  INCLUDE_DIRECTORIES(${MXNET_INCLUDE_DIRS})
  add_definitions(-DHAVE_MXNET)
  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_CUDA=1)
  add_definitions(-DMSHADOW_USE_MKL=0)
  add_definitions(-DMSHADOW_USE_CUDNN=1)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/MetaObject/dependencies/cereal/include")

INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${PPLX_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/remotery/lib
)

if(Qt5_FOUND)
    include_directories(
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Gui_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS}
        ${Qt5Network_INCLUDE_DIRS}
        )
    add_definitions(-DHAVE_QT)
endif()

set_target_properties(pplx PROPERTIES FOLDER Dependencies)
set_target_properties(MetaObject PROPERTIES FOLDER Dependencies)
IF(NOT WIN32)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
ENDIF()

# Link directories
LINK_DIRECTORIES(${Parameters_LIB_DIRS_DEBUG})
LINK_DIRECTORIES(${Parameters_LIB_DIRS_RELEASE})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR_RELEASE})

# ---------------- Reporting section
if(RCC_VERBOSE_CONFIG)
message(STATUS
"================ Aquila ===============
    Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}
    OpenCV found at: ${OpenCV_CONFIG_PATH}
    NVCC Flags: ${CUDA_NVCC_FLAGS}
    Jetson: ${Aquila_JETSON}
    Bulid Type: ${CMAKE_BUILD_TYPE}

    Includes: ${Aquila_INCLUDE_DIRECTORIES}

    Libs: ${Aquila_LIBRARIES}

    Lib dirs:
    ${Aquila_LIB_DIRS_DEBUG}

    ${Aquila_LIB_DIRS_RELEASE}

    Definitions: ${defs_}


++++++++++ RCC++ Defines ++++++++++++++++++
    OpenCV include dir: ${OpenCV_INCLUDE_DIR}
    Boost library dir: ${Boost_LIBRARY_DIRS}
    OpenCV library dir: ${CV_LIB_DIR}
    Cuda include dir: ${CUDA_INCLUDE_DIRS}
    NVCC: ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc
    Cuda RT libs: ${CUDA_LIBRARIES}
    Cuda Dev RT libs: ${CUDA_DEVRT}
    Cuda root dir: ${CUDA_TOOLKIT_ROOT_DIR}
    Binary Dir: ${CMAKE_BINARY_DIR}

    Boost LIBS: ${Boost_LIBRARIES}

    PCL_found: ${PCL_FOUND}
    PCL_LIB_DIR: ${PCL_LIBRARY_DIRS}
    Eigen inc: ${EIGEN3_INCLUDE_DIR}
++++++++++++ Parameters ++++++++++++++++++++
    Includes: ${Parameters_INCLUDE}
    UI: ${Parameters_UI_qt}
")
endif()
ADD_DEFINITIONS(${defs_})


macro(GroupSources curdir)
file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)
    foreach(child ${children})
        if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
            GroupSources(${curdir}/${child})
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
        endif()
   endforeach()
endmacro()

GroupSources(include)
GroupSources(src)

# This is needed becaus cuda + cmake does not properly support target_include_directories https://gitlab.kitware.com/cmake/cmake/issues/16571
# in the version of cmake that is include with ubuntu 16.04.  Thus to make everything happy we also propagate up the needed build include dirs
# using the same variables that would be set if included from an external package, and we set them here so that cuda can be happy
set(Aquila_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include;${MetaObject_INCLUDE_DIRS};${EIGEN_INCLUDE_DIRS};${PPLX_INCLUDE_DIRS}" CACHE PATH "")
set(Aquila_LIBRARIES "Aquila;${MetaObject_LIBRARIES}" CACHE STRING "")
INCLUDE_DIRECTORIES(${MetaObject_INCLUDE_DIRS})

cuda_add_library(Aquila SHARED
    ${src}
    ${hdr}
    ${knl}
    ${RuntimeObjectSystem_SRCS}
    src/precompiled.hpp
)

set_target_properties(Aquila
    PROPERTIES
        CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(Aquila
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(Aquila
  PUBLIC
    $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIRS}>
)

ocv_add_precompiled_header_to_target(Aquila src/precompiled.hpp)
SOURCE_GROUP("Header Files\\External" FILES ${ext_inc})


TARGET_LINK_LIBRARIES(Aquila
    ${OpenCV_LIBS}
    ${CUDA_CUDART_LIBRARY}
    MetaParameters
    RuntimeObjectSystem
    RuntimeCompiler
    ${LINK_LIBS}
    MetaObject
    ${Boost_LIBRARIES}
    pplx
)
if(Qt5_FOUND)
    target_link_libraries(Aquila
        Qt5::OpenGL
    )
endif()

if(MXNET_FOUND)
  TARGET_LINK_LIBRARIES(Aquila ${MXNET_LIBRARY})
endif()

if(WIN32)
    target_link_libraries(Aquila OpenGL32)
    target_compile_options(Aquila PRIVATE "/bigobj")
else(WIN32)
  if(UNIX)
    target_link_libraries(Aquila GL GLU)
  endif(UNIX)
endif(WIN32)

ADD_DEPENDENCIES(Aquila
    RuntimeObjectSystem
    RuntimeCompiler
)

if(UNIX)
    TARGET_LINK_LIBRARIES(Aquila dl)
endif()
if(WITH_PYTHON)
  add_subdirectory("python")
endif()
# =====================
# modules
set(aquila_export_targets "Aquila" CACHE INTERNAL "" FORCE)

function(aquila_declare_module)
    set(oneValueArgs NAME CXX_STANDARD)
    set(multiValueArgs SRC DEPENDS FLAGS)
    cmake_parse_arguments(aquila_declare_module "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    if(${aquila_declare_module_SRC})
        cuda_add_library(aquila_${aquila_declare_module_NAME} ${${aquila_declare_module_SRC}})
    else()
        file(GLOB_RECURSE src "src/*.cpp" "src/*.h" "src/*.cu" "src/*.hpp")
        cuda_add_library(aquila_${aquila_declare_module_NAME} ${src})
    endif()
    if(aquila_declare_module_DEPENDS)
        rcc_link_lib(aquila_${aquila_declare_module_NAME} ${aquila_declare_module_DEPENDS})
    endif()
    target_include_directories(aquila_${aquila_declare_module_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )
    set_target_properties(aquila_${aquila_declare_module_NAME}
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if(aquila_declare_module_CXX_STANDARD)
        set_target_properties(aquila_${aquila_declare_module_NAME} PROPERTIES CXX_STANDARD ${aquila_declare_module_CXX_STANDARD})
    endif()
    if(aquila_declare_module_FLAGS)
        target_compile_options(aquila_${aquila_declare_module_NAME} PUBLIC ${aquila_declare_module_FLAGS})
    endif()

    export(TARGETS aquila_${aquila_declare_module_NAME}
        FILE "${PROJECT_BINARY_DIR}/AquilaTargets-${aquila_declare_module_NAME}.cmake"
    )
    install(TARGETS aquila_${aquila_declare_module_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(TARGETS aquila_${aquila_declare_module_NAME}
        DESTINATION lib
        EXPORT aquila_${aquila_declare_module_NAME}Targets
    )
    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )

    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
    )
install(EXPORT aquila_${aquila_declare_module_NAME}Targets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Aquila" COMPONENT dev)
endfunction()

SUBDIRLIST(modules "${CMAKE_CURRENT_LIST_DIR}/modules")
foreach( module ${modules})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/modules/${module})
endforeach()
set(PROJECT_BIN_DIRS_DEBUG "")
set(PROJECT_BIN_DIRS_RELEASE "")
set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
target_link_libraries(Aquila aquila_types aquila_serialization)
MACRO(APPEND_BIN_DIR ARG1 ARG2)
IF(EXISTS "${${ARG1}}" AND IS_DIRECTORY "${${ARG1}}")
    LIST(APPEND ${ARG2} ${${ARG1}})
ELSE()
    if(RCC_VERBOSE_CONFIG)
        MESSAGE("${ARG1} : ${${ARG1}} is not a valid directory")
    endif()
ENDIF()
ENDMACRO(APPEND_BIN_DIR)

FOREACH(dir ${BIN_DIRS})
  APPEND_BIN_DIR(${dir}_BIN_DIR_DBG PROJECT_BIN_DIRS_DEBUG)
  APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELEASE)
  APPEND_BIN_DIR(${dir}_BIN_DIR_OPT PROJECT_BIN_DIRS_RELWITHDEBINFO)
  APPEND_BIN_DIR(${dir}_BIN_DIR PROJECT_BIN_DIRS_DEBUG)
  APPEND_BIN_DIR(${dir}_BIN_DIR PROJECT_BIN_DIRS_RELEASE)
  APPEND_BIN_DIR(${dir}_BIN_DIR PROJECT_BIN_DIRS_RELWITHDEBINFO)
ENDFOREACH(dir ${BIN_DIRS})

list(APPEND PROJECT_BIN_DIRS_DEBUG ${MetaObject_BIN_DIRS})

if(MSVC)
  if(RCC_VERBOSE_CONFIG)
    message(STATUS "Aquila tests binary runtime directories:\n ${PROJECT_BIN_DIRS_DEBUG}")
  endif()
endif(MSVC)

if(Boost_UNIT_TEST_FRAMEWORK_FOUND AND BUILD_TESTS)
    include(CTest)
    enable_testing()
    SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}/tests")
    foreach(test ${tests})
        file(GLOB_RECURSE test_srcs "tests/${test}/*.cpp")
        file(GLOB_RECURSE test_knl "tests/${test}/*.cu")
        LIST(LENGTH test_knl num_knl)
        IF(${num_knl} GREATER 0)
            cuda_add_executable(${test} ${test_srcs} ${test_knl})
        else()
            add_executable(${test} ${test_srcs})
        endif()
        TARGET_LINK_LIBRARIES(${test} MetaObject Aquila MetaParameters ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
        ADD_DEPENDENCIES(${test} MetaObject Aquila)
        set_target_properties(${test} PROPERTIES FOLDER Tests/Aquila)
        add_test(NAME ${test} COMMAND $<TARGET_FILE:${test}>)
        if(MSVC)
          CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/Aquila/${test}.vcxproj.user @ONLY)
          CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/${test}.vcxproj.user @ONLY)
        endif()
    endforeach()
endif()
message(STATUS
    "-- Aquila config\n"
    "  MetaObject\t\t${MetaObject_VERSION}\n"
    "  MxNet\t\t${MXNET_FOUND}\n"
    "  Eigen3\t\t${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}\n"
    "  Tests\t\t${Boost_UNIT_TEST_FRAMEWORK_FOUND}"
)

# ----------------- Install
if(Qt5_FOUND)
    set(AQ_QT ON)
else()
    set(AQ_QT OFF)
endif()

configure_file(AquilaConfig.cmake.in
    "${PROJECT_BINARY_DIR}/AquilaConfig.cmake" @ONLY
)

export(TARGETS ${aquila_export_targets}
    FILE "${PROJECT_BINARY_DIR}/AquilaTargets.cmake"
)

export(PACKAGE Aquila)

file(COPY "cmake" DESTINATION ${PROJECT_BINARY_DIR})

install(TARGETS ${aquila_export_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS ${aquila_export_targets}
    DESTINATION lib
    EXPORT AquilaTargets
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

install(EXPORT AquilaTargets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Aquila" COMPONENT dev)

