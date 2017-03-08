cmake_minimum_required(VERSION 2.8)
PROJECT(MetaObject)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
CMAKE_POLICY(SET CMP0020 NEW)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CMakeMacros.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(link_libs "")

set(WITH_QT "TRUE" CACHE BOOL "Enable QT support")
set(WITH_PYTHON "TRUE" CACHE BOOL "Enable Python bindings")
set(WITH_OPENCV "TRUE" CACHE BOOL "Enable OpenCV types")
set(WITH_CUDA "TRUE" CACHE BOOL "Enable CUDA recompilation")
set(WITH_WT "TRUE" CACHE BOOL "Enable WT support")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(cmake/OpenCVPCHSupport.cmake)

if(WITH_PYTHON)
  find_package(PythonLibs QUIET)
  find_package(PythonInterp QUIET)
  if(PythonLibs_FOUND AND PythonInterp_FOUND)
    find_package(Boost COMPONENTS python)
    if(Boost_FOUND)
      list(APPEND link_libs "${PYTHON_LIBRAY}")
      get_filename_component(_python_lib_dir "${PYTHON_DEBUG_LIBRARY}" DIRECTORY)
      link_directories(${_python_lib_dir})
      include_directories(${PYTHON_INCLUDE_DIR})
      add_definitions(-DHAVE_BOOST_PYTHON)
    endif()
  endif()
endif()

if(WIN32)
  SET(CMAKE_CXX_FLAGS "/EHsc /FC /Zi /MP /Zm512 /bigobj /FS" CACHE STRING "")
  SET(CMAKE_CXX_FLAGS_RELEASE "/Zo /d2Zi+ /Oy-" CACHE STRING "")
else(WIN32)
  SET(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "" FORCE)
  list(APPEND link_libs pthread)
  add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
endif(WIN32)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*)")
  set(AARCH64 1)
  list(APPEND link_libs atomic)
else()
  set(AARCH64 0)
endif()
set(CMAKE_DEBUG_POSTFIX "d")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")
file(GLOB_RECURSE src "src/*.cpp")
file(GLOB_RECURSE knl "src/*.cu")
file(GLOB_RECURSE hdr "include/*.h" "include/*.hpp")
include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

# ---------------------- External Dependencies ---------------------------------

# ---------------------- Boost ---------------------------------
set(Boost_required_components system thread log log_setup unit_test_framework)
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
ADD_DEFINITIONS(-DBOOST_ALL_DYN_LINK)
find_package(Boost REQUIRED COMPONENTS ${Boost_required_components})
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIR_DEBUG})
  link_directories(${Boost_LIBRARY_DIR})
  list(APPEND link_libs ${Boost_LIBRARIES})
  list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR})
  list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${Boost_LIBRARY_DIR_DEBUG})
  list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${Boost_LIBRARY_DIR_RELEASE})
  list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${Boost_LIBRARY_DIR_RELEASE})
  set(BIN_DIRS "${BIN_DIRS};Boost" CACHE STRING "" FORCE)


# ---------------------- cereal ---------------------------------
set(cereal_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal/include" CACHE PATH "")
find_package(cereal REQUIRED)
if(cereal_FOUND)
  include_directories(${cereal_INCLUDE_DIRS})
endif()
set(PROJECT_BIN_DIRS_RELWITHDEBINFO "")
set(PROJECT_BIN_DIRS_RELEASE "")
set(PROJECT_BIN_DIRS_DEBUG "")

# CUDA
if(WITH_CUDA)
  FIND_PACKAGE(CUDA REQUIRED)
  include("cmake/Cuda.cmake")
  IF(${CUDA_FOUND})
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DHAVE_CUDA)
    list(APPEND PROJECT_BIN_DIRS_DEBUG_ ${CUDA_BIN_DIR})
    list(APPEND PROJECT_BIN_DIRS_RELEASE_ ${CUDA_BIN_DIR})
    list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO_ ${CUDA_BIN_DIR})
    if(NOT WIN32)
      #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--std=c++11")
    endif(NOT WIN32)
  ENDIF(${CUDA_FOUND})
endif(WITH_CUDA)

# ---------------------- rcc ---------------------------------  
SET(IObject_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(RCC_INCLUDE_DEPENDENCIES "${Boost_INCLUDE_DIRS};${CMAKE_CURRENT_LIST_DIR}/include;${cereal_INCLUDE_DIRS}")
set(RCC_LIBRARY_DIRS_DEBUG "${Boost_LIBRARY_DIR};${Boost_LIBRARY_DIR_DEBUG};${CMAKE_BINARY_DIR}/Debug")
set(RCC_LIBRARY_DIRS_RELWITHDEBINFO "${Boost_LIBRARY_DIR};${Boost_LIBRARY_DIR_DEBUG};${CMAKE_BINARY_DIR}/RelWithDebInfo")
if(MSVC)
  set(RCC_COMPILE_FLAGS "/DBOOST_ALL_DYN_LINK")
else()
  set(RCC_COMPILE_FLAGS "-DBOOST_ALL_DYN_LINK")
endif()
ADD_SUBDIRECTORY("dependencies/rcc")
include_directories(${RCC_INCLUDE_DIRS})
set_target_properties(RuntimeCompiler PROPERTIES FOLDER Dependencies)
set_target_properties(RuntimeObjectSystem PROPERTIES FOLDER Dependencies)
SET(RCC_INCLUDE_DIRS "${RCC_INCLUDE_DIRS}" PARENT_SCOPE)
set(RCC_PERMODULE_INTERFACE_SOURCE ${RCC_PERMODULE_INTERFACE_SOURCE} PARENT_SCOPE)

set(MO_PROFILING_MODE "BUILTIN" CACHE STRING "Set MetaObject profiling mode")
set_property(CACHE MO_PROFILING_MODE PROPERTY STRINGS BUILTIN DYNAMIC NONE)
include_directories("dependencies/Remotery/lib")
if(${MO_PROFILING_MODE} STREQUAL "BUILTIN")
  add_definitions(-DRMT_BUILTIN)
  list(APPEND link_libs Remotery)
endif()
if(${MO_PROFILING_MODE} STREQUAL "NONE")
  add_definitions(-DPROFILING_NONE)
endif()

add_definitions(-DRMT_USE_CUDA)

# ---------------------- Optional dependencies ------------------
# Qt5
if(WITH_QT)
  find_package(Qt5 QUIET COMPONENTS Core Widgets Gui)
  if(Qt5_FOUND)
    set(AUTOMOC OFF)
    add_definitions(-DHAVE_QT5)
    QT5_WRAP_CPP(MOC "${CMAKE_CURRENT_LIST_DIR}/include/MetaObject/Parameters/UI/Qt/SignalProxy.hpp")
    list(APPEND src ${MOC})
    INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS})
    list(APPEND link_libs Qt5::Core Qt5::Widgets Qt5::Gui)
    get_target_property(qt5_core_bin_ Qt5::Core IMPORTED_LOCATION_DEBUG)
    get_filename_component(qt5_bin_dir_ "${qt5_core_bin_}" DIRECTORY)
    list(APPEND PROJECT_BIN_DIRS_DEBUG ${qt5_bin_dir_})
    list(APPEND PROJECT_BIN_DIRS_RELEASE ${qt5_bin_dir_})
    list(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${qt5_bin_dir_})
    set(Qt5_BIN_DIR_OPT ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(Qt5_BIN_DIR ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(Qt5_BIN_DIR_DBG ${qt5_bin_dir_} CACHE PATH "" FORCE)
    set(BIN_DIRS "${BIN_DIRS};Qt5" CACHE STRING "" FORCE)
    rcc_find_path(Qt5_PLUGIN_PATH qwindows.dll PATHS ${Qt5_BIN_DIR}/../plugins/platforms)
  endif()
endif()

# WT
if(WITH_WT)
find_package(Wt QUIET)
  if(Wt_FOUND)
    add_definitions(-DHAVE_WT)
    include_directories(${Wt_INCLUDE_DIR})
    list(APPEND link_libs
      debug;${Wt_HTTP_DEBUG_LIBRARY}
      debug;${Wt_DEBUG_LIBRARY}
      optimized;${Wt_LIBRARY}
      optimized;${Wt_HTTP_LIBRARY}
    )

    find_path(Wt_BIN_DIR
      NAMES wt.dll wtd.dll
      PATHS ${Wt_DIR} "${Wt_INCLUDE_DIR}/../"
      ENV ${Wt_DIR}
      PATH PATH_SUFFIXES bin)

    if(Wt_BIN_DIR)
      set(Wt_BIN_DIR_DBG ${Wt_BIN_DIR} CACHE PATH "" FORCE)
      set(Wt_BIN_DIR_OPT ${Wt_BIN_DIR} CACHE PATH "" FORCE)
      set(BIN_DIRS "${BIN_DIRS};Wt" CACHE STRING "" FORCE)
    endif(Wt_BIN_DIR)
  endif(Wt_FOUND)
endif(WITH_WT)
# OpenCV
if(WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core)
  if(OpenCV_FOUND)
    add_definitions(-DHAVE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
	set(BIN_DIRS "${BIN_DIRS};OpenCV" CACHE STRING "" FORCE)
  endif()
endif(WITH_OPENCV)

# ZeroMQ
find_package(ZeroMQ QUIET)
if(${ZeroMQ_FOUND})
  include_directories(${ZeroMQ_INCLUDE_DIR})
  include_directories("${CMAKE_CURRENT_LIST_DIR}/dependencies/cppzmq") # cpp bindings
  list(APPEND link_libs "optimized;${ZeroMQ_LIBRARY_RELEASE};debug;${ZeroMQ_LIBRARY_DEBUG}")
  add_definitions(-DHAVE_ZEROMQ)
endif(${ZeroMQ_FOUND})



add_subdirectory("dependencies/Remotery")
set_target_properties(Remotery PROPERTIES FOLDER Dependencies)
GroupSources(include)
GroupSources(src)
cuda_add_library(MetaObject SHARED ${src} ${hdr} ${knl} src/precompiled.hpp)
target_link_libraries(MetaObject
  RuntimeObjectSystem
  RuntimeCompiler
  ${link_libs}
  ${OpenCV_LIBS}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUDA_LIBRARY}
)
ocv_add_precompiled_header_to_target(MetaObject src/precompiled.hpp)

if(RCC_VERBOSE_CONFIG)
  message(STATUS "MetaObject compile flags\n"
                 "  CXX flags: ${CMAKE_CXX_FLAGS}\n"
                 "  CXX flags Release: ${CMAKE_CXX_FLAGS_RELEASE}\n"
                 "  CXX flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}\n"
                 "  CUDA flags: ${CUDA_NVCC_FLAGS}\n"
                 "  CUDA flags Release: ${CUDA_NVCC_FLAGS_RELEASE}\n"
                 "  CUDA flags Debug: ${CUDA_NVCC_FLAGS_DEBUG}\n")
endif()


# -------------------- instantiations --------------------------
file(GLOB_RECURSE instantiations_src "instantiations/*.cpp")
include_directories("instantiations")
add_library(instantiations SHARED ${instantiations_src})
set_target_properties(instantiations PROPERTIES FOLDER Dependencies)
target_link_libraries(instantiations MetaObject)
if(Qt5_FOUND)
  target_link_libraries(instantiations Qt5::Core Qt5::Gui Qt5::Widgets)
endif()
if(OpenCV_FOUND AND WITH_OPENCV)
  target_link_libraries(instantiations ${OpenCV_LIBS})
endif()

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

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
ENDFOREACH(dir ${BIN_DIRS})

IF(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")
 LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR_DEBUG})
ELSE(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")
  IF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${BOOST_LIBRARYDIR})
  ELSE(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    MESSAGE("Could not find boost binary dir in Boost_LIBRARY_DIR_DEBUG or BOOST_LIBRARYDIR")
  ENDIF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
ENDIF(Boost_LIBRARY_DIR_DEBUG AND EXISTS "${Boost_LIBRARY_DIR_DEBUG}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_DEBUG}")

IF(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
  LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR_RELEASE})
  LIST(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${Boost_LIBRARY_DIR_RELEASE})
ELSE(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
  IF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${BOOST_LIBRARYDIR})
    LIST(APPEND PROJECT_BIN_DIRS_RELWITHDEBINFO ${BOOST_LIBRARYDIR})
  ELSE(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
    IF(Boost_LIBRARY_DIR)
      LIST(APPEND PROJECT_BIN_DIRS_RELEASE ${Boost_LIBRARY_DIR})
      LIST(APPEND PROJECT_BIN_DIRS_DEBUG ${Boost_LIBRARY_DIR})
    ELSE()
      MESSAGE("Could not find boost binary dir in Boost_LIBRARY_DIR_DEBUG or BOOST_LIBRARYDIR")
    ENDIF()
  ENDIF(BOOST_LIBRARYDIR AND EXISTS "${BOOST_LIBRARYDIR}" AND IS_DIRECTORY "${BOOST_LIBRARYDIR}")
ENDIF(Boost_LIBRARY_DIR_RELEASE AND EXISTS "${Boost_LIBRARY_DIR_RELEASE}" AND IS_DIRECTORY "${Boost_LIBRARY_DIR_RELEASE}")
# ----------------------- tests -----------------------------------
if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
  include(CTest)
  enable_testing()
  SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}/tests")
  foreach(test ${tests})
    file(GLOB_RECURSE test_srcs "tests/${test}/*.cpp")
    IF(CUDA_FOUND)
      file(GLOB_RECURSE test_knl "tests/${test}/*.cu")
    ENDIF()
    LIST(LENGTH test_knl num_knl)
    IF(${num_knl} GREATER 0)
      cuda_add_executable(${test} ${test_srcs} ${test_knl})
    else()
      add_executable(${test} ${test_srcs})
    endif()
    TARGET_LINK_LIBRARIES(${test} MetaObject ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${link_libs} instantiations)
    if(OpenCV_FOUND)
      TARGET_LINK_LIBRARIES(${test} ${OpenCV_LIBS})
    endif()
    ADD_DEPENDENCIES(${test} MetaObject)
    set_target_properties(${test} PROPERTIES FOLDER Tests/MetaObject)
    add_test(${test} ${test})
    if(MSVC)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/${test}.vcxproj.user @ONLY)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_CURRENT_BINARY_DIR}/${test}.vcxproj.user @ONLY)
    endif()
  endforeach()
endif()

# ----------------------- examples ---------------------------
ADD_SUBDIRECTORY("examples")
