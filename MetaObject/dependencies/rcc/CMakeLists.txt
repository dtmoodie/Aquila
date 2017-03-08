# How to include RCC in your own project
# 
#  # Get all include directories for this project and pass them to RCC.
#  # Alternatively you can only include the directories that you know you need in runtime compilable classes 
#  GET_PROPERTY(RCC_INCLUDE_DEPENDENCIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES) 
#  
#  # Set the include directory for RCC to find your definition of an IObject
#  SET(IObject_INCLUDE ".....")
#  
#  # Semi-colon separated list of directories for searching for libraries
#  SET(RCC_LIBRARY_DIRS "..... linked library dirs ..........") 
#  Add this to your project to gain access to the object factory system
#  ADD_SUBDIRECTORY("RCC")
#  INCLUDE_DIRECTORIES(${RCC_INCLUDE_DIRS})
# 
#
# ================ C++ code example ====================================
# 
# #include "RuntimeObjectSystem.h"
#
# int main()
# {
#   RuntimeObjectSystem objectFactory;
#   objectFactory.Initialise(new CompileLogger(), nullptr);
#   auto constructor = objectFactory.GetObjectFactorySystem()->GetConstructor("MyObject");
#   MyObjectInterface* myObject = nullptr;
#   if(constructor)
#   {
#      IObject* obj = constructor->construct();
#      if(obj && obj->GetInterface(MyObjectInterfaceID))
#      {
#         myObject = static_cast<MyObjectInterface*>(obj);
#      }
#   }
#
#   boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
#    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
#    boost::posix_time::time_duration delta = currentTime - prevTime;
#   while(true)
#   {
#      myObject->RunVirtualFunction();
#      prevTime = boost::posix_time::microsec_clock::universal_time();
#      currentTime = boost::posix_time::microsec_clock::universal_time();
#      delta = currentTime - prevTime;
#      if(delta.total_milliseconds() > 10)
#         continue;
#      prevTime = currentTime;
#    if( objectFactory.GetIsCompiledComplete())
#    {
#        objectFactory.LoadCompiledModule();
#    }
#    if(objectFactory.GetIsCompiling())
#    {
#       return true;
#    }else
#    {
#        objectFactory.GetFileChangeNotifier()->Update(float(delta.total_milliseconds())/1000.0);
#    }
#   }
#   return 0;
# }

cmake_minimum_required(VERSION 2.8)

project(RCC)

SET(RCC_VERBOSE_CONFIG FALSE CACHE BOOL "Set to true to output verbose debugging information in RCC macros")


macro(rcc_find_library var_name)
    find_library(${var_name} ${ARGN})
    if(RCC_VERBOSE_CONFIG)
        if(${var_name})
            message(STATUS "${var_name} found at ${${var_name}}")
        else(${var_name})
            message(STATUS "${var_name} not found")
        endif(${var_name})
    endif(RCC_VERBOSE_CONFIG)
endmacro(rcc_find_library var_name)

macro(rcc_find_path var_name)
  find_path(${var_name} ${ARGN})
  if(RCC_VERBOSE_CONFIG)
    if(${var_name})
      message(STATUS "${var_name} found at ${${var_name}}")
    else(${var_name})
      message(STATUS "${var_name} not found")
    endif(${var_name})
  endif(RCC_VERBOSE_CONFIG)
endmacro(rcc_find_path var_name)

macro(rcc_strip_extension input output)
  get_filename_component(tmp ${${input}} NAME)
  string(FIND ${tmp} ".so" pos REVERSE)
  string(LENGTH ${tmp} len)
  if(${pos} EQUAL "-1")
    string(FIND ${tmp} ".a" pos REVERSE)
    if(NOT ${pos} EQUAL "-1")
        string(SUBSTRING "${tmp}" 0 ${pos} ${output})
    else()
      string(FIND ${tmp} ".lib" pos REVERSE)
      if(NOT ${pos} EQUAL "-1")
        if(RCC_VERBOSE_CONFIG)
          message(STATUS "Unable to find .so .a or .lib extension in ${${input}}")
        endif()
      else()
        string(SUBSTRING "${tmp}" 0 ${pos} ${output})
      endif()
    endif()
  else()
    string(SUBSTRING "${tmp}" 0 ${pos} ${output})
  endif()
endmacro()

# This macro is used for linking libs to a target, but also adding the correct directories to the RCC config
MACRO(RCC_LINK_LIB TARGET)
  SET(PREFIX "")
  set(LINK_LIBS_RELEASE "")
  set(LINK_LIBS_DEBUG "")
  if(RCC_VERBOSE_CONFIG)
    message(STATUS "=================================================================== 
             RCC config information for ${TARGET}")
  endif(RCC_VERBOSE_CONFIG)
  FOREACH(lib ${ARGN})
    SET(SKIP "0")
    SET(IS_DEBUG "2") # default to a release library
    IF(${lib} STREQUAL "optimized")
      SET(PREFIX "optimized;")
      SET(SKIP "1")
      SET(IS_DEBUG "0")
    ENDIF(${lib} STREQUAL "optimized")
  
    IF(${lib} STREQUAL "debug")
      SET(PREFIX "debug;")
      SET(SKIP "1")
      SET(IS_DEBUG "1")
    ENDIF(${lib} STREQUAL "debug")

    IF(${SKIP} STREQUAL "0")
      IF(TARGET ${lib})
        get_target_property(rel ${lib} IMPORTED_IMPLIB_RELEASE)
        get_target_property(_rel ${lib} IMPORTED_LOCATION_RELEASE)
        get_target_property(deb ${lib} IMPORTED_IMPLIB_DEBUG)
        get_target_property(_deb ${lib} IMPORTED_LOCATION_DEBUG)

        if(NOT rel AND _rel)
          set(rel ${_rel})
        endif(NOT rel AND _rel)
        if(NOT deb AND _deb)
          set(deb ${_deb})
        endif(NOT deb AND _deb)

        if(NOT rel AND deb)
          set(rel ${deb})
        endif(NOT rel AND deb)

        if(NOT deb AND rel)
          set(deb ${rel})
        endif(NOT deb AND rel)

        IF(NOT rel AND NOT deb AND RCC_VERBOSE_CONFIG)
          message("Target ${lib} does not contain IMPORTED_IMPLIB_RELEASE or IMPORTED_IMPLIB_DEBUG")
        endif(NOT rel AND NOT deb AND RCC_VERBOSE_CONFIG)

        GET_FILENAME_COMPONENT(rel_ ${rel} DIRECTORY)
        GET_FILENAME_COMPONENT(deb_ ${deb} DIRECTORY)
        get_filename_component(rel__ ${rel} NAME)
        get_filename_component(deb__ ${deb} NAME)

        LIST(APPEND LINK_DIRS_DEBUG "${deb_}")
        LIST(APPEND LINK_DIRS_RELEASE "${rel_}")
        LIST(APPEND LINK_FILE_DEBUG "${deb__}")
        LIST(APPEND LINK_FILE_RELEASE "${rel__}")

        if(RCC_VERBOSE_CONFIG AND deb_)
          message(STATUS "  Link dir: [${lib}] (debug): ${deb_} - ${deb__}")
        endif(RCC_VERBOSE_CONFIG AND deb_)
        if(RCC_VERBOSE_CONFIG AND rel_)
          message(STATUS "  Link dir: [${lib}] (release): ${rel_} - ${rel__}")
        endif(RCC_VERBOSE_CONFIG AND rel_)

        if(rel__)
          #get_filename_component(rel__ "${rel__}" NAME_WE)
          rcc_strip_extension(rel__ rel__)
          string(SUBSTRING "${rel__}" 0 3 sub)
          if(${sub} STREQUAL lib)
            string(SUBSTRING "${rel__}" 3 -1 rel__)
          endif()
        #STRING(REPLACE "lib" "" rel__ "${rel__}")

          LIST(APPEND LINK_LIBS_RELEASE ${rel__})
        endif()

        if(deb__)
          #get_filename_component(deb__ "${deb__}" NAME_WE)
          rcc_strip_extension(deb__ deb__)
          string(SUBSTRING "${rel__}" 0 3 sub)
          if(${sub} STREQUAL lib)
            string(SUBSTRING "${rel__}" 3 -1 rel__)
          endif()

          #STRING(REPLACE "lib" "" deb__ "${deb__}")
          LIST(APPEND LINK_LIBS_DEBUG ${deb__})
        endif()

        GET_TARGET_PROPERTY(imp_libs_rel_with_deb ${lib} IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO)
        GET_TARGET_PROPERTY(imp_libs_rel ${lib} IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE)
        GET_TARGET_PROPERTY(imp_libs_deb ${lib} IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG)

        foreach(imp_lib ${imp_libs_rel})
          if(EXISTS ${imp_lib})
            GET_FILENAME_COMPONENT(dir ${imp_lib} DIRECTORY)
            LIST(APPEND LINK_DIRS_RELEASE "${dir}")
            #get_filename_component(name ${imp_lib} NAME_WE)
            rcc_strip_extension(imp_lib name)
            if(RCC_VERBOSE_CONFIG)
              message(STATUS "  Link dir: [${name}] ${dir}")
            endif(RCC_VERBOSE_CONFIG)
            #string(REPLACE "lib" "" name "${name}")
            string(SUBSTRING "${name}" 0 3 sub)
            if(${sub} STREQUAL lib)
              string(SUBSTRING "${name}" 3 -1 name)
            endif()
            LIST(APPEND LINK_LIBS_RELEASE ${name})
          endif()
        endforeach()

        foreach(imp_lib ${imp_libs_rel_with_deb})
          if(EXISTS ${imp_lib})
            GET_FILENAME_COMPONENT(dir ${imp_lib} DIRECTORY)
            LIST(APPEND LINK_DIRS_RELEASE "${dir}")
            #get_filename_component(name ${imp_lib} NAME_WE)
            rcc_strip_extension(imp_lib name)
            if(RCC_VERBOSE_CONFIG)
              message(STATUS "  Link dir: [${name}] ${dir}")
            endif(RCC_VERBOSE_CONFIG)
            list(APPEND LINK_LIBS_RELEASE ${name})
          endif()
        endforeach()

        foreach(imp_lib ${imp_libs_deb})
          if(EXISTS ${imp_lib})
            GET_FILENAME_COMPONENT(dir ${imp_lib} DIRECTORY)
            LIST(APPEND LINK_DIRS_DEBUG "${dir}")
            #get_filename_component(name ${imp_lib} NAME_WE)
            rcc_strip_extension(imp_lib name)
            if(RCC_VERBOSE_CONFIG)
              message(STATUS "  Link dir: [${name}] ${dir}")
            endif(RCC_VERBOSE_CONFIG)
            list(APPEND LINK_LIBS_DEBUG ${name})
          endif()
        endforeach()
      else(TARGET ${lib})
        IF(EXISTS ${lib})
        # if it's a file on disk
          TARGET_LINK_LIBRARIES(${TARGET} ${lib})
          GET_FILENAME_COMPONENT(DIR ${lib} DIRECTORY)
          LIST(APPEND LINK_DIRS_DEBUG "${DIR}")
          LIST(APPEND LINK_DIRS_RELEASE "${DIR}")
          #get_filename_component(name ${lib} NAME_WE)
          rcc_strip_extension(lib name)
          if(RCC_VERBOSE_CONFIG AND DIR)
            message(STATUS "  Link dir: [${name}] ${dir}")
          endif(RCC_VERBOSE_CONFIG AND DIR)
          string(SUBSTRING "${name}" 0 3 sub)
          if(${sub} STREQUAL lib)
            string(SUBSTRING "${name}" 3 -1 name)
          endif()
        #string(REPLACE "lib" "" name "${name}")
          if(${IS_DEBUG} STREQUAL "0")
            list(APPEND LINK_LIBS_RELEASE "${name}")
          elseif(${IS_DEBUG} STREQUAL "1")
            list(APPEND LINK_LIBS_DEBUG "${name}")
          else() # IS_DEBUG == 2  thus debug / optimized was not specified
            list(APPEND LINK_LIBS_DEBUG "${name}")
            list(APPEND LINK_LIBS_RELEASE "${name}")
          endif()
        endif(EXISTS ${lib})
      endif(TARGET ${lib})
      TARGET_LINK_LIBRARIES(${TARGET} ${PREFIX}${lib})
    ENDIF(${SKIP} STREQUAL "0")
  ENDFOREACH(lib)
  LIST(REMOVE_DUPLICATES LINK_DIRS_DEBUG)
  LIST(REMOVE_DUPLICATES LINK_DIRS_RELEASE)

ENDMACRO(RCC_LINK_LIB TARGET)




if(UNIX AND NOT APPLE)
    set(BUILD_TYPE STATIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
else()
    set(BUILD_TYPE STATIC)
    list(APPEND CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG")
endif()

include(Aurora/cmake/ProjectFilesExternal.cmake)
include_directories(${IObject_INCLUDE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Aurora/RuntimeCompiler)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Aurora/RuntimeObjectSystem)

FIND_PACKAGE(CUDA QUIET)
message(STATUS 
"RCC config:
  IObject include dir: ${IObject_INCLUDE}
  NVCC: ${CUDA_TOOLKIT_ROOT_DIR}
"
)
set(NVCC_COMPILER "")
IF(CUDA_FOUND)
    set(NVCC_COMPILER "${CUDA_NVCC_EXECUTABLE}")
ENDIF()
set(COMPILER_PATH ${CMAKE_CXX_COMPILER};${NVCC_COMPILER})


SET(RCC_INCLUDE_DIRS "${IObject_INCLUDE};${CMAKE_CURRENT_SOURCE_DIR}/Aurora/RuntimeObjectSystem;${CMAKE_CURRENT_SOURCE_DIR}/Aurora/RuntimeCompiler" PARENT_SCOPE)
if(RCC_INCLUDE_DEPENDENCIES)
  LIST(REMOVE_DUPLICATES RCC_INCLUDE_DEPENDENCIES) 
endif()
if(RCC_LIBRARY_DIRS_RELEASE)
  LIST(REMOVE_DUPLICATES RCC_LIBRARY_DIRS_RELEASE)
endif()
if(RCC_LIBRARY_DIRS_DEBUG)
  LIST(REMOVE_DUPLICATES RCC_LIBRARY_DIRS_DEBUG)
endif()
if(RCC_LIBRARY_DIRS_RELWITHDEBINFO)
  LIST(REMOVE_DUPLICATES RCC_LIBRARY_DIRS_RELWITHDEBINFO)
endif()

string(REGEX REPLACE ";" "\n" RCC_INCLUDE_DEPENDENCIES "${RCC_INCLUDE_DEPENDENCIES}")
string(REGEX REPLACE ";" "\n" RCC_LIBRARY_DIRS_DEBUG "${RCC_LIBRARY_DIRS_DEBUG}")
string(REGEX REPLACE ";" "\n" RCC_LIBRARY_DIRS_RELEASE "${RCC_LIBRARY_DIRS_RELEASE}")
string(REGEX REPLACE ";" "\n" RCC_COMPILE_FLAGS "${RCC_COMPILE_FLAGS}")
string(REGEX REPLACE ";" "\n" RCC_LIBRARY_DIRS_RELWITHDEBINFO "${RCC_LIBRARY_DIRS_RELWITHDEBINFO}")
IF(WIN32)
    FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/RCC_config.txt" 
        "project_id:\n0\n"
        "include_dirs:\n${RCC_INCLUDE_DEPENDENCIES}\n"
        "lib_dirs_debug:\n${RCC_LIBRARY_DIRS_DEBUG};\n"
        "compile_options:\n/FS ${RCC_COMPILE_FLAGS}\n"
        "compiler_locations:\n${COMPILER_PATH}")

    FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/RCC_config.txt" 
        "project_id:\n0\n"
        "\ninclude_dirs:\n${RCC_INCLUDE_DEPENDENCIES}\n"
        "\nlib_dirs_release:\n${RCC_LIBRARY_DIRS_RELEASE};\n"
        "\ncompile_options:\n/FS ${RCC_COMPILE_FLAGS}\n"
        "\ncompiler_location:\n${COMPILER_PATH}")
	
    FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RelWithDebInfo/RCC_config.txt" 
        "project_id:\n0\n"
        "\ninclude_dirs:\n${RCC_INCLUDE_DEPENDENCIES}\n"
        "lib_dirs_release:\n${RCC_LIBRARY_DIRS_RELWITHDEBINFO};\n"
        "compile_options:\n/FS ${RCC_COMPILE_FLAGS}\n"
        "compiler_location:\n${COMPILER_PATH}")
	
    FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RCC_config.txt" 
        "project_id:\n0\n"
        "\ninclude_dirs:\n${RCC_INCLUDE_DEPENDENCIES}\n"
        "\nlib_dirs_debug:\n${RCC_LIBRARY_DIRS_DEBUG}\n"
        "\nlib_dirs_release:\n${RCC_LIBRARY_DIRS_RELWITHDEBINFO};\n"
        "\ncompile_options:\n\n/FS ${RCC_COMPILE_FLAGS}\n"
        "\ncompiler_location:\n${COMPILER_PATH}")
ELSE()
    FILE(WRITE "${CMAKE_BINARY_DIR}/RCC_config.txt"
        "project_id:\n0\n"
        "\ninclude_dirs:\n${RCC_INCLUDE_DEPENDENCIES}\n"
        "\nlib_dirs_debug:\n${RCC_LIBRARY_DIRS_DEBUG}\n${CMAKE_BINARY_DIR}\n"
        "\nlib_dirs_release:\n${RCC_LIBRARY_DIRS_RELEASE}\n${CMAKE_BINARY_DIR}"
        "\ncompile_options:\n${RCC_COMPILE_FLAGS}\n"
        "\ncompiler_location:\n${COMPILER_PATH}")

    FILE(WRITE "${CMAKE_CURRENT_BINARY_DIR}/RCC_config.txt"
        "project_id:\n0\n"
        "\ninclude_dirs:\n${RCC_INCLUDE_DEPENDENCIES}\n"
        "\nlib_dirs_debug:\n${RCC_LIBRARY_DIRS_DEBUG}\n${CMAKE_BINARY_DIR}\n"
        "\nlib_dirs_release:\n${RCC_LIBRARY_DIRS_RELEASE}\n${CMAKE_BINARY_DIR}"
        "\ncompile_options:\n${RCC_COMPILE_FLAGS}\n"
        "\ncompiler_location:\n${COMPILER_PATH}")
ENDIF()

set(RCC_PERMODULE_INTERFACE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/Aurora/RuntimeObjectSystem/ObjectInterfacePerModuleSource.cpp" PARENT_SCOPE)

add_library(RuntimeCompiler ${BUILD_TYPE} ${RuntimeCompiler_SRCS})

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden")
endif(NOT WIN32)
add_library(RuntimeObjectSystem STATIC ${RuntimeObjectSystem_SRCS})
target_link_libraries(RuntimeObjectSystem RuntimeCompiler)

install(TARGETS RuntimeObjectSystem RuntimeCompiler
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY Aurora/include
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY Aurora/include
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

if(UNIX)
        target_link_libraries(RuntimeObjectSystem dl)
endif()
