# - Config file for the RuntimeCompiledCplusplus package
# It defines the following variables
#  RCC_INCLUDE_DIRS - include directories for rcc
#  RCC_LIBRARIES    - libraries to link against
#  RCC_DIR          - base directory

# Compute paths
get_filename_component(RCC_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set(RCC_INCLUDE_DIRS "${RCC_DIR}/../../include")
include(${RCC_DIR}/cmake/rcc_link_lib.cmake)
include(${RCC_DIR}/cmake/rcc_find_path.cmake)
include(${RCC_DIR}/cmake/rcc_find_library.cmake)
include(${RCC_DIR}/cmake/rcc_config.cmake)
include(${RCC_DIR}/cmake/rcc_strip_extension.cmake)

find_package(ct REQUIRED)

if(NOT TARGET RuntimeCompiler AND NOT TARGET RuntimeObjectSystem)
  include("${RCC_DIR}/RCCTargets.cmake")
  set_target_properties(RuntimeObjectSystem PROPERTIES
    IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE ct
  )
endif()

set(RCC_LIBRARIES RuntimeCompiler RuntimeObjectSystem)

set(RCC_FOUND ON)
