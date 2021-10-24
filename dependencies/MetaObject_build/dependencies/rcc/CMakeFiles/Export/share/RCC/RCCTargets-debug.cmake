#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "RuntimeObjectSystem" for configuration "Debug"
set_property(TARGET RuntimeObjectSystem APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(RuntimeObjectSystem PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "RuntimeCompiler;dl"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libRuntimeObjectSystemd.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS RuntimeObjectSystem )
list(APPEND _IMPORT_CHECK_FILES_FOR_RuntimeObjectSystem "${_IMPORT_PREFIX}/lib/libRuntimeObjectSystemd.a" )

# Import target "RuntimeCompiler" for configuration "Debug"
set_property(TARGET RuntimeCompiler APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(RuntimeCompiler PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libRuntimeCompilerd.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS RuntimeCompiler )
list(APPEND _IMPORT_CHECK_FILES_FOR_RuntimeCompiler "${_IMPORT_PREFIX}/lib/libRuntimeCompilerd.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
