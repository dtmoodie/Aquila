#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_runtime_reflection" for configuration "Debug"
set_property(TARGET metaobject_runtime_reflection APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_runtime_reflection PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_core;metaobject_types"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_runtime_reflectiond.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_runtime_reflectiond.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_runtime_reflection )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_runtime_reflection "${_IMPORT_PREFIX}/lib/libmetaobject_runtime_reflectiond.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
