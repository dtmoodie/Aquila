#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_params" for configuration "Debug"
set_property(TARGET metaobject_params APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_params PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_core;ct;metaobject_types;metaobject_runtime_reflection"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_paramsd.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_paramsd.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_params )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_params "${_IMPORT_PREFIX}/lib/libmetaobject_paramsd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
