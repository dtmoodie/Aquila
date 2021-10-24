#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_serialization" for configuration "Debug"
set_property(TARGET metaobject_serialization APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_serialization PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_core;metaobject_params;metaobject_types;cereal"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_serializationd.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_serializationd.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_serialization )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_serialization "${_IMPORT_PREFIX}/lib/libmetaobject_serializationd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
