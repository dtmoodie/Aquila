#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_object" for configuration "Debug"
set_property(TARGET metaobject_object APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_object PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_core;metaobject_params;RuntimeObjectSystem;RuntimeCompiler;Boost::filesystem"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_objectd.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_objectd.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_object )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_object "${_IMPORT_PREFIX}/lib/libmetaobject_objectd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
