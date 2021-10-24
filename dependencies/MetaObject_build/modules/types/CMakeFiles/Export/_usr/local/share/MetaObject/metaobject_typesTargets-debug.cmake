#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_types" for configuration "Debug"
set_property(TARGET metaobject_types APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_types PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "ct;metaobject_core;/home/dan/code/boost/stage/lib/libboost_filesystem.so;/home/dan/code/boost/stage/lib/libboost_filesystem.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_typesd.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_typesd.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_types )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_types "${_IMPORT_PREFIX}/lib/libmetaobject_typesd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
