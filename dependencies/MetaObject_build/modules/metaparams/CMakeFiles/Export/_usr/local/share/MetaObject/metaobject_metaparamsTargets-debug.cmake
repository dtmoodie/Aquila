#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_metaparams" for configuration "Debug"
set_property(TARGET metaobject_metaparams APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_metaparams PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_params;metaobject_serialization;metaobject_types"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_metaparamsd.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_metaparamsd.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_metaparams )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_metaparams "${_IMPORT_PREFIX}/lib/libmetaobject_metaparamsd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
