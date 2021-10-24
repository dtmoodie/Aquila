#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_cuda" for configuration "Debug"
set_property(TARGET metaobject_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_cuda PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_core;metaobject_types;/usr/local/cuda/lib64/libcudart.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_cudad.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_cudad.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_cuda "${_IMPORT_PREFIX}/lib/libmetaobject_cudad.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
