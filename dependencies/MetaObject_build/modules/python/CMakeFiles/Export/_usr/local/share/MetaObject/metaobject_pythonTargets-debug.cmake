#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_python" for configuration "Debug"
set_property(TARGET metaobject_python APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_python PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "metaobject_core;metaobject_object;metaobject_cuda;opencv_core;/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5m.so;ct"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_pythond.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_pythond.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_python )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_python "${_IMPORT_PREFIX}/lib/libmetaobject_pythond.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
