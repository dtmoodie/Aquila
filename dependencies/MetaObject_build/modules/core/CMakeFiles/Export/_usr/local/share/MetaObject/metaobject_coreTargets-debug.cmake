#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metaobject_core" for configuration "Debug"
set_property(TARGET metaobject_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(metaobject_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "Boost::thread;Boost::fiber;Boost::system;Boost::chrono;RuntimeObjectSystem;spdlog::spdlog;ct;pthread;/usr/local/cuda/lib64/libcudart.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libmetaobject_cored.so"
  IMPORTED_SONAME_DEBUG "libmetaobject_cored.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metaobject_core )
list(APPEND _IMPORT_CHECK_FILES_FOR_metaobject_core "${_IMPORT_PREFIX}/lib/libmetaobject_cored.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
