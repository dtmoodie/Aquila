CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(Casablanca_LIBRARY pplx)
set(CMAKE_DEBUG_POSTFIX d)

set(SOURCES_COMMON
  src/pplx/pplx.cpp
  src/utilities/asyncrt_utils.cpp
)
SET(PPLX_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATHS "" FORCE)
SET(PPLX_LIBRARIES pplx CACHE PATHS "" FORCE)


# THE ORDER OF FILES IS VERY /VERY/ IMPORTANT
if(UNIX)
  set(SOURCES
    ${SOURCES_COMMON}
    src/pplx/threadpool.cpp
  )
  if(APPLE)
    list(APPEND SOURCES
      src/pplx/pplxapple.cpp
    )
    find_library(COREFOUNDATION CoreFoundation "/")
    find_library(SECURITY Security "/")
    set(EXTRALINKS ${COREFOUNDATION} ${SECURITY})
  else(APPLE)
    list(APPEND SOURCES
      src/pplx/pplxlinux.cpp
    )
  endif(APPLE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS} -Werror -pedantic")
else(UNIX)
  if(WIN32)
    set(SOURCES
      ${SOURCES_COMMON}
      src/pplx/pplxwin.cpp
    )
    set(EXTRALINKS
      bcrypt.lib
      crypt32.lib
      httpapi.lib
      Winhttp.lib
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS} /Zm200")

  #set_source_files_properties(pch/stdafx.cpp PROPERTIES COMPILE_FLAGS "/Ycstdafx.h")
    if (BUILD_SHARED_LIBS)
      add_definitions(-D_ASYNCRT_EXPORT -D_PPLX_EXPORT -D_USRDLL)
      ADD_DEFINITIONS("-DCPPREST_FORCE_PPLX=1")
      ADD_DEFINITIONS("-D_PPLX_EXPORT")
      ADD_DEFINITIONS("-D_ASYNCRT_EXPORT")
    endif(BUILD_SHARED_LIBS)
  endif(WIN32)
endif(UNIX)
include_directories("src/pch")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_library(${Casablanca_LIBRARY} SHARED ${SOURCES})
#cotire(${Casablanca_LIBRARY})
target_link_libraries(${Casablanca_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_ATOMIC_LIBRARY}
  ${Boost_CHRONO_LIBRARY}
  ${Boost_RANDOM_LIBRARY}
  ${EXTRALINKS}
  ${Boost_FRAMEWORK}
  )

# Portions specific to cpprest binary versioning.
set (CPPREST_VERSION_MAJOR 2)
set (CPPREST_VERSION_MINOR 7)
set (CPPREST_VERSION_REVISION 0)

if(WIN32)
  set_target_properties(${Casablanca_LIBRARY} PROPERTIES
    OUTPUT_NAME "${Casablanca_LIBRARY}_${CPPREST_VERSION_MAJOR}_${CPPREST_VERSION_MINOR}")
elseif(ANDROID)
  # Do not use SOVERSION on android. It is completely unsupported (and causes problems).
  # Perhaps revisit in the future? (NDK r9d, 8/7/14)
else()
  set_target_properties(${Casablanca_LIBRARY} PROPERTIES
    SOVERSION ${CPPREST_VERSION_MAJOR}.${CPPREST_VERSION_MINOR})

  install(
    TARGETS ${Casablanca_LIBRARY}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

	install(DIRECTORY include
		DESTINATION include
        FILES_MATCHING PATTERN "*.h")
	install(DIRECTORY include
		DESTINATION include
		FILES_MATCHING PATTERN "*.hpp")
endif()

