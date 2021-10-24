# Install script for directory: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/cereal/include/" FILES_MATCHING REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/cereal/include/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/MetaObject" TYPE DIRECTORY FILES "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/cmake" FILES_MATCHING REGEX "/[^/]*\\.cmake$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/spdlog/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/core/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/params/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/metaparams/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/object/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/python/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/gui/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
