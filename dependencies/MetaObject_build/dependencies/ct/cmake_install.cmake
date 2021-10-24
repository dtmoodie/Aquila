# Install script for directory: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/ct/ct-config.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/ct/ct-config.cmake"
         "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/CMakeFiles/Export/share/ct/ct-config.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/ct/ct-config-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/ct/ct-config.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ct" TYPE FILE FILES "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/CMakeFiles/Export/share/ct/ct-config.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/include/ct")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/cpgf/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/hash/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/python/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/reflect/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/type_trait_detectors/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/variadic_typedef/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/all_the_things/cmake_install.cmake")
  include("/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options/cmake_install.cmake")

endif()

