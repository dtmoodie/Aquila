# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dan/code/EagleEye/Aquila/dependencies/MetaObject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build

# Utility rule file for spdlog_headers_for_ide.

# Include the progress variables for this target.
include dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/progress.make

spdlog_headers_for_ide: dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/build.make

.PHONY : spdlog_headers_for_ide

# Rule to build all files generated by this target.
dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/build: spdlog_headers_for_ide

.PHONY : dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/build

dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/spdlog && $(CMAKE_COMMAND) -P CMakeFiles/spdlog_headers_for_ide.dir/cmake_clean.cmake
.PHONY : dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/clean

dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/spdlog /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/spdlog /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/spdlog/CMakeFiles/spdlog_headers_for_ide.dir/depend

