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

# Include any dependencies generated for this target.
include dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/depend.make

# Include the progress variables for this target.
include dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/progress.make

# Include the compile flags for this target's objects.
include dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/flags.make

dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/enum.cpp.o: dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/flags.make
dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/enum.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/enum/enum.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/enum.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_ct_enum.dir/enum.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/enum/enum.cpp

dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/enum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ct_enum.dir/enum.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/enum/enum.cpp > CMakeFiles/test_ct_enum.dir/enum.cpp.i

dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/enum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ct_enum.dir/enum.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/enum/enum.cpp -o CMakeFiles/test_ct_enum.dir/enum.cpp.s

# Object files for target test_ct_enum
test_ct_enum_OBJECTS = \
"CMakeFiles/test_ct_enum.dir/enum.cpp.o"

# External object files for target test_ct_enum
test_ct_enum_EXTERNAL_OBJECTS =

test_ct_enum: dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/enum.cpp.o
test_ct_enum: dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/build.make
test_ct_enum: lib/libgtestd.a
test_ct_enum: lib/libgtest_maind.a
test_ct_enum: lib/libgtestd.a
test_ct_enum: dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../test_ct_enum"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ct_enum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/build: test_ct_enum

.PHONY : dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/build

dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum && $(CMAKE_COMMAND) -P CMakeFiles/test_ct_enum.dir/cmake_clean.cmake
.PHONY : dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/clean

dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/enum /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/ct/tests/enum/CMakeFiles/test_ct_enum.dir/depend

