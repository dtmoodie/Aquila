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
include dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/depend.make

# Include the progress variables for this target.
include dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/progress.make

# Include the compile flags for this target's objects.
include dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/flags.make

dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.o: dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/flags.make
dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/string/runtime_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/string/runtime_test.cpp

dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/string/runtime_test.cpp > CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.i

dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/string/runtime_test.cpp -o CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.s

# Object files for target test_ct_string_runtime
test_ct_string_runtime_OBJECTS = \
"CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.o"

# External object files for target test_ct_string_runtime
test_ct_string_runtime_EXTERNAL_OBJECTS =

test_ct_string_runtime: dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/runtime_test.cpp.o
test_ct_string_runtime: dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/build.make
test_ct_string_runtime: lib/libgtestd.a
test_ct_string_runtime: lib/libgtest_maind.a
test_ct_string_runtime: lib/libgtestd.a
test_ct_string_runtime: dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../test_ct_string_runtime"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ct_string_runtime.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/build: test_ct_string_runtime

.PHONY : dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/build

dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string && $(CMAKE_COMMAND) -P CMakeFiles/test_ct_string_runtime.dir/cmake_clean.cmake
.PHONY : dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/clean

dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/string /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/ct/tests/string/CMakeFiles/test_ct_string_runtime.dir/depend

