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
include dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/depend.make

# Include the progress variables for this target.
include dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/progress.make

# Include the compile flags for this target's objects.
include dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/flags.make

dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.o: dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/flags.make
dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/tests/test_shared_ptr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/tests/test_shared_ptr.cpp

dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/tests/test_shared_ptr.cpp > CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.i

dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/tests/test_shared_ptr.cpp -o CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.s

# Object files for target test_shared_ptr
test_shared_ptr_OBJECTS = \
"CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.o"

# External object files for target test_shared_ptr
test_shared_ptr_EXTERNAL_OBJECTS =

test_shared_ptr: dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/test_shared_ptr.cpp.o
test_shared_ptr: dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/build.make
test_shared_ptr: libRuntimeObjectSystemd.a
test_shared_ptr: libRuntimeCompilerd.a
test_shared_ptr: dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../test_shared_ptr"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_shared_ptr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/build: test_shared_ptr

.PHONY : dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/build

dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_shared_ptr.dir/cmake_clean.cmake
.PHONY : dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/clean

dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/tests /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/rcc/Aurora/tests/CMakeFiles/test_shared_ptr.dir/depend

