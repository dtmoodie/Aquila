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
include dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/depend.make

# Include the progress variables for this target.
include dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/progress.make

# Include the compile flags for this target's objects.
include dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/flags.make

dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/main.cpp.o: dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/flags.make
dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/main.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/examples/boost_program_options/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/main.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_ct_boost.dir/main.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/examples/boost_program_options/main.cpp

dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_ct_boost.dir/main.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/examples/boost_program_options/main.cpp > CMakeFiles/example_ct_boost.dir/main.cpp.i

dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_ct_boost.dir/main.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/examples/boost_program_options/main.cpp -o CMakeFiles/example_ct_boost.dir/main.cpp.s

# Object files for target example_ct_boost
example_ct_boost_OBJECTS = \
"CMakeFiles/example_ct_boost.dir/main.cpp.o"

# External object files for target example_ct_boost
example_ct_boost_EXTERNAL_OBJECTS =

example_ct_boost: dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/main.cpp.o
example_ct_boost: dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/build.make
example_ct_boost: /home/dan/code/boost/stage/lib/libboost_program_options.so
example_ct_boost: dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../example_ct_boost"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_ct_boost.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/build: example_ct_boost

.PHONY : dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/build

dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options && $(CMAKE_COMMAND) -P CMakeFiles/example_ct_boost.dir/cmake_clean.cmake
.PHONY : dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/clean

dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/examples/boost_program_options /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/ct/examples/boost_program_options/CMakeFiles/example_ct_boost.dir/depend

