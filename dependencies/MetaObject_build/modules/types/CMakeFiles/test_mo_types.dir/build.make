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
include modules/types/CMakeFiles/test_mo_types.dir/depend.make

# Include the progress variables for this target.
include modules/types/CMakeFiles/test_mo_types.dir/progress.make

# Include the compile flags for this target's objects.
include modules/types/CMakeFiles/test_mo_types.dir/flags.make

modules/types/CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.o: modules/types/CMakeFiles/test_mo_types.dir/flags.make
modules/types/CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/test_mo_small_vec.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/types/CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/test_mo_small_vec.cpp

modules/types/CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/test_mo_small_vec.cpp > CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.i

modules/types/CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/test_mo_small_vec.cpp -o CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.s

modules/types/CMakeFiles/test_mo_types.dir/tests/main.cpp.o: modules/types/CMakeFiles/test_mo_types.dir/flags.make
modules/types/CMakeFiles/test_mo_types.dir/tests/main.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/types/CMakeFiles/test_mo_types.dir/tests/main.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_types.dir/tests/main.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/main.cpp

modules/types/CMakeFiles/test_mo_types.dir/tests/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_types.dir/tests/main.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/main.cpp > CMakeFiles/test_mo_types.dir/tests/main.cpp.i

modules/types/CMakeFiles/test_mo_types.dir/tests/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_types.dir/tests/main.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/tests/main.cpp -o CMakeFiles/test_mo_types.dir/tests/main.cpp.s

# Object files for target test_mo_types
test_mo_types_OBJECTS = \
"CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.o" \
"CMakeFiles/test_mo_types.dir/tests/main.cpp.o"

# External object files for target test_mo_types
test_mo_types_EXTERNAL_OBJECTS =

test_mo_types: modules/types/CMakeFiles/test_mo_types.dir/tests/test_mo_small_vec.cpp.o
test_mo_types: modules/types/CMakeFiles/test_mo_types.dir/tests/main.cpp.o
test_mo_types: modules/types/CMakeFiles/test_mo_types.dir/build.make
test_mo_types: libmetaobject_typesd.so
test_mo_types: lib/libgtestd.a
test_mo_types: lib/libgtest_maind.a
test_mo_types: libmetaobject_cored.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_thread.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_date_time.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_atomic.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_fiber.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_context.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_system.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_chrono.so
test_mo_types: libRuntimeObjectSystemd.a
test_mo_types: libRuntimeCompilerd.a
test_mo_types: /usr/local/cuda/lib64/libcudart.so
test_mo_types: /home/dan/code/boost/stage/lib/libboost_filesystem.so
test_mo_types: lib/libgtestd.a
test_mo_types: modules/types/CMakeFiles/test_mo_types.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../test_mo_types"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_mo_types.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/types/CMakeFiles/test_mo_types.dir/build: test_mo_types

.PHONY : modules/types/CMakeFiles/test_mo_types.dir/build

modules/types/CMakeFiles/test_mo_types.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types && $(CMAKE_COMMAND) -P CMakeFiles/test_mo_types.dir/cmake_clean.cmake
.PHONY : modules/types/CMakeFiles/test_mo_types.dir/clean

modules/types/CMakeFiles/test_mo_types.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/types/CMakeFiles/test_mo_types.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/types/CMakeFiles/test_mo_types.dir/depend

