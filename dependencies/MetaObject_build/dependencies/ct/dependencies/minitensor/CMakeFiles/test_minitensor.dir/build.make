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
include dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/depend.make

# Include the progress variables for this target.
include dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/progress.make

# Include the compile flags for this target's objects.
include dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/flags.make

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/array.cpp.o: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/flags.make
dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/array.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/array.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/array.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_minitensor.dir/tests/array.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/array.cpp

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/array.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_minitensor.dir/tests/array.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/array.cpp > CMakeFiles/test_minitensor.dir/tests/array.cpp.i

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/array.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_minitensor.dir/tests/array.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/array.cpp -o CMakeFiles/test_minitensor.dir/tests/array.cpp.s

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/main.cpp.o: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/flags.make
dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/main.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/main.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_minitensor.dir/tests/main.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/main.cpp

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_minitensor.dir/tests/main.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/main.cpp > CMakeFiles/test_minitensor.dir/tests/main.cpp.i

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_minitensor.dir/tests/main.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/main.cpp -o CMakeFiles/test_minitensor.dir/tests/main.cpp.s

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/print.cpp.o: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/flags.make
dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/print.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/print.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/print.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_minitensor.dir/tests/print.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/print.cpp

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/print.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_minitensor.dir/tests/print.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/print.cpp > CMakeFiles/test_minitensor.dir/tests/print.cpp.i

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/print.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_minitensor.dir/tests/print.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/print.cpp -o CMakeFiles/test_minitensor.dir/tests/print.cpp.s

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/shape.cpp.o: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/flags.make
dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/shape.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/shape.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/shape.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_minitensor.dir/tests/shape.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/shape.cpp

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/shape.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_minitensor.dir/tests/shape.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/shape.cpp > CMakeFiles/test_minitensor.dir/tests/shape.cpp.i

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/shape.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_minitensor.dir/tests/shape.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/shape.cpp -o CMakeFiles/test_minitensor.dir/tests/shape.cpp.s

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/tensor.cpp.o: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/flags.make
dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/tensor.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/tensor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/tensor.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_minitensor.dir/tests/tensor.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/tensor.cpp

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_minitensor.dir/tests/tensor.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/tensor.cpp > CMakeFiles/test_minitensor.dir/tests/tensor.cpp.i

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_minitensor.dir/tests/tensor.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/tests/tensor.cpp -o CMakeFiles/test_minitensor.dir/tests/tensor.cpp.s

# Object files for target test_minitensor
test_minitensor_OBJECTS = \
"CMakeFiles/test_minitensor.dir/tests/array.cpp.o" \
"CMakeFiles/test_minitensor.dir/tests/main.cpp.o" \
"CMakeFiles/test_minitensor.dir/tests/print.cpp.o" \
"CMakeFiles/test_minitensor.dir/tests/shape.cpp.o" \
"CMakeFiles/test_minitensor.dir/tests/tensor.cpp.o"

# External object files for target test_minitensor
test_minitensor_EXTERNAL_OBJECTS =

test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/array.cpp.o
test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/main.cpp.o
test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/print.cpp.o
test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/shape.cpp.o
test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/tests/tensor.cpp.o
test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/build.make
test_minitensor: lib/libgtestd.a
test_minitensor: lib/libgtest_maind.a
test_minitensor: lib/libgtestd.a
test_minitensor: dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../../../../test_minitensor"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_minitensor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/build: test_minitensor

.PHONY : dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/build

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor && $(CMAKE_COMMAND) -P CMakeFiles/test_minitensor.dir/cmake_clean.cmake
.PHONY : dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/clean

dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/ct/dependencies/minitensor/CMakeFiles/test_minitensor.dir/depend

