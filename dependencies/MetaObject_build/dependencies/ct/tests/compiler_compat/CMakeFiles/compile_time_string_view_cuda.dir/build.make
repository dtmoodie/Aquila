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
include dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/depend.make

# Include the progress variables for this target.
include dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/flags.make

dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o: dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o.depend
dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o: dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o.Debug.cmake
dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/compiler_compat/00_compile_time_string_view.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir && /usr/local/bin/cmake -E make_directory /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir//.
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir//./compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o -D generated_cubin_file:STRING=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir//./compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o.cubin.txt -P /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir//compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o.Debug.cmake

# Object files for target compile_time_string_view_cuda
compile_time_string_view_cuda_OBJECTS =

# External object files for target compile_time_string_view_cuda
compile_time_string_view_cuda_EXTERNAL_OBJECTS = \
"/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o"

compile_time_string_view_cuda: dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o
compile_time_string_view_cuda: dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/build.make
compile_time_string_view_cuda: /usr/local/cuda/lib64/libcudart_static.a
compile_time_string_view_cuda: /usr/lib/x86_64-linux-gnu/librt.so
compile_time_string_view_cuda: dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../compile_time_string_view_cuda"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_time_string_view_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/build: compile_time_string_view_cuda

.PHONY : dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/build

dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat && $(CMAKE_COMMAND) -P CMakeFiles/compile_time_string_view_cuda.dir/cmake_clean.cmake
.PHONY : dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/clean

dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/depend: dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/compile_time_string_view_cuda_generated_00_compile_time_string_view.cu.o
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/tests/compiler_compat /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/ct/tests/compiler_compat/CMakeFiles/compile_time_string_view_cuda.dir/depend

