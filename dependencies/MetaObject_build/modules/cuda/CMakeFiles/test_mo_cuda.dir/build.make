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
include modules/cuda/CMakeFiles/test_mo_cuda.dir/depend.make

# Include the progress variables for this target.
include modules/cuda/CMakeFiles/test_mo_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include modules/cuda/CMakeFiles/test_mo_cuda.dir/flags.make

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o.depend
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o.Debug.cmake
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/device_funcs.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests && /usr/local/bin/cmake -E make_directory /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/.
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/./test_mo_cuda_generated_device_funcs.cu.o -D generated_cubin_file:STRING=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/./test_mo_cuda_generated_device_funcs.cu.o.cubin.txt -P /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o.Debug.cmake

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/flags.make
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/allocator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/allocator.cpp

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/allocator.cpp > CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.i

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/allocator.cpp -o CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.s

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/flags.make
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/async_stream.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/async_stream.cpp

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/async_stream.cpp > CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.i

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/async_stream.cpp -o CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.s

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/flags.make
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/device_synchronization.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/device_synchronization.cpp

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/device_synchronization.cpp > CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.i

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/device_synchronization.cpp -o CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.s

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/event.cpp.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/flags.make
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/event.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/event.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/event.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_cuda.dir/tests/event.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/event.cpp

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/event.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_cuda.dir/tests/event.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/event.cpp > CMakeFiles/test_mo_cuda.dir/tests/event.cpp.i

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/event.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_cuda.dir/tests/event.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/event.cpp -o CMakeFiles/test_mo_cuda.dir/tests/event.cpp.s

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/main.cpp.o: modules/cuda/CMakeFiles/test_mo_cuda.dir/flags.make
modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/main.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/main.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_cuda.dir/tests/main.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/main.cpp

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_cuda.dir/tests/main.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/main.cpp > CMakeFiles/test_mo_cuda.dir/tests/main.cpp.i

modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_cuda.dir/tests/main.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda/tests/main.cpp -o CMakeFiles/test_mo_cuda.dir/tests/main.cpp.s

# Object files for target test_mo_cuda
test_mo_cuda_OBJECTS = \
"CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.o" \
"CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.o" \
"CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.o" \
"CMakeFiles/test_mo_cuda.dir/tests/event.cpp.o" \
"CMakeFiles/test_mo_cuda.dir/tests/main.cpp.o"

# External object files for target test_mo_cuda
test_mo_cuda_EXTERNAL_OBJECTS = \
"/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o"

test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/allocator.cpp.o
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/async_stream.cpp.o
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/device_synchronization.cpp.o
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/event.cpp.o
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/main.cpp.o
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/build.make
test_mo_cuda: /usr/local/cuda/lib64/libcudart_static.a
test_mo_cuda: /usr/lib/x86_64-linux-gnu/librt.so
test_mo_cuda: libmetaobject_cudad.so
test_mo_cuda: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
test_mo_cuda: lib/libgtestd.a
test_mo_cuda: lib/libgtest_maind.a
test_mo_cuda: libmetaobject_typesd.so
test_mo_cuda: libmetaobject_cored.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_thread.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_date_time.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_atomic.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_fiber.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_context.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_system.so
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_chrono.so
test_mo_cuda: libRuntimeObjectSystemd.a
test_mo_cuda: libRuntimeCompilerd.a
test_mo_cuda: /home/dan/code/boost/stage/lib/libboost_filesystem.so
test_mo_cuda: /usr/local/cuda/lib64/libcudart.so
test_mo_cuda: /usr/local/lib/libopencv_core.so.3.4.1
test_mo_cuda: /usr/local/lib/libopencv_cudev.so.3.4.1
test_mo_cuda: lib/libgtestd.a
test_mo_cuda: modules/cuda/CMakeFiles/test_mo_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable ../../test_mo_cuda"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_mo_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/cuda/CMakeFiles/test_mo_cuda.dir/build: test_mo_cuda

.PHONY : modules/cuda/CMakeFiles/test_mo_cuda.dir/build

modules/cuda/CMakeFiles/test_mo_cuda.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda && $(CMAKE_COMMAND) -P CMakeFiles/test_mo_cuda.dir/cmake_clean.cmake
.PHONY : modules/cuda/CMakeFiles/test_mo_cuda.dir/clean

modules/cuda/CMakeFiles/test_mo_cuda.dir/depend: modules/cuda/CMakeFiles/test_mo_cuda.dir/tests/test_mo_cuda_generated_device_funcs.cu.o
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/cuda /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/cuda/CMakeFiles/test_mo_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/cuda/CMakeFiles/test_mo_cuda.dir/depend

