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
include modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/depend.make

# Include the progress variables for this target.
include modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/progress.make

# Include the compile flags for this target's objects.
include modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/flags.make

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.o: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/flags.make
modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/main.cpp

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/main.cpp > CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.i

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/main.cpp -o CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.s

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.o: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/flags.make
modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/print.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/print.cpp

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/print.cpp > CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.i

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/print.cpp -o CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.s

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.o: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/flags.make
modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static.cpp

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static.cpp > CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.i

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static.cpp -o CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.s

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.o: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/flags.make
modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static_checks.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static_checks.cpp

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static_checks.cpp > CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.i

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/static_checks.cpp -o CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.s

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.o: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/flags.make
modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/traits.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/traits.cpp

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/traits.cpp > CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.i

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/tests/traits.cpp -o CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.s

# Object files for target test_mo_runtime_reflection
test_mo_runtime_reflection_OBJECTS = \
"CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.o" \
"CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.o" \
"CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.o" \
"CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.o" \
"CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.o"

# External object files for target test_mo_runtime_reflection
test_mo_runtime_reflection_EXTERNAL_OBJECTS =

test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/main.cpp.o
test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/print.cpp.o
test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static.cpp.o
test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/static_checks.cpp.o
test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/tests/traits.cpp.o
test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/build.make
test_mo_runtime_reflection: libmetaobject_runtime_reflectiond.so
test_mo_runtime_reflection: lib/libgtestd.a
test_mo_runtime_reflection: lib/libgtest_maind.a
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudastereo.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_dnn.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_ml.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_shape.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_stitching.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_superres.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_videostab.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_viz.so.3.4.1
test_mo_runtime_reflection: libmetaobject_typesd.so
test_mo_runtime_reflection: libmetaobject_cored.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_thread.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_date_time.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_atomic.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_fiber.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_context.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_system.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_chrono.so
test_mo_runtime_reflection: libRuntimeObjectSystemd.a
test_mo_runtime_reflection: libRuntimeCompilerd.a
test_mo_runtime_reflection: /usr/local/cuda/lib64/libcudart.so
test_mo_runtime_reflection: /home/dan/code/boost/stage/lib/libboost_filesystem.so
test_mo_runtime_reflection: lib/libgtestd.a
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudacodec.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_calib3d.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudawarping.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_features2d.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_flann.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_highgui.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_objdetect.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_photo.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudafilters.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_video.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_videoio.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_imgproc.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_core.so.3.4.1
test_mo_runtime_reflection: /usr/local/lib/libopencv_cudev.so.3.4.1
test_mo_runtime_reflection: modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../../test_mo_runtime_reflection"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_mo_runtime_reflection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/build: test_mo_runtime_reflection

.PHONY : modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/build

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection && $(CMAKE_COMMAND) -P CMakeFiles/test_mo_runtime_reflection.dir/cmake_clean.cmake
.PHONY : modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/clean

modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/runtime_reflection/CMakeFiles/test_mo_runtime_reflection.dir/depend

