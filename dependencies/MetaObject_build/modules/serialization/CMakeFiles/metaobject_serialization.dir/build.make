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
include modules/serialization/CMakeFiles/metaobject_serialization.dir/depend.make

# Include the progress variables for this target.
include modules/serialization/CMakeFiles/metaobject_serialization.dir/progress.make

# Include the compile flags for this target's objects.
include modules/serialization/CMakeFiles/metaobject_serialization.dir/flags.make

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.o: modules/serialization/CMakeFiles/metaobject_serialization.dir/flags.make
modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinaryLoader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinaryLoader.cpp

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinaryLoader.cpp > CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.i

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinaryLoader.cpp -o CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.s

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.o: modules/serialization/CMakeFiles/metaobject_serialization.dir/flags.make
modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinarySaver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinarySaver.cpp

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinarySaver.cpp > CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.i

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/BinarySaver.cpp -o CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.s

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.o: modules/serialization/CMakeFiles/metaobject_serialization.dir/flags.make
modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.o: /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/JSONPrinter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.o"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.o -c /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/JSONPrinter.cpp

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.i"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/JSONPrinter.cpp > CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.i

modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.s"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && /usr/bin/g++-5 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization/src/MetaObject/serialization/JSONPrinter.cpp -o CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.s

# Object files for target metaobject_serialization
metaobject_serialization_OBJECTS = \
"CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.o" \
"CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.o" \
"CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.o"

# External object files for target metaobject_serialization
metaobject_serialization_EXTERNAL_OBJECTS =

libmetaobject_serializationd.so: modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinaryLoader.cpp.o
libmetaobject_serializationd.so: modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/BinarySaver.cpp.o
libmetaobject_serializationd.so: modules/serialization/CMakeFiles/metaobject_serialization.dir/src/MetaObject/serialization/JSONPrinter.cpp.o
libmetaobject_serializationd.so: modules/serialization/CMakeFiles/metaobject_serialization.dir/build.make
libmetaobject_serializationd.so: libmetaobject_paramsd.so
libmetaobject_serializationd.so: libmetaobject_runtime_reflectiond.so
libmetaobject_serializationd.so: libmetaobject_typesd.so
libmetaobject_serializationd.so: libmetaobject_cored.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_thread.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_date_time.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_atomic.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_fiber.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_context.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_system.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_chrono.so
libmetaobject_serializationd.so: libRuntimeObjectSystemd.a
libmetaobject_serializationd.so: libRuntimeCompilerd.a
libmetaobject_serializationd.so: /usr/local/cuda/lib64/libcudart.so
libmetaobject_serializationd.so: /home/dan/code/boost/stage/lib/libboost_filesystem.so
libmetaobject_serializationd.so: modules/serialization/CMakeFiles/metaobject_serialization.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../../libmetaobject_serializationd.so"
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/metaobject_serialization.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/serialization/CMakeFiles/metaobject_serialization.dir/build: libmetaobject_serializationd.so

.PHONY : modules/serialization/CMakeFiles/metaobject_serialization.dir/build

modules/serialization/CMakeFiles/metaobject_serialization.dir/clean:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization && $(CMAKE_COMMAND) -P CMakeFiles/metaobject_serialization.dir/cmake_clean.cmake
.PHONY : modules/serialization/CMakeFiles/metaobject_serialization.dir/clean

modules/serialization/CMakeFiles/metaobject_serialization.dir/depend:
	cd /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dan/code/EagleEye/Aquila/dependencies/MetaObject /home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/serialization /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization /home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/modules/serialization/CMakeFiles/metaobject_serialization.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/serialization/CMakeFiles/metaobject_serialization.dir/depend

