# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/joy_hsiao/Haitec/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/joy_hsiao/Haitec/caffe/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/ssd_detect.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/ssd_detect.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/ssd_detect.dir/flags.make

examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o: examples/CMakeFiles/ssd_detect.dir/flags.make
examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o: ../examples/ssd/ssd_detect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/joy_hsiao/Haitec/caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o"
	cd /home/joy_hsiao/Haitec/caffe/build/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o -c /home/joy_hsiao/Haitec/caffe/examples/ssd/ssd_detect.cpp

examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.i"
	cd /home/joy_hsiao/Haitec/caffe/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/joy_hsiao/Haitec/caffe/examples/ssd/ssd_detect.cpp > CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.i

examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.s"
	cd /home/joy_hsiao/Haitec/caffe/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/joy_hsiao/Haitec/caffe/examples/ssd/ssd_detect.cpp -o CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.s

examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.requires:

.PHONY : examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.requires

examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.provides: examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/ssd_detect.dir/build.make examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.provides.build
.PHONY : examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.provides

examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.provides.build: examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o


# Object files for target ssd_detect
ssd_detect_OBJECTS = \
"CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o"

# External object files for target ssd_detect
ssd_detect_EXTERNAL_OBJECTS =

examples/ssd/ssd_detect: examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o
examples/ssd/ssd_detect: examples/CMakeFiles/ssd_detect.dir/build.make
examples/ssd/ssd_detect: lib/libcaffe.so.1.0.0-rc3
examples/ssd/ssd_detect: lib/libproto.a
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_regex.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libglog.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libprotobuf.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libsz.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libz.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libdl.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libm.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libglog.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libprotobuf.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libsz.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libz.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libdl.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libm.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/liblmdb.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libleveldb.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libsnappy.so
examples/ssd/ssd_detect: /usr/local/cuda-9.0/lib64/libcudart.so
examples/ssd/ssd_detect: /usr/local/cuda-9.0/lib64/libcurand.so
examples/ssd/ssd_detect: /usr/local/cuda-9.0/lib64/libcublas.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libcudnn.so
examples/ssd/ssd_detect: /usr/local/lib/libopencv_highgui.so.3.4.0
examples/ssd/ssd_detect: /usr/local/lib/libopencv_videoio.so.3.4.0
examples/ssd/ssd_detect: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
examples/ssd/ssd_detect: /usr/local/lib/libopencv_imgproc.so.3.4.0
examples/ssd/ssd_detect: /usr/local/lib/libopencv_core.so.3.4.0
examples/ssd/ssd_detect: /usr/lib/liblapack.so
examples/ssd/ssd_detect: /usr/lib/libcblas.so
examples/ssd/ssd_detect: /usr/lib/libatlas.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libpython2.7.so
examples/ssd/ssd_detect: /usr/lib/x86_64-linux-gnu/libboost_python.so
examples/ssd/ssd_detect: examples/CMakeFiles/ssd_detect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/joy_hsiao/Haitec/caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ssd/ssd_detect"
	cd /home/joy_hsiao/Haitec/caffe/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ssd_detect.dir/link.txt --verbose=$(VERBOSE)
	cd /home/joy_hsiao/Haitec/caffe/build/examples && ln -sf /home/joy_hsiao/Haitec/caffe/build/examples/ssd/ssd_detect /home/joy_hsiao/Haitec/caffe/build/examples/ssd/ssd_detect.bin

# Rule to build all files generated by this target.
examples/CMakeFiles/ssd_detect.dir/build: examples/ssd/ssd_detect

.PHONY : examples/CMakeFiles/ssd_detect.dir/build

examples/CMakeFiles/ssd_detect.dir/requires: examples/CMakeFiles/ssd_detect.dir/ssd/ssd_detect.cpp.o.requires

.PHONY : examples/CMakeFiles/ssd_detect.dir/requires

examples/CMakeFiles/ssd_detect.dir/clean:
	cd /home/joy_hsiao/Haitec/caffe/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/ssd_detect.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/ssd_detect.dir/clean

examples/CMakeFiles/ssd_detect.dir/depend:
	cd /home/joy_hsiao/Haitec/caffe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/joy_hsiao/Haitec/caffe /home/joy_hsiao/Haitec/caffe/examples /home/joy_hsiao/Haitec/caffe/build /home/joy_hsiao/Haitec/caffe/build/examples /home/joy_hsiao/Haitec/caffe/build/examples/CMakeFiles/ssd_detect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/ssd_detect.dir/depend
