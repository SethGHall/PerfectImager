# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/seth/perfectimager/PerfectImager

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/seth/perfectimager/PerfectImager/build

# Include any dependencies generated for this target.
include CMakeFiles/perfect_imager.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/perfect_imager.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/perfect_imager.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/perfect_imager.dir/flags.make

CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o: CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o.depend
CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o: CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o.cmake
CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o: ../src/inverse_direct_fourier_transform.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/seth/perfectimager/PerfectImager/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o"
	cd /home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src && /usr/bin/cmake -E make_directory /home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src/.
	cd /home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src/./perfect_imager_generated_inverse_direct_fourier_transform.cu.o -D generated_cubin_file:STRING=/home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src/./perfect_imager_generated_inverse_direct_fourier_transform.cu.o.cubin.txt -P /home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o.cmake

CMakeFiles/perfect_imager.dir/main.cpp.o: CMakeFiles/perfect_imager.dir/flags.make
CMakeFiles/perfect_imager.dir/main.cpp.o: ../main.cpp
CMakeFiles/perfect_imager.dir/main.cpp.o: CMakeFiles/perfect_imager.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/seth/perfectimager/PerfectImager/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/perfect_imager.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/perfect_imager.dir/main.cpp.o -MF CMakeFiles/perfect_imager.dir/main.cpp.o.d -o CMakeFiles/perfect_imager.dir/main.cpp.o -c /home/seth/perfectimager/PerfectImager/main.cpp

CMakeFiles/perfect_imager.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perfect_imager.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/seth/perfectimager/PerfectImager/main.cpp > CMakeFiles/perfect_imager.dir/main.cpp.i

CMakeFiles/perfect_imager.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perfect_imager.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/seth/perfectimager/PerfectImager/main.cpp -o CMakeFiles/perfect_imager.dir/main.cpp.s

# Object files for target perfect_imager
perfect_imager_OBJECTS = \
"CMakeFiles/perfect_imager.dir/main.cpp.o"

# External object files for target perfect_imager
perfect_imager_EXTERNAL_OBJECTS = \
"/home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o"

CMakeFiles/perfect_imager.dir/cmake_device_link.o: CMakeFiles/perfect_imager.dir/main.cpp.o
CMakeFiles/perfect_imager.dir/cmake_device_link.o: CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o
CMakeFiles/perfect_imager.dir/cmake_device_link.o: CMakeFiles/perfect_imager.dir/build.make
CMakeFiles/perfect_imager.dir/cmake_device_link.o: /usr/local/cuda-11.7/lib64/libcudart_static.a
CMakeFiles/perfect_imager.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/librt.a
CMakeFiles/perfect_imager.dir/cmake_device_link.o: /usr/local/lib/libfyaml-0.5.a
CMakeFiles/perfect_imager.dir/cmake_device_link.o: CMakeFiles/perfect_imager.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/seth/perfectimager/PerfectImager/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/perfect_imager.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/perfect_imager.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/perfect_imager.dir/build: CMakeFiles/perfect_imager.dir/cmake_device_link.o
.PHONY : CMakeFiles/perfect_imager.dir/build

# Object files for target perfect_imager
perfect_imager_OBJECTS = \
"CMakeFiles/perfect_imager.dir/main.cpp.o"

# External object files for target perfect_imager
perfect_imager_EXTERNAL_OBJECTS = \
"/home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o"

perfect_imager: CMakeFiles/perfect_imager.dir/main.cpp.o
perfect_imager: CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o
perfect_imager: CMakeFiles/perfect_imager.dir/build.make
perfect_imager: /usr/local/cuda-11.7/lib64/libcudart_static.a
perfect_imager: /usr/lib/x86_64-linux-gnu/librt.a
perfect_imager: /usr/local/lib/libfyaml-0.5.a
perfect_imager: CMakeFiles/perfect_imager.dir/cmake_device_link.o
perfect_imager: CMakeFiles/perfect_imager.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/seth/perfectimager/PerfectImager/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable perfect_imager"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/perfect_imager.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/perfect_imager.dir/build: perfect_imager
.PHONY : CMakeFiles/perfect_imager.dir/build

CMakeFiles/perfect_imager.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/perfect_imager.dir/cmake_clean.cmake
.PHONY : CMakeFiles/perfect_imager.dir/clean

CMakeFiles/perfect_imager.dir/depend: CMakeFiles/perfect_imager.dir/src/perfect_imager_generated_inverse_direct_fourier_transform.cu.o
	cd /home/seth/perfectimager/PerfectImager/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/seth/perfectimager/PerfectImager /home/seth/perfectimager/PerfectImager /home/seth/perfectimager/PerfectImager/build /home/seth/perfectimager/PerfectImager/build /home/seth/perfectimager/PerfectImager/build/CMakeFiles/perfect_imager.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/perfect_imager.dir/depend
