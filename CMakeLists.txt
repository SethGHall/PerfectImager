
#! Base inverse direct fourier transform project
project(perfect_imager CXX CUDA)

#! Flags for the compiler
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#! Locate CUDA
find_package(CUDA REQUIRED)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11 -arch=sm_60 -Xptxas -v -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo 
)

#! C based YAML Parser - https://github.com/pantoniou/libfyaml
find_library(fyaml_loc NAMES libfyaml-0.5.a)
message(STATUS ">>> Location of C YAML parser lib: ${fyaml_loc}")
add_library(fyaml-0.5 STATIC IMPORTED)
set_target_properties(fyaml-0.5 PROPERTIES IMPORTED_LOCATION ${fyaml_loc})

#! build and link idft
cuda_add_executable(
	perfect_imager
	main.cpp
	src/inverse_direct_fourier_transform.cu
)
target_link_libraries(perfect_imager m fyaml-0.5)