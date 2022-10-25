
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifdef __cplusplus
extern "C" {
#endif

#ifndef INVERSE_DIRECT_FOURIER_TRANSFORM_H_
#define INVERSE_DIRECT_FOURIER_TRANSFORM_H_

#include <ctime>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <libfyaml.h>


#define MAX_CHARS 512

//define function for checking CUDA errors
#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

// Speed of light
#ifndef C
	#define SPEED_OF_LIGHT 299792458.0
#endif



#ifndef HALF
    #define HALF 0
#endif

#ifndef SINGLE
    #define SINGLE 1
#endif

#ifndef DOUBLE
    #define DOUBLE 2
#endif

#ifndef COMPUTE_PRECISION
    #define COMPUTE_PRECISION SINGLE
    //#define COMPUTE_PRECISION DOUBLE
#endif

#ifndef VISIBILITY_PRECISION
	//#define VISIBILITY_PRECISION HALF
    #define VISIBILITY_PRECISION SINGLE
    //#define VISIBILITY_PRECISION DOUBLE
#endif

#if COMPUTE_PRECISION == SINGLE
    #define PRECISION float
    #define PRECISION2 float2
    #define PRECISION3 float3
    #define PRECISION4 float4
    #define PRECISION_MAX FLT_MAX
    #define PI ((float) 3.141592654)
    #define CUFFT_C2C_PLAN CUFFT_C2C
    #define CUFFT_C2P_PLAN CUFFT_C2R
#else // COMPUTE_PRECISION == DOUBLE
    #define PRECISION double
    #define PRECISION2 double2
    #define PRECISION3 double3
    #define PRECISION4 double4
    #define PRECISION_MAX DBL_MAX
    #define PI ((double) 3.1415926535897931)
    #define CUFFT_C2C_PLAN CUFFT_Z2Z
    #define CUFFT_C2P_PLAN CUFFT_Z2D
#endif

#if COMPUTE_PRECISION == SINGLE
    #define SIN(x) sinf(x)
    #define COS(x) cosf(x)
    #define SINCOS(x, y, z) sincosf(x, y, z)
    #define ABS(x) fabsf(x)
    #define SQRT(x) sqrtf(x)
    #define ROUND(x) roundf(x)
    #define CEIL(x) ceilf(x)
    #define LOG(x) logf(x)
    #define POW(x, y) powf(x, y)
    #define FLOOR(x) floorf(x)
    #define MAKE_PRECISION2(x,y) make_float2(x,y)
    #define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
    #define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)
    #define CUFFT_EXECUTE_C2P(a,b,c) cufftExecC2R(a,b,c)
    #define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecC2C(a,b,c,d)
#else // COMPUTE_PRECISION == DOUBLE
    #define SIN(x) sin(x)
    #define COS(x) cos(x)
    #define SINCOS(x, y, z) sincos(x, y, z)
    #define ABS(x) fabs(x)
    #define SQRT(x) sqrt(x)
    #define ROUND(x) round(x)
    #define CEIL(x) ceil(x)
    #define LOG(x) log(x)
    #define POW(x, y) pow(x, y)
    #define FLOOR(x) floor(x)
    #define MAKE_PRECISION2(x,y) make_double2(x,y)
    #define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
    #define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)
    #define CUFFT_EXECUTE_C2P(a,b,c) cufftExecZ2D(a,b,c)
    #define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecZ2Z(a,b,c,d)
#endif

// Defines the precision of visibility buffers and computation
#if VISIBILITY_PRECISION == HALF
    #define VIS_PRECISION __half
    #define VIS_PRECISION2 __half2
    #define MAKE_VIS_PRECISION2(x,y) make_half2(x,y)
    #define VEXP(x)  hexp(x)
    #define VSQRT(x) hsqrt(x)
    #define VDISP(x) __half2float(x)
#elif VISIBILITY_PRECISION == SINGLE
    #define VIS_PRECISION float
    #define VIS_PRECISION2 float2
    #define MAKE_VIS_PRECISION2(x,y) make_float2(x,y)
    #define VEXP(x)  expf(x)
    #define VSQRT(x) sqrtf(x)
    #define VDISP(x) (x)
#else // VISIBILITY_PRECISION == DOUBLE
    #define VIS_PRECISION double
    #define VIS_PRECISION2 double2
    #define MAKE_VIS_PRECISION2(x,y) make_double2(x,y)
    #define VEXP(x)  exp(x)
    #define VSQRT(x) sqrt(x)
    #define VDISP(x) (x)
#endif

// OLD GARBAGE BELOW
//Define struct for the configuration
typedef struct Config {

	uint32_t num_visibilities;
    uint32_t num_uvw_coords;
	uint32_t image_size;
	uint32_t render_size;
	int32_t x_render_offset;
	int32_t y_render_offset;
	uint32_t num_channels;
	uint32_t num_baselines;
	uint32_t num_timesteps;
	uint32_t ts_batch_size;
	double fov_deg;
	double cell_size_rads;
	double frequency_hz_start;
	double frequency_bandwidth;
	bool right_ascension;
	char visibility_source_file[MAX_CHARS];
	char uvw_source_file[MAX_CHARS];
	char data_input_path[MAX_CHARS];
	char data_output_path[MAX_CHARS];

} Config;

static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

void init_config_from_yaml(Config *config, char* yaml_file);

void parse_config_attribute(struct fy_document *fyd, const char *attrib_name, const char *format, void* data);

void parse_config_attribute_bool(struct fy_document *fyd, const char *attrib_name, const char *format, bool* data);

void load_visibilities(Config *config, VIS_PRECISION2 *h_visibilities, PRECISION3 *h_uvw_coords);

void save_image_to_file(Config *config, PRECISION *h_image);


void create_image(Config *config, PRECISION *h_image, VIS_PRECISION2 *h_visibilities, PRECISION3 *h_uvw_coords);


__global__ void direct_imaging_with_w_correction(
	PRECISION *image, 
	const uint32_t render_size,
	const int32_t x_offset,
	const int32_t y_offset,
	const PRECISION cell_size_rads, 
	const VIS_PRECISION2 *visibilities,
	const uint32_t num_visibilities, 
	const PRECISION3 *d_uvw_coords,
	const uint32_t num_uvw_coords,
	const PRECISION frequency_hz_start, 
	const PRECISION bandwidth_increment,
	const uint32_t num_channels,
    const uint32_t num_baselines	
);

__device__ PRECISION complex_to_real_mult(PRECISION2 z1, PRECISION2 z2);

float time_difference_msec(struct timeval t0, struct timeval t1);


#endif /* INVERSE_DIRECT_FOURIER_TRANSFORM_H_ */

#ifdef __cplusplus
}
#endif

