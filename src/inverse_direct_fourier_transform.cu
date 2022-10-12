
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


//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

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "../inverse_direct_fourier_transform.h"

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

/*
 * Intialize the configuration struct - IMPORTANT observation parameters
 */
void init_config_from_yaml(Config *config, char* yaml_file)
{
	// Parse YAML document into config struct
	struct fy_document *fyd = NULL;
	fyd = fy_document_build_from_file(NULL, yaml_file);

	if(!fyd)
	{
		printf("Error: Unable to locate your yaml file...\n\n");
		exit(EXIT_FAILURE);
	}

	uint32_t num_receivers = 0;
	parse_config_attribute(fyd, "NUM_RECEIVERS", "%d", &num_receivers);
	parse_config_attribute(fyd, "IMAGE_SIZE", "%d", &(config->image_size));
	
	parse_config_attribute(fyd, "FREQUENCY_NUM_CHANNELS", "%d", &(config->num_channels));
	parse_config_attribute(fyd, "NUM_TIMESTEPS", "%d", &(config->num_timesteps));
	parse_config_attribute(fyd, "TIMESTEP_BATCH_SIZE", "%d", &(config->ts_batch_size));

	parse_config_attribute(fyd, "FOV_DEG", "%lf", &(config->fov_deg));
	parse_config_attribute(fyd, "FREQUENCY_START_HZ", "%lf", &(config->frequency_hz_start));
	parse_config_attribute(fyd, "FREQUENCY_BANDWIDTH", "%lf", &(config->frequency_bandwidth));
	parse_config_attribute_bool(fyd, "RIGHT_ASCENSION", "%d", &(config->right_ascension));
	parse_config_attribute(fyd, "VIS_INTENSITY_FILE", "%s", &(config->visibility_source_file));
	parse_config_attribute(fyd, "VIS_UVW_FILE", "%s", &(config->uvw_source_file));
	parse_config_attribute(fyd, "INPUT_DATA_PATH", "%s", &(config->data_input_path));
	parse_config_attribute(fyd, "OUTPUT_DATA_PATH", "%s", &(config->data_output_path));

	

	fy_document_destroy(fyd);

	config->render_size = config->image_size;
	config->x_render_offset = 0;
	config->y_render_offset = 0;
	config->num_baselines = (num_receivers*(num_receivers-1))/2;

	config->num_uvw_coords = config->num_baselines * config->num_timesteps;
	config->num_visibilities = config->num_uvw_coords * config->num_channels;
	config->cell_size_rads = asin(2.0 * sin(0.5 * config->fov_deg*PI/(180.0)) / (double)config->image_size);
}

void parse_config_attribute(struct fy_document *fyd, const char *attrib_name, const char *format, void* data)
{
	char buffer[MAX_CHARS];
	snprintf(buffer, MAX_CHARS, "/%s %s", attrib_name, format);

	if(fy_document_scanf(fyd, buffer, data))
	{
		printf("Successfully parsed attribute %s...\n\n", attrib_name);
	}
	else
	{
		printf("Error: unable to find attribute %s in yaml config file...\n\n", attrib_name);
		exit(EXIT_FAILURE);
	}
}

void parse_config_attribute_bool(struct fy_document *fyd, const char *attrib_name, const char *format, bool* data)
{
	int obtained = 0;
	parse_config_attribute(fyd, attrib_name, format, &obtained);
	*data = (obtained == 1) ? true : false;
}



void create_image(Config *config, PRECISION *h_image, VIS_PRECISION2 *h_visibilities, PRECISION3 *h_uvw_coords)
{
	printf(">>> UPDATE:  Configuring iDFT and allocating GPU memory for grid (image)...\n\n");


	uint32_t vis_batch_size = config->num_baselines * config->num_channels  * config->ts_batch_size;
	uint32_t uvw_batch_size = config->num_baselines * config->ts_batch_size;



	PRECISION *d_image;
	VIS_PRECISION2 *d_visibilities;
	PRECISION3 *d_uvw_coords;

	CUDA_CHECK_RETURN(cudaMalloc(&(d_visibilities), sizeof(VIS_PRECISION2) * vis_batch_size));
	CUDA_CHECK_RETURN(cudaMalloc(&(d_uvw_coords), sizeof(PRECISION3) * uvw_batch_size));
	CUDA_CHECK_RETURN(cudaMalloc(&d_image, sizeof(PRECISION) * config->render_size*config->render_size));


	uint32_t total_num_batches = (int)CEIL(double(config->num_timesteps) / double(config->ts_batch_size));
		
	uint32_t visLeftToProcess = config->num_visibilities;
	uint32_t uwvLeftToProcess = config->num_uvw_coords;


	PRECISION freq_inc = 0.0;
	if(config->num_channels > 1)
		freq_inc = PRECISION(config->frequency_bandwidth) / (config->num_channels-1); 



	int max_threads_per_block = min(1024, config->render_size*config->render_size);
	int num_blocks = (int) ceil((double) (config->render_size*config->render_size) / max_threads_per_block);
	printf("Max threads per block: %d\n", max_threads_per_block);
	printf("Num blocks: %d\n", num_blocks);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);





	for(uint32_t ts_batch=0;ts_batch<total_num_batches; ts_batch++)
	{	//copy vis UVW here
		//fill UVW and VIS
		printf("Gridding batch number %d of %d batches...\n",ts_batch, total_num_batches);

		uint32_t current_vis_batch_size = min(visLeftToProcess, vis_batch_size);
		uint32_t current_uvw_batch_size = min(uwvLeftToProcess, uvw_batch_size);
		printf("Number of Visibilities %d and UVW %d to process...\n",current_vis_batch_size, current_uvw_batch_size);


		CUDA_CHECK_RETURN(cudaMemcpy(d_visibilities, h_visibilities+(ts_batch*vis_batch_size), 
				sizeof(VIS_PRECISION2) * current_vis_batch_size, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_uvw_coords, h_uvw_coords+(ts_batch*uvw_batch_size), 
			   sizeof(PRECISION3) * current_uvw_batch_size, cudaMemcpyHostToDevice));

		//grid me.
		


		direct_imaging_with_w_correction<<<kernel_blocks, kernel_threads>>>(
 			d_image,
			config->image_size,
			config->cell_size_rads,
 			d_visibilities,
			current_vis_batch_size,
			d_uvw_coords,
 			current_uvw_batch_size,
			config->frequency_hz_start,
			freq_inc,
 			config->num_channels,
			config->num_baselines
		);



		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );


		visLeftToProcess -= current_vis_batch_size;
		uwvLeftToProcess -= current_uvw_batch_size;
		printf("Number of Visibilities %d and UVW %d remaining...\n\n",visLeftToProcess, uwvLeftToProcess);

	}

	//TODO  batch via timesteps!;




	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

    CUDA_CHECK_RETURN(cudaMemcpy(h_image, d_image, config->image_size*config->image_size * sizeof(PRECISION),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	CUDA_CHECK_RETURN(cudaFree(d_image));
	CUDA_CHECK_RETURN(cudaFree(d_visibilities));

	printf("UPDATE SAVING IMAGE TO FILE ");
	save_image_to_file(config, h_image);

}

__global__ void direct_imaging_with_w_correction(
	PRECISION *image, 
	const uint32_t image_size,
	const PRECISION cell_size_rads, 
	const VIS_PRECISION2 *visibilities,
	const uint32_t num_visibilities, 
	const PRECISION3 *uvw_coords,
	const uint32_t num_uvw_coords,
	const PRECISION frequency_hz_start, 
	const PRECISION bandwidth_increment,
	const uint32_t num_channels,
	const uint32_t num_baselines	
)
{
	uint32_t pixel_index = blockIdx.x*blockDim.x + threadIdx.x;
	if(pixel_index >= (image_size*image_size))
		return;

	PRECISION x = ((int32_t)(pixel_index % image_size) - (int32_t)image_size/2) * cell_size_rads;
    PRECISION y = ((int32_t)(pixel_index / image_size) - (int32_t)image_size/2) * cell_size_rads;

	PRECISION image_correction = SQRT(1.0f - (x * x) - (y * y));
	PRECISION w_correction = image_correction - 1.0f;

    PRECISION2 theta_complex = MAKE_PRECISION2(0.0, 0.0);
	PRECISION3 uvw_coord = MAKE_PRECISION3(0.0, 0.0, 0.0);

    PRECISION theta_vis_product = 0.0f;
	PRECISION sum = 0.0f;
	
	//TOdo update
	for(uint32_t v = 0; v < num_visibilities; v++)
	{
		//convert to meters_wavelengths
		uint32_t timeStepOffset = v/(num_channels*num_baselines);
		uint32_t uvwIndex = (v % num_baselines) + (timeStepOffset*num_baselines);
		PRECISION3 local_uvw = uvw_coords[uvwIndex];	
		//chnnelNum - visIndex 
		uint32_t channelNumber = (v / num_baselines) % num_channels; 
		
		PRECISION freqScale = (frequency_hz_start + channelNumber*bandwidth_increment) / PRECISION(SPEED_OF_LIGHT); //meters to wavelengths conversion
		local_uvw.x *= freqScale;
		local_uvw.y *= freqScale;
		local_uvw.z *= freqScale;

		PRECISION theta = 2.0f * PI * (x * local_uvw.x + y * local_uvw.y - (w_correction * local_uvw.z));
		SINCOS(theta, &(theta_complex.y), &(theta_complex.x));
		theta_vis_product = complex_to_real_mult(MAKE_PRECISION2(visibilities[v].x,visibilities[v].y), theta_complex);
		sum += theta_vis_product;

	}

	image[pixel_index] += (sum * image_correction);
}



void load_visibilities(Config *config, VIS_PRECISION2 *h_visibilities, PRECISION3 *h_uvw_coords)
{
	char vis_file_location[MAX_CHARS*2];
	snprintf(vis_file_location, MAX_CHARS*2, "%s%s", config->data_input_path, config->visibility_source_file);
	char uvw_file_location[MAX_CHARS*2];
	snprintf(uvw_file_location, MAX_CHARS*2, "%s%s", config->data_input_path, config->uvw_source_file);

	FILE *vis_file = fopen(vis_file_location, "rb");
	FILE *uvw_file = fopen(uvw_file_location, "rb");

	uint32_t num_vis_header = 0;
    fread(&num_vis_header, sizeof(uint32_t), 1, vis_file);
	uint32_t num_uvw_header = 0;
    fread(&num_uvw_header, sizeof(uint32_t), 1, uvw_file);

    if(num_vis_header < config->num_visibilities || num_uvw_header < config->num_uvw_coords)
    {
    	printf("Error: the file headers for visibilities/uvw coords does not match calculations based on yaml file...\n\n");
    	exit(EXIT_FAILURE);
    }
    
    if(num_vis_header > config->num_visibilities || num_uvw_header > config->num_uvw_coords)
    {
    	printf("Warning: the file headers for visibilities/uvw coords are greater than the amount of calculated vis/uvw...\n\n");
    }

    uint32_t num_vis_read = fread(h_visibilities, sizeof(VIS_PRECISION2), config->num_visibilities, vis_file);
    uint32_t num_uvw_read = fread(h_uvw_coords, sizeof(PRECISION3), config->num_uvw_coords, uvw_file);

    if(num_vis_read != config->num_visibilities || num_uvw_read != config->num_uvw_coords)
    {
    	printf("Error: number of visibilities read from file failed, check your precision types...\n\n");
    	exit(EXIT_FAILURE);
    }

    // PRECISION right_ascension = (PRECISION)(config->right_ascension ? -1.0 : 1.0);
     for(uint32_t coord = 0; coord < config->num_uvw_coords; coord++)
     {
     	//h_uvw_coords[coord].x *= -1.0;
     	//h_uvw_coords[coord].z *= -1.0;
     }

	fclose(vis_file);
	fclose(uvw_file);

	// TESTING...
	// for(int i = 0; i < 10; i++)
	// {
	// 	printf("UVW => Vis: (%f %f %f) => %f %f\n", h_uvw_coords[i].x, h_uvw_coords[i].y, h_uvw_coords[i].z,
	// 		h_visibilities[i].x, h_visibilities[i].y);
	// }
}

// saves a csv file of the rendered iDFT region only
void save_image_to_file(Config *config, PRECISION *h_image)
{
	char buffer[MAX_CHARS * 2];
	snprintf(buffer, MAX_CHARS*2, "%sdirect_image.bin", config->data_output_path);

	FILE *f = fopen(buffer, "wb");
	int saved = fwrite(h_image, sizeof(PRECISION), config->image_size * config->image_size, f);
	printf(">>> GRID DIMS IS : %d\n", config->image_size);
	printf(">>> SAVED TO FILE: %d\n", saved);
    fclose(f);
}



/*
// visibilities (x,y,z) = (uu,vv,ww), visIntensity (x,y) = (real, imaginary), grid (x,y) = (real, imaginary)
__global__ void inverse_dft_with_w_correction(double2 *grid, size_t grid_pitch, const double3 *visibilities,
		const double2 *vis_intensity, int vis_count, int batch_count, int x_offset, int y_offset,
		int render_size, double cell_size)
{
	// look up id of thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	if(idx >= render_size || idy >= render_size)
		return;

	double real_sum = 0;
	double imag_sum = 0;

	// convert to x and y image coordinates
	double x = (idx+x_offset) * cell_size;
    double y = (idy+y_offset) * cell_size;

	double2 vis;
	double2 theta_complex = make_double2(0.0, 0.0);

	// precalculate image correction and wCorrection
	double image_correction = sqrt(1.0 - (x * x) - (y * y));
	double w_correction = image_correction - 1.0;

	// NOTE: below is an approximation... Uncomment if needed
	// double wCorrection = -((x*x)+(y*y))/2.0;

	// loop through all visibilities and create sum using iDFT formula
	for(int i = 0; i < batch_count; ++i)
	{	
		double theta = 2.0 * M_PI * (x * visibilities[i].x + y * visibilities[i].y
				+ (w_correction * visibilities[i].z));
		sincos(theta, &(theta_complex.y), &(theta_complex.x));
		vis = complex_multiply(vis_intensity[i], theta_complex);
		real_sum += vis.x;
		imag_sum += vis.y;
	}

	// adjust sum by image correction
	real_sum *= image_correction;
	imag_sum *= image_correction;

	// look up destination in image (grid) and divide by amount of visibilities (N)
	double2 *row = (double2*)((char*)grid + idy * grid_pitch);
	row[idx].x += (real_sum / vis_count);
	row[idx].y += (imag_sum / vis_count);
}
*/

// done on GPU, performs a complex multiply of two complex numbers
__device__ PRECISION complex_to_real_mult(PRECISION2 z1, PRECISION2 z2)
{
    return z1.x*z2.x - z1.y*z2.y;
}

// used for performance testing to return the difference in milliseconds between two timeval structs
float time_difference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}
