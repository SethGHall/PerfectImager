
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

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <sys/time.h>

#include "inverse_direct_fourier_transform.h"

int main(int argc, char **argv)
{
    printf("=============================================================================\n");
	printf(">>> AUT HPC Research Laboratory - Inverse Direct Fourier Transform (CUDA) <<<\n");
	printf("=============================================================================\n\n");

	if(argc < 2)
	{
		printf("Error: double check you are providing yaml config file...\n\n");
		exit(EXIT_FAILURE);
	}

	//initialize config struct
	Config config;
	init_config_from_yaml(&config, argv[1]);

	PRECISION *h_image = (PRECISION*) calloc(config.render_size*config.render_size, sizeof(PRECISION));
	VIS_PRECISION2 *h_visibilities = (VIS_PRECISION2*) calloc(config.num_visibilities, sizeof(VIS_PRECISION2));
	PRECISION3 *h_uvw_coords = (PRECISION3*) calloc(config.num_uvw_coords, sizeof(PRECISION3));
	
	
	load_visibilities(&config, h_visibilities, h_uvw_coords);

	create_image(&config, h_image, h_visibilities, h_uvw_coords);

    // Clean up
    free(h_uvw_coords);
	free(h_visibilities);
	free(h_image);
	return EXIT_SUCCESS;
}
