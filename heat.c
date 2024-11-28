#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "mpi.h"

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simulation parameters
static const unsigned int N = 500;

static const float SOURCE_TEMP = 5000.0f;
static const float BOUNDARY_TEMP = 1000.0f;

static const float MIN_DELTA = 0.05f;
static const unsigned int MAX_ITERATIONS = 20000;


static unsigned int idx(unsigned int x, unsigned int y, unsigned int stride) {
    return y * stride + x;
}

// Print matrix
static void print_matrix(float * matrix) {
	for (unsigned int y = 0; y < N; ++y) {
		printf("rank %d: \n", y);
		for (unsigned int x = 0; x < N; ++x) {
			printf("%f ", matrix[idx(x, y, N)]);
		}
		printf("\n");
	}
}

// print local matrix
static void print_local_matrix(float * matrix, unsigned int local_N) {
	for (unsigned int y = 0; y < local_N + 2; ++y) {
		printf("rank %d: \n", y);
		for (unsigned int x = 0; x < N; ++x) {
			printf("%f ", matrix[idx(x, y, N)]);
		}
		printf("\n");
	}
}



static void init(unsigned int source_x, unsigned int source_y, float * matrix) {
	// init
	memset(matrix, 0, N * N * sizeof(float));

	// place source
	matrix[idx(source_x, source_y, N)] = SOURCE_TEMP;

	// fill borders
	for (unsigned int x = 0; x < N; ++x) {
		matrix[idx(x, 0,   N)] = BOUNDARY_TEMP;
		matrix[idx(x, N-1, N)] = BOUNDARY_TEMP;
	}
	for (unsigned int y = 0; y < N; ++y) {
		matrix[idx(0,   y, N)] = BOUNDARY_TEMP;
		matrix[idx(N-1, y, N)] = BOUNDARY_TEMP;
	}
}


static void step(unsigned int source_x, unsigned int source_y, const float * local_current, float * local_next, unsigned int local_N, int world_rank, int world_size) {
    unsigned int start_y = (world_rank == 0) ? 2 : 1; // Si es el primer proceso, empieza en la fila 2 (evita fila 1)
    unsigned int end_y = (world_rank == world_size - 1) ? local_N - 1 : local_N; // Si es el último proceso, detente en local_N-1

    for (unsigned int y = start_y; y <= end_y; ++y) { // Solo filas locales reales
        for (unsigned int x = 1; x < N - 1; ++x) {
			unsigned int global_y = y-1 + world_rank * (N/world_size);
            if (global_y == source_y && x == source_x) { // Verificar si el origen está en este proceso
                continue;
            }
            local_next[idx(x, y, N)] = (
                local_current[idx(x, y - 1, N)] +
                local_current[idx(x - 1, y, N)] +
                local_current[idx(x + 1, y, N)] +
                local_current[idx(x, y + 1, N)]
            ) / 4.0f;
        }
    }
}


static float diff(const float * current, const float * next, unsigned int local_N, int world_rank, int world_size) {
    float maxdiff = 0.0f;

    // Determinar las filas que deben ser evaluadas en cada proceso
    unsigned int start_y = (world_rank == 0) ? 2 : 1;  // Para el primer proceso, comienza en la fila 2
    unsigned int end_y = (world_rank == world_size - 1) ? local_N - 1 : local_N;  // El último proceso se detiene en local_N - 1

    for (unsigned int y = start_y; y <= end_y; ++y) { // Solo filas locales reales
        for (unsigned int x = 1; x < N - 1; ++x) {
            maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
        }
    }

    return maxdiff;
}


void write_png(float * current, int iter) {
	char file[100];
	uint8_t * image = malloc(3 * N * N * sizeof(uint8_t));
	float maxval = fmaxf(SOURCE_TEMP, BOUNDARY_TEMP);

	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 0; x < N; ++x) {
			unsigned int i = idx(x, y, N);
			colormap_rgb(COLORMAP_MAGMA, current[i], 0.0f, maxval, &image[3*i], &image[3*i + 1], &image[3*i + 2]);
		}
	}
	sprintf(file,"heat%i.png", iter);
	stbi_write_png(file, N, N, 3, image, 3 * N);

	free(image);
}


int main() {
	MPI_Init(NULL, NULL);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	unsigned int source_x;
	unsigned int source_y;
	
	size_t array_size;

	float * current;
	float * next;
	if (world_rank == 0)
	{
		array_size = N * N * sizeof(float);
		current = malloc(array_size);
		next = malloc(array_size);
		srand(0);
		source_x = rand() % (N-2) + 1;
		source_y = rand() % (N-2) + 1;
		init(source_x, source_y, current);
		memcpy(next, current, array_size);
		printf("source value: %f\n", current[idx(source_x, source_y, N)]);
	}
	MPI_Bcast(&source_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&source_y, 1, MPI_INT, 0, MPI_COMM_WORLD);

	printf("Heat source at (%u, %u)\n", source_x, source_y);

	


	unsigned int local_N = N/world_size;
	unsigned int local_start = world_rank*local_N;
	float * local_current = malloc(((local_N + 2) * N) * sizeof(float));
	float * local_next = malloc(((local_N + 2) * N) * sizeof(float));
	// Scatter the data
	MPI_Scatter(current, local_N * N, MPI_FLOAT, local_current + N, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(next, local_N * N, MPI_FLOAT, local_next + N, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	double start = omp_get_wtime();
	float global_t_diff = SOURCE_TEMP;
	for (unsigned int it = 0; (it < MAX_ITERATIONS) && (global_t_diff > MIN_DELTA); ++it) {
		// Exchange the data
		if (world_rank > 0) {
			MPI_Send(local_current + N, N, MPI_FLOAT, world_rank - 1, 0, MPI_COMM_WORLD);
			MPI_Recv(local_current, N, MPI_FLOAT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		if (world_rank < world_size - 1) {
			MPI_Send(local_current + local_N * N, N, MPI_FLOAT, world_rank + 1, 0, MPI_COMM_WORLD);
			MPI_Recv(local_current + (local_N + 1) * N, N, MPI_FLOAT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		
		step(source_x, source_y, local_current, local_next, local_N, world_rank, world_size);
		float local_t_diff =  diff(local_current, local_next, local_N, world_rank, world_size);

		if(it%(MAX_ITERATIONS/10)==0){
			printf("%u: %f\n", it, local_t_diff);
		}
		float * swap = local_current;
		local_current = local_next;
		local_next = swap;
		//Get Minimum of all the local_t_diff
		MPI_Allreduce(&local_t_diff, &global_t_diff, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	}

	// Gather the data
	MPI_Gather(local_current + N, local_N * N, MPI_FLOAT, current, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	double stop = omp_get_wtime();
	if(world_rank == 0) {
	printf("Computing time %f s.\n", stop-start);
	write_png(current, MAX_ITERATIONS);
	free(current);
	free(next);
	}

	
	MPI_Finalize();
	return 0;
}
