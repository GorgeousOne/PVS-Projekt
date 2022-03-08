// Aaron Kammer 122461
// David Krug 122427

#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include <omp.h>
#include <mpi.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


void checkErr(cl_int const &err, char const *message) {
	if (CL_SUCCESS != err) {
		printf(message, err);
		printf("\nError: %d\n", err);
		exit(0);
	}
}

unsigned int get_nvidia_platform(cl_platform_id const *platforms, cl_uint num_of_platforms) {
	char platform_name[1024];

	//Gibt die ID der Plattform mit NVIDIA im Namen zurück
	for (unsigned int i = 0; i < num_of_platforms; i++) {
		//Erfragt Informationen über alle verfügbaren Plattformen
		cl_int err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
		checkErr(err, "Could not get information about platform.");
		//Überprüft ob NVIDIA im Plattformnamen enthalten ist
		if (strstr(platform_name, "NVIDIA") != NULL) {
			return i;
		}
	}
	return 0;
}

int read_source_from_file(const char *fileName, char **source, size_t *sourceSize) {
	//Liest Kernel source von einem externen Dokument
	FILE *fp = NULL;
	fp = fopen(fileName, "rb");

	if (fp == NULL) {
		printf("Error: Couldn't find program source file.");
		return CL_INVALID_VALUE;
	}
	fseek(fp, 0, SEEK_END);
	*sourceSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	*source = (char *) malloc(sizeof(char) * (*sourceSize));

	if (*source == NULL) {
		printf("Error: Couldn't allocate for program source.");
		return CL_OUT_OF_HOST_MEMORY;
	}
	fread(*source, 1, *sourceSize, fp);
	return CL_SUCCESS;
}

template<class T>
T **alloc_mat(int cols, int rows) {
	T **A1, *A2;
	A1 = (T **) calloc(cols, sizeof(T *));     // pointer on columns
	A2 = (T *) calloc(rows * cols, sizeof(T));    // all matrix elements
	for (int x = 0; x < cols; ++x) {
		A1[x] = A2 + x * rows;
	}
	return A1;
}

template<class T>
void print_mat(T **A, int cols, int rows, char const *tag) {
	int y, x;

	printf("Matrix: %s\n", tag);
	for (y = 0; y < rows; y++) {
		for (x = 0; x < cols; x++) {
			printf("%i  ", A[x][y]);
		}
		printf("\n");
	}
}

void free_mat(float **A) {
	free(A[0]); // free contiguous block of float elements (row*col floats)
	free(A);    // free memory for pointers pointing to the beginning of each row
}

// https://stackoverflow.com/questions/61410931/write-a-c-program-to-convert-1d-array-to-2d-array-using-pointers Besucht: 03.03.2022
void array_to_matrix(unsigned char **matrix, const unsigned char *arr, int cols, int rows) {
	int k = 0;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			matrix[x][y] = arr[k++];
		}
	}
}

void print_path(char *file_name) {
	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		printf("File path: %s%s%s\n", cwd, "/", file_name);
	} else {
		perror("getcwd() error");
	}
}

/** **/
int main(int argc, char **argv) {
	cl_int err;
	cl_platform_id *platforms = NULL;
	cl_device_id device_id = NULL;
	cl_uint num_of_platforms = 0;
	cl_uint num_of_devices = 0;
	cl_context context;
	cl_kernel kernel;
	cl_command_queue command_queue;
	cl_program program;

	cl_mem img_buffer, patch_buffer, result_buffer, dims_buffer;

	char *const img_path = argv[1];
	char *const patch_path = argv[2];

	int img_w = 0;
	int img_h = 0;
	int img_c = 0; // number of image channels
	int desired_c = 1;

	unsigned char *img = NULL;
	//load RGB image as 1 channel grayscale image (1x unsigned 8 bit per pixel)
	img = stbi_load(img_path, &img_w, &img_h, &img_c, desired_c);
	printf("\nLoaded image: %s\n", (img != NULL ? "true" : "false"));
	print_path(img_path);
	printf("\twidth: %d\n", img_w);
	printf("\theight: %d\n", img_h);

	int patch_w = 0;
	int patch_h = 0;
	int patch_c = 0;
	unsigned char *patch = NULL;
	patch = stbi_load(patch_path, &patch_w, &patch_h, &patch_c, desired_c);
	printf("\nLoaded patch: %s\n", (patch != NULL ? "true" : "false"));
	print_path(patch_path);
	printf("\twidth: %d\n", patch_w);
	printf("\theight: %d\n", patch_h);

	int result_w = (img_w - patch_w);
	int result_h = (img_h - patch_h);

	unsigned char **img2d = alloc_mat<unsigned char>(img_w, img_h);
	unsigned char **patch2d = alloc_mat<unsigned char>(patch_w, patch_h);
	int **result2d = alloc_mat<int>(result_w, result_h);

	array_to_matrix(img2d, img, img_w, img_h);
	array_to_matrix(patch2d, patch, patch_w, patch_h);

	size_t global[2] = {
			static_cast<size_t>(result_w),
			static_cast<size_t>(result_h)};

	int heights[4] {img_w, img_h, patch_w, patch_h};
	auto start = std::chrono::steady_clock::now();

	/* 1) */
	//Speichert die Anzahl der Plattformen in num_of_platforms
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	checkErr(err, "No platforms found.");

	//Ruft die IDs der Plattformen ab
	platforms = (cl_platform_id *) malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	checkErr(err, "No platforms found.");

	unsigned int nvidia_platform = get_nvidia_platform(platforms, num_of_platforms);
	//Erfragt ID's von verfügbaren Geräten
	err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
	checkErr(err, "Could not get device in platform.");

	//Erstellt einen Kontext für OpenCL Programm
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	checkErr(err, "Unable to create context.");

	//Erstellt eine Befehlswarteschlange in diesem Kontext mit FIFO Prinzip
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	checkErr(err, "Unable to create command queue.");

	char *source = NULL;
	size_t src_size = 0;
	err = read_source_from_file("./kernel_source.cl", &source, &src_size);
	checkErr(err, "Unable to load kernel source.");

	//Erstellt ein Programm mit dem Quellcode für den Kernel
	program = clCreateProgramWithSource(context, 1, (const char **) &source, &src_size, &err);
	free(source);
	checkErr(err, "Unable to create program.");

	//Kompiliert und linkt den Kernel Quellcode
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	checkErr(err, "Error building program.");

	//Definiert Kernel Einsprungspunkt
	kernel = clCreateKernel(program, "match_patch", &err);
	checkErr(err, "Error setting kernel.");

	/* 2) */
	//Erstellt Buffer für input und output
	size_t img_mem_size = sizeof(unsigned char) * img_w * img_h;
	size_t patch_mem_size = sizeof(unsigned char) * patch_w * patch_h;
	size_t result_mem_size = sizeof(int) * result_w * result_h;
	size_t heights_mem_size = sizeof(int) * 4;

	img_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, img_mem_size, NULL, &err);
	patch_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, patch_mem_size, NULL, &err);
	result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, result_mem_size, NULL, &err);
	dims_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, heights_mem_size, NULL, &err);

	//Kopiert zusammenhängende Daten aus data in den Eingabe-Puffer
	clEnqueueWriteBuffer(command_queue, img_buffer, CL_TRUE, 0, img_mem_size, img2d[0], 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, patch_buffer, CL_TRUE, 0, patch_mem_size, patch2d[0], 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, dims_buffer, CL_TRUE, 0, heights_mem_size, heights, 0, NULL, NULL);

	//Definiert Reihenfolge der Kernel-Argumente
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &patch_buffer);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dims_buffer);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &result_buffer);

	/* 3)  */
	//Einreihen des Kerns in die Befehlswarteschlange und Aufteilungsbereich angeben
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
	//Auf die Beendigung der Operation warten
	clFinish(command_queue);
	//Kopiere die Ergebnisse vom Ausgabe-Puffer 'output' in das Ergebnisfeld 'results'
	clEnqueueReadBuffer(command_queue, result_buffer, CL_TRUE, 0, result_mem_size, result2d[0], 0, NULL, NULL);

//	print_mat(img2d, 10, 10, "original");
//	print_mat(result2d, 10, 10, "result");

	int max_correlation = 999999;
	int max_x = -1;
	int max_y = -1;
	for (int j = 0; j < result_h; j++) {
		for (int i = 0; i < result_w; i++) {
			if (result2d[i][j] < max_correlation) {
				max_correlation = result2d[i][j];
				max_x = i;
				max_y = j;
			}
		}
	}
	printf("Max correlation: %i, %i\n", max_x, max_y, max_correlation);

	/* 4) */
	clReleaseMemObject(img_buffer);
	clReleaseMemObject(patch_buffer);
	clReleaseMemObject(result_buffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	auto finish = std::chrono::steady_clock::now();
	auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
	printf("Execution time parallel: %d ms\n", delta);
	return 0;
}