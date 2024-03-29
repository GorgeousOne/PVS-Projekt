// Aaron Kammer 122461
// David Krug 122427

#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// berechnet die absolute Differenz von Helligkeiten des Bildes und des Templates und gibt die Summe zurück
int calc_pixels_abs_a_minus_b_sum(unsigned char **img_a, unsigned char **patch_b, int w, int h, int img_x, int img_y) {
	int pixel_sum = 0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			pixel_sum += abs(img_a[img_x + x][img_y + y] - patch_b[x][y]);
		}
	}
	return pixel_sum;
}

// führt das template matching aus und schreibt die Koordinaten mit der größten Korrelation in den result Array
void match_patch(unsigned char** img, int img_w, int img_h, unsigned char** patch, int patch_w, int patch_h, int result[3]) {
	int max_correlation = 9999999;
	int correlation_x = -1;
	int correlation_y = -1;

	for (int y = 0; y <= img_h - patch_h; ++y) {
		for (int x = 0; x <= img_w - patch_w; ++x) {
			int correlation = calc_pixels_abs_a_minus_b_sum(img, patch, patch_w, patch_h, x, y);

			if (correlation < max_correlation) {
				max_correlation = correlation;
				correlation_x = x;
				correlation_y = y;
			}
		}
	}
	result[0] = max_correlation;
	result[1] = correlation_x;
	result[2] = correlation_y;
}

// schreibt Helligkeitswerte aus einem Array in eine Matrix gegebener Größe
void array_to_matrix(unsigned char** matrix, const unsigned char* arr, int cols, int rows) {
	int k = 0;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			matrix[x][y] = arr[k++];
		}
	}
}

unsigned char** alloc_mat(int cols, int rows) {
	unsigned char** A1, * A2;
	A1 = (unsigned char**) calloc(cols, sizeof(unsigned char*));     // pointer on columns
	A2 = (unsigned char*) calloc(rows * cols, sizeof(unsigned char));    // all matrix elements
	for (int x = 0; x < cols; ++x) {
		A1[x] = A2 + x * rows;
	}
	return A1;
}

template<class T>
void free_mat(T **A) {
	free(A[0]); // free contiguous block of float elements (row*col floats)
	free(A);    // free memory for pointers pointing to the beginning of each row
}

int main(int argc, char** argv) {
	char *const img_path = argv[1];
	char *const patch_path = argv[2];

	int processID, comm_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &processID);
	MPI_Status status;

	int offsetTag = 1;

	int img_w;
	int img_h;
	int img_c; // number of image channels
	int desired_c = 1;

	unsigned char* img = NULL;

	int patch_w;
	int patch_h;
	int patch_c;
	unsigned char* patch = NULL;

	unsigned char** img2d;
	unsigned char** patch2d;

	int offset;

	if (processID == 0) {
		//load RGB image as 1 channel grayscale image (1x unsigned 8 bit per pixel)
		img = stbi_load(img_path, &img_w, &img_h, &img_c, desired_c);
		printf("\nLoaded image: %s\n", (img != NULL ? "true" : "false"));
		printf("\twidth: %d\n", img_w);
		printf("\theight: %d\n", img_h);
		patch = stbi_load(patch_path, &patch_w, &patch_h, &patch_c, desired_c);
		printf("\nLoaded patch: %s\n", (patch != NULL ? "true" : "false"));
		printf("\twidth: %d\n", patch_w);
		printf("\theight: %d\n", patch_h);

		//convert
		img2d = alloc_mat(img_w, img_h);
		patch2d = alloc_mat(patch_w, patch_h);
		array_to_matrix(img2d, img, img_w, img_h);
		array_to_matrix(patch2d, patch, patch_w, patch_h);

		double start, end;
		start = MPI_Wtime();

		int worker_count = comm_size - 1;
		int whole_part = (img_w - patch_w) / worker_count;
		int remainder = (img_w - patch_w) % worker_count;
		offset = 0;

		for (int worker_id = 1; worker_id < comm_size; ++worker_id) {
			int segment_width = worker_id <= remainder ? whole_part + 1 : whole_part;
			int col_count = segment_width + (patch_w - 1);

			MPI_Send(&patch_w, 1, MPI_INT, worker_id, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&patch_h, 1, MPI_INT, worker_id, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&patch2d[0][0], patch_w * patch_h, MPI_UNSIGNED_CHAR, worker_id, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&col_count, 1, MPI_INT, worker_id, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&img_h, 1, MPI_INT, worker_id, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, worker_id, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&img2d[offset][0], col_count * img_h, MPI_UNSIGNED_CHAR, worker_id, offsetTag, MPI_COMM_WORLD);
			offset += segment_width;
		}
		int max_correlation[3] = {999999, -1, -1};

		for (int workerID = 1; workerID < comm_size; ++workerID) {
			int worker_correlation[3];
			MPI_Recv(&worker_correlation[0], 3, MPI_INT, workerID, offsetTag, MPI_COMM_WORLD, &status);

			if (worker_correlation[0] < max_correlation[0]) {
				max_correlation[0] = worker_correlation[0];
				max_correlation[1] = worker_correlation[1];
				max_correlation[2] = worker_correlation[2];
			}
		}
		end = MPI_Wtime();
		printf("Found Nemo at x: %i y: %i\n", max_correlation[1], max_correlation[2]);
		printf("Task took %fs to complete.\n", end - start);
		free_mat(img2d);
		free_mat(patch2d);
		free(img);
		free(patch);
	}
	if (processID != 0) {
		MPI_Recv(&patch_w, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);
		MPI_Recv(&patch_h, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);

		patch2d = alloc_mat(patch_w, patch_h);
		MPI_Recv(&patch2d[0][0], patch_w * patch_h, MPI_UNSIGNED_CHAR, 0, offsetTag, MPI_COMM_WORLD, &status);

		MPI_Recv(&img_w, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);
		MPI_Recv(&img_h, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);
		MPI_Recv(&offset, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);
		img2d = alloc_mat(img_w, img_h);
		MPI_Recv(&img2d[0][0], img_w * img_h, MPI_UNSIGNED_CHAR, 0, offsetTag, MPI_COMM_WORLD, &status);

		int worker_correlation[3];
		match_patch(img2d, img_w, img_h, patch2d, patch_w, patch_h, worker_correlation);
		worker_correlation[1] += offset;

		MPI_Send(&worker_correlation[0], 3, MPI_INT, 0, offsetTag, MPI_COMM_WORLD);
		free_mat(img2d);
		free_mat(patch2d);
	}
	MPI_Finalize();
	return 0;
}