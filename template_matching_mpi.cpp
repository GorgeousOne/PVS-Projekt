#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


float calc_pixels_mean_value(unsigned char** img, int w, int h, int x = 0, int y = 0) {
	int pixel_sum = 0;
	for (int dy = 0; dy < h; ++dy) {
		for (int dx = 0; dx < w; ++dx) {
			pixel_sum += img[x + dx][y + dy];
		}
	}
	return 1.0f * pixel_sum / (w * h);
}

int calc_pixels_a_times_b_sum(unsigned char** img_a, unsigned char** patch_b, int w, int h, int img_x, int img_y) {
	int pixel_sum = 0;
	for (int dy = 0; dy < h; ++dy) {
		for (int dx = 0; dx < w; ++dx) {
			pixel_sum += img_a[img_x + dx][img_y + dy] * patch_b[dx][dy];
		}
	}
	return pixel_sum;
}

int calc_pixels_squared_sum(unsigned char** img, int w, int h, int x = 0, int y = 0) {
	int pixels_squared_sum = 0;
	for (int dy = 0; dy < h; ++dy) {
		for (int dx = 0; dx < w; ++dx) {
			int pixel = img[x + dx][y + dy];
			pixels_squared_sum += pixel * pixel;
		}
	}
	return pixels_squared_sum;
}

int calc_pixels_abs_a_minus_b_sum(unsigned char **img_a, unsigned char **patch_b, int w, int h, int img_x, int img_y) {
	int pixel_sum = 0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			pixel_sum += abs(img_a[img_x + x][img_y + y] - patch_b[x][y]);
		}
	}
	if (img_x < 5 && img_y < 5) {
		printf("%i\n", pixel_sum);
	}
	return pixel_sum;
}

void match_patch(unsigned char** img, int img_w, unsigned char** patch, int patch_w, int patch_h, int start_y, int end_y, int part, int cor[3]) {

	int max_correlation = 9999999;
	int correlation_x = -1;
	int correlation_y = -1;

	for (int y = start_y; y <= end_y - patch_h; ++y) {
		for (int x = 0; x <= img_w - patch_w; ++x) {
			int correlation = calc_pixels_abs_a_minus_b_sum(img, patch, patch_w, patch_h, x, y);

			if (correlation < max_correlation) {
				max_correlation = correlation;
				correlation_x = x;
				correlation_y = y;
			}
		}
	}
	cor[0] = max_correlation;
	cor[1] = correlation_x;
	cor[2] = correlation_y;
	printf("maximum of patch %i: %i, %i\n", part, correlation_x, correlation_y);
}

// https://stackoverflow.com/questions/61410931/write-a-c-program-to-convert-1d-array-to-2d-array-using-pointers Besucht: 03.03.2022
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
	A1 = (unsigned char**)calloc(cols, sizeof(unsigned char*));     // pointer on columns
	A2 = (unsigned char*)calloc(rows * cols, sizeof(unsigned char));    // all matrix elements
	for (int x = 0; x < cols; ++x) {
		A1[x] = A2 + x * rows;
	}
	return A1;
}

int main(int argc, char** argv) {
	char *const img_path = argv[1];
	char *const patch_path = argv[2];

	int processID, commSize;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &processID);
	MPI_Status status;

	int offsetTag = 1;

	int img_w = 0;
	int img_h = 0;
	int img_c = 0; // number of image channels
	int desired_c = 1;

	unsigned char* img = NULL;
	//load RGB image as 1 channel grayscale image (1x unsigned 8 bit per pixel)
	img = stbi_load("../search_area_small.jpg", &img_w, &img_h, &img_c, desired_c);
	if (processID == 0) {
		printf("\nLoaded image: %s\n", (img != NULL ? "true" : "false"));
		printf("\twidth: %d\n", img_w);
		printf("\theight: %d\n", img_h);
	}

	int patch_w = 0;
	int patch_h = 0;
	int patch_c = 0;
	unsigned char* patch = NULL;
	patch = stbi_load("../nemo_template.png", &patch_w, &patch_h, &patch_c, desired_c);
	if (processID == 0) {
		printf("\nLoaded patch: %s\n", (patch != NULL ? "true" : "false"));
		printf("\twidth: %d\n", patch_w);
		printf("\theight: %d\n", patch_h);
	}

	unsigned char** img2d = alloc_mat(img_w, img_h);
	unsigned char** patch2d = alloc_mat(patch_w, patch_h);

	array_to_matrix(img2d, img, img_w, img_h);
	array_to_matrix(patch2d, patch, patch_w, patch_h);

	int offset_y;
	int worker_count;
	int start_y;
	int end_y;

	int w_max_correlation;
	int w_correlation_x;
	int w_correlation_y;

	if (processID == 0) {
		double start, end;
		start = MPI_Wtime();

		offset_y = img_h / commSize;
		start_y = 0;
		worker_count = commSize - 1;
		end_y = img_h - offset_y * worker_count;

		for (int workerID = 1; workerID < commSize; ++workerID) {
			start_y += offset_y;
			end_y += offset_y;
			MPI_Send(&start_y, 1, MPI_INT, workerID, offsetTag, MPI_COMM_WORLD);
			MPI_Send(&end_y, 1, MPI_INT, workerID, offsetTag, MPI_COMM_WORLD);
		}

		int max_cor[3];
		match_patch(img2d, img_w, patch2d, patch_w, patch_h, 0, img_h - offset_y * worker_count, 0, max_cor);
		int max_correlation = max_cor[0];
		int max_correlation_x = max_cor[1];
		int max_correlation_y = max_cor[2];

		for (int workerID = 1; workerID < commSize; ++workerID) {
			MPI_Recv(&w_max_correlation, 1, MPI_INT, workerID, offsetTag, MPI_COMM_WORLD, &status);
			MPI_Recv(&w_correlation_x, 1, MPI_INT, workerID, offsetTag, MPI_COMM_WORLD, &status);
			MPI_Recv(&w_correlation_y, 1, MPI_INT, workerID, offsetTag, MPI_COMM_WORLD, &status);
			if (w_max_correlation < max_correlation) {
				max_correlation = w_max_correlation;
				max_correlation_x = w_correlation_x;
				max_correlation_y = w_correlation_y;
			}
		}
		end = MPI_Wtime();
		printf("Task took %fs to complete.\n", end - start);
		printf("Found nemo at x: %i y: %i\n", max_correlation_x, max_correlation_y);
	}
	if (processID != 0) {
		MPI_Recv(&start_y, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);
		MPI_Recv(&end_y, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD, &status);

		int cor_w[3];
		match_patch(img2d, img_w, patch2d, patch_w, patch_h, start_y, end_y, processID, cor_w);
		w_max_correlation = cor_w[0];
		w_correlation_x = cor_w[1];
		w_correlation_y = cor_w[2];
		MPI_Send(&w_max_correlation, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD);
		MPI_Send(&w_correlation_x, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD);
		MPI_Send(&w_correlation_y, 1, MPI_INT, 0, offsetTag, MPI_COMM_WORLD);
	}
	MPI_Finalize();
}