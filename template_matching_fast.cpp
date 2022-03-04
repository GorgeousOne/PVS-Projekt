#include <omp.h>
#include <mpi.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void print_path(char *file_name) {
	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		printf("File path: %s%s%s\n", cwd, "/", file_name);
	} else {
		perror("getcwd() error");
	}
}

int calc_pixels_abs_a_minus_b_sum(unsigned char **img_a, unsigned char **patch_b, int w, int h, int img_x, int img_y) {
	int pixel_sum = 0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			pixel_sum += abs(img_a[img_x + x][img_y + y] - patch_b[x][y]);
		}
	}
	return pixel_sum;
}

void crop(unsigned char **matrix, int x, int y, int w, int h, const char* fileName) {
	unsigned char *patch = (unsigned char *) calloc(w * h, sizeof(unsigned char));
	int k = 0;
	for (int dy = 0; dy < h; ++dy) {
		for (int dx = 0; dx < w; ++dx) {
			patch[k++] = matrix[x + dx][y + dy];
		}
	}
	stbi_write_jpg(fileName, w, h, 1, patch, 100);
}

void match_patch(unsigned char **img, int img_w, int img_h, unsigned char **patch, int patch_w, int patch_h) {

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
	printf("maximum: %i, %i\n", correlation_x, correlation_y);
	crop(img, correlation_x, correlation_y, patch_w, patch_h, "nemo_found.jpg");
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

unsigned char **alloc_mat(int cols, int rows) {
	unsigned char **A1, *A2;
	A1 = (unsigned char **) calloc(cols, sizeof(unsigned char *));     // pointer on rows
	A2 = (unsigned char *) calloc(rows * cols, sizeof(unsigned char));    // all matrix elements
	for (int x = 0; x < cols; ++x) {
		A1[x] = A2 + x * rows;
	}
	return A1;
}

int main(int argc, char **argv) {
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

	unsigned char **img2d = alloc_mat(img_w, img_h);
	unsigned char **patch2d = alloc_mat(patch_w, patch_h);

	array_to_matrix(img2d, img, img_w, img_h);
	array_to_matrix(patch2d, patch, patch_w, patch_h);

	double start, end;
	start = omp_get_wtime();
	match_patch(img2d, img_w, img_h, patch2d, patch_w, patch_h);
	end = omp_get_wtime();
	printf("Task took %fs to complete.\n", end - start);
}