#include <omp.h>
#include <mpi.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

void print_path(char *file_name) {
	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		printf("File path: %s%s%s\n", cwd, "/", file_name);
	} else {
		perror("getcwd() error");
	}
}

float calc_pixels_mean_value(const unsigned char *img, int w, int h, int x = 0, int y = 0) {
	int pixel_sum = 0;
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			pixel_sum += img[x + i, y + j];
		}
	}
	return 1.0f * pixel_sum / (w * h);
}

int calc_pixels_a_times_b_sum(const unsigned char *img_a, const unsigned char *patch_b, int w, int h, int img_x, int img_y) {
	int pixel_sum = 0;
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			pixel_sum += img_a[img_x + i, img_y + j] * patch_b[i, j];
		}
	}
	return pixel_sum;
}

int calc_pixels_squared_sum(const unsigned char *img, int w, int h, int x = 0, int y = 0) {
	int pixels_squared_sum = 0;
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			int pixel = img[x + i, y + j];
			pixels_squared_sum += pixel * pixel;
		}
	}
	return pixels_squared_sum;
}

void match_patch(unsigned char *img, int img_h, int img_w, unsigned char *patch, int patch_h, int patch_w) {

	float n_squared_inv = 1.0f / (patch_w * patch_h);
	float patch_mean_value = calc_pixels_mean_value(patch, patch_w, patch_h);
	int patch_pixels_squared_sum = calc_pixels_squared_sum(patch, patch_w, patch_h);
	float patch_pixels_squared_normalized = n_squared_inv * patch_pixels_squared_sum - patch_mean_value * patch_mean_value;

	float min_correlation = -1;
	int min_x = -1;
	int min_y = -1;

	for (int x = 0; x < img_w - patch_w; ++x) {
		for (int y = 0; y < img_h - patch_h; ++y) {

			float img_mean_value = calc_pixels_mean_value(img, patch_w, patch_h, x, y);
			int img_pixels_squared_sum = calc_pixels_squared_sum(img, patch_w, patch_h, x, y);
			float img_pixels_squared_normalized = n_squared_inv * img_pixels_squared_sum - img_mean_value * img_mean_value;

			float denominator = sqrtf(img_pixels_squared_normalized * patch_pixels_squared_normalized);

			if (denominator == 0) {
				continue;
			}

			float numerator = n_squared_inv * calc_pixels_a_times_b_sum(img, patch, patch_w, patch_h, x, y) - img_mean_value * patch_mean_value;
			float correlation = numerator / denominator;

			printf("x: %d y: %d -> %2f\n", x, y, correlation);
			if (correlation > min_correlation) {
				min_correlation = correlation;
				min_x = x;
				min_y = y;
			}
		}
	}
	printf("minimum: %d, %d", min_x, min_y);
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
	printf("\theight: %d\n", img_h);
	printf("\twidth: %d\n", img_w);

	int patch_w = 0;
	int patch_h = 0;
	int patch_c = 0;
	unsigned char *patch = NULL;
	patch = stbi_load(patch_path, &patch_w, &patch_h, &patch_c, desired_c);
	printf("\nLoaded patch: %s\n", (patch != NULL ? "true" : "false"));
	print_path(patch_path);
	printf("\theight: %d\n", patch_h);
	printf("\twidth: %d\n", patch_w);

	match_patch(img, img_h, img_w, patch, patch_h, patch_w);
	return 0;
}
