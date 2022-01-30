#include <omp.h>
#include <mpi.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

void print_path(char* file_name) {
	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		printf("File path: %s%s%s\n", cwd, "/", file_name);
	} else {
		perror("getcwd() error");
	}
}

int main(int argc, char **argv) {

	char *const img_path = argv[1];
	int img_w = 0;
	int img_h = 0;
	int img_c = 0; // number of image channels

	unsigned char *img = NULL;

	//load RGB image as 1 channel grayscale image (1x unsigned 8 bit per pixel)
	img = stbi_load(img_path, &img_w, &img_h, &img_c, 1);
	printf("\nLoaded Image: %s\n", (img != NULL? "true" : "false"));
	print_path(img_path);
	printf("\theight: %d\n", img_h);
	printf("\twidth: %d\n", img_w);

	return 0;
}
