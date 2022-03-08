#define DIMX 914
#define DIMY 665

__kernel void match_patch(__global uchar *A, __global uchar *B, __global int *C) {
	uint x, y;
	x = get_global_id(0);
	y = get_global_id(1);

    //calculate brightness differences
    float sum = 0;
    uint dx, dy;
    for (dy = 0; dy < 44; ++dy) {
        for (dx = 0; dx < 59; ++dx) {
            sum += abs(A[(y + dy) * DIMY + x + dx] - B[dy * 44 + dx]);
        }
    }
    C[y * (DIMY - 44) + x] = sum;
    if (x < 5 && y < 5) {
        printf("%i \i\n", x, y);
    }

    //print a segment of the matrix as example
	if (x == 1 && y == 1) {
        printf("Kernel Matrix:\n");
        uint i, j;
        for (i = 0; i < 10; ++i) {
            for (j = 0; j < 10; ++j) {
                printf("%i  ", A[j * DIMX + i]);
            }
            printf("\n");
        }
    }
}
