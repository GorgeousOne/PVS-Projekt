#define DIMX 914
#define DIMY 665

__kernel void match_patch(__global uchar *A, __global uchar *B, __global uchar *C) {
	int x, y;
	x = get_global_id(0);
	y = get_global_id(1);


	if (x == 1 && y == 1) {
        printf("Kernel Matrix:\n");
        uint i, j;
        for (i = 0; i < 10; ++i) {
            for (j = 0; j < 10; ++j) {
                printf("%i  ", A[j * DIMY + i]);
            }
            printf("\n");
        }
    }
    if (x < 20 && y < 20) {
        C[y * (DIMY - 44 )+ x] = A[y * DIMY + x];
    }
}
