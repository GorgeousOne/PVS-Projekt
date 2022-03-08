
__kernel void match_patch(__global uchar* img, __global uchar* patch, __global int* heights, __global int* result) {
	uint x, y;
	y = get_global_id(0);
	x = get_global_id(1);

    uint img_h, patch_w, patch_h, result_h;
    img_h = heights[1];
    patch_w = heights[2];
    patch_h = heights[3];
    result_h = img_h - patch_h;

    int sum = 0;
    uint dx, dy;
    for (dy = 0; dy < patch_h; ++dy) {
        for (dx = 0; dx < patch_w; ++dx) {
            sum += abs(img[(y + dy) * img_h + x + dx] - patch[dy * patch_h + dx]);
        }
    }
    result[y * result_h + x] = sum;
}
