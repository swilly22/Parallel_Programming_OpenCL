const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

kernel void grayScale(read_only image2d_t input, write_only image2d_t output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    uint4 tap = read_imageui(input, sampler, (int2)(x, y));
    uint r = tap.x;
    uint g = tap.y;
    uint b = tap.z;
    uint gray = 0.299 * r + 0.587 * g + 0.114f * b;
    
    uint4 intensity;
    intensity.x = gray;
    intensity.y = gray;
    intensity.z = gray;
    
    write_imageui(output, (int2)(x,y), intensity);
}