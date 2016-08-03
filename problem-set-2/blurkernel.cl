const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

kernel void blur(read_only image2d_t input, write_only image2d_t output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int width = get_global_size(0);
    int height = get_global_size(1);
    
    // Don't process edges
    if(x == 0 || x == width || y == 0 || y == height) {
        return;
    }
    
    uint r = 0;
    uint g = 0;
    uint b = 0;
    
    // Center
    uint4 tap = read_imageui(input, sampler, (int2)(x, y));
    b += 0.2 * tap.x;
    g += 0.2 * tap.y;
    r += 0.2 * tap.z;
    
    // North
    tap = read_imageui(input, sampler, (int2)(x, y-1));
    b += 0.2 * tap.x;
    g += 0.2 * tap.y;
    r += 0.2 * tap.z;
    
    // South
    tap = read_imageui(input, sampler, (int2)(x, y+1));
    b += 0.2 * tap.x;
    g += 0.2 * tap.y;
    r += 0.2 * tap.z;
    
    // West
    tap = read_imageui(input, sampler, (int2)(x-1, y));
    b += 0.2 * tap.x;
    g += 0.2 * tap.y;
    r += 0.2 * tap.z;
    
    // East
    tap = read_imageui(input, sampler, (int2)(x+1, y));
    b += 0.2 * tap.x;
    g += 0.2 * tap.y;
    r += 0.2 * tap.z;
    
    uint4 blur;
    blur.x = b;
    blur.y = g;
    blur.z = r;
    
    write_imageui(output, (int2)(x,y), blur);
}