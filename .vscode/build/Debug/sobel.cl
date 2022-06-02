__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;

__kernel void sobel(read_only image2d_t src, write_only image2d_t dst)
{
    
	int x = get_global_id(0);
    int y = get_global_id(1);

    int w = get_global_size(0);
    int h = get_global_size(1);
    	
    if(x > 0 && x+1 < w && y > 0 && y+1 < h && y%2 == 0 && x%2 == 0) {
        float4 sum = read_imagef(src, sampler, (int2)(x, y));
        
		write_imagef(dst, (int2)(x/2, y/2 + h/2), sum);
		write_imagef(dst, (int2)(x/2 + w/2, y/2 + h/2), sum);
		write_imagef(dst, (int2)(x/2 + w/4, y/2), sum);
    }

    
    
}