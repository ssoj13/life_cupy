"""CUDA kernels for Game of Life simulation."""
import cupy as cp


# Optimized Game of Life kernel without conditionals to avoid warp divergence
LIFE_KERNEL = r'''
extern "C" __global__
void life_step(const unsigned char* current, unsigned char* next, 
               const int width, const int height) {
    // Grid-stride loop for handling arbitrary sizes
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < width * height; i += stride) {
        int x = i % width;
        int y = i / width;
        
        // Count neighbors with boundary wrapping (toroidal topology)
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                count += current[ny * width + nx];
            }
        }
        
        // Apply Game of Life rules without conditionals
        // Cell is alive if: (count == 3) OR (count == 2 AND current_state == 1)
        unsigned char alive = (count == 3) | ((count == 2) & current[i]);
        next[i] = alive;
    }
}
'''

# Kernel for drawing circles on the field
DRAW_CIRCLE_KERNEL = r'''
extern "C" __global__
void draw_circle(unsigned char* field, const int width, const int height,
                 const int center_x, const int center_y, const int radius,
                 const unsigned char value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int radius_sq = radius * radius;
    
    for (int i = idx; i < width * height; i += stride) {
        int x = i % width;
        int y = i / width;
        
        int dx = x - center_x;
        int dy = y - center_y;
        int dist_sq = dx * dx + dy * dy;
        
        if (dist_sq <= radius_sq) {
            field[i] = value;
        }
    }
}
'''

# Kernel for adding noise pattern
NOISE_KERNEL = r'''
#include <curand_kernel.h>

extern "C" __global__
void add_noise(unsigned char* field, const int width, const int height,
               const float density, const unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize random state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    for (int i = idx; i < width * height; i += stride) {
        float random = curand_uniform(&state);
        if (random < density) {
            field[i] = 1;
        }
    }
}
'''

# Kernel for clearing the field
CLEAR_KERNEL = r'''
extern "C" __global__
void clear_field(unsigned char* field, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        field[i] = 0;
    }
}
'''

# Kernel for converting field to RGBA for display
FIELD_TO_RGBA_KERNEL = r'''
extern "C" __global__
void field_to_rgba(const unsigned char* field, unsigned char* rgba,
                   const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        unsigned char value = field[i] * 255;
        int rgba_idx = i * 4;
        rgba[rgba_idx + 0] = value;  // R
        rgba[rgba_idx + 1] = value;  // G
        rgba[rgba_idx + 2] = value;  // B
        rgba[rgba_idx + 3] = 255;    // A
    }
}
'''


def compile_kernels():
    """Compile all CUDA kernels and return them."""
    kernels = {
        'life_step': cp.RawKernel(LIFE_KERNEL, 'life_step'),
        'draw_circle': cp.RawKernel(DRAW_CIRCLE_KERNEL, 'draw_circle'),
        'add_noise': cp.RawKernel(NOISE_KERNEL, 'add_noise', 
                                  options=('-lcurand',)),
        'clear_field': cp.RawKernel(CLEAR_KERNEL, 'clear_field'),
        'field_to_rgba': cp.RawKernel(FIELD_TO_RGBA_KERNEL, 'field_to_rgba')
    }
    return kernels