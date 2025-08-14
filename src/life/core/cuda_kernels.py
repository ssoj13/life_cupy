"""CUDA kernels for unified multi-channel cellular automata simulation."""
import cupy as cp
from enum import IntEnum


class RuleType(IntEnum):
    """Types of cellular automata rules."""
    BINARY_BS = 0          # Classic binary B/S notation rules (RGBA interpreted as binary)
    MULTISTATE = 1         # Multi-state rules (Brian's Brain, Generations)
    MULTICHANNEL = 2       # Multi-channel rules (Life simulation, RGB)
    EXTENDED_CLASSIC = 3   # RGB layers with independent classic rules


class BinaryRule(IntEnum):
    """Classic binary B/S notation rules."""
    CONWAY_LIFE = 0        # B3/S23
    HIGHLIFE = 1           # B36/S23
    SEEDS = 2              # B2/S
    DAY_NIGHT = 3          # B3678/S34678
    MAZE = 4               # B3/S12345
    REPLICATOR = 5         # B1357/S1357
    DRYLIFE = 6            # B37/S23
    LIVE_FREE_DIE = 7      # B2/S0
    RULE_2X2 = 8           # B36/S125
    LIFE_WITHOUT_DEATH = 9 # B3/S012345678
    GRADUAL_CONWAY = 10    # Gradual Conway with 10% growth/decay
    CLASSIC_CONWAY = 11    # Original Conway B3/S23


class MultiStateRule(IntEnum):
    """Multi-state cellular automata rules."""
    BRIANS_BRAIN = 0
    GENERATIONS = 1


class MultiChannelRule(IntEnum):
    """Multi-channel cellular automata rules."""
    LIFE_SIMULATION = 0    # Health/Money simulation
    RGB_EVOLUTION = 1      # Color-based evolution


class ExtendedClassicRule(IntEnum):
    """Extended classic rules with RGB interpretation.""" 
    RGB_CONWAY = 0         # Conway's Life on each RGB channel independently


# Cell structure (4 bytes per cell)
CELL_STRUCT = r'''
struct Cell {
    unsigned char ch1;  // Channel 1: alive/dead, mental_health, R
    unsigned char ch2;  // Channel 2: dying_state, body_health, G  
    unsigned char ch3;  // Channel 3: unused, social_health, B
    unsigned char ch4;  // Channel 4: unused, money, alpha
};
'''

# Anti-aliased line segment kernel
ANTIALIASED_LINE_KERNEL = r'''
extern "C" __global__
void draw_antialiased_line(unsigned char* field,
                          const float x1, const float y1,
                          const float x2, const float y2,
                          const float thickness,
                          const unsigned char ch1, const unsigned char ch2,
                          const unsigned char ch3, const unsigned char ch4,
                          const int width, const int height) {
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate bounding box for the line
    float radius = thickness * 0.5f;
    int min_x = max(0, (int)floorf(fminf(x1, x2) - radius - 1));
    int max_x = min(width - 1, (int)ceilf(fmaxf(x1, x2) + radius + 1));
    int min_y = max(0, (int)floorf(fminf(y1, y2) - radius - 1));
    int max_y = min(height - 1, (int)ceilf(fmaxf(y1, y2) + radius + 1));
    
    int total_pixels = (max_x - min_x + 1) * (max_y - min_y + 1);
    if (thread_id >= total_pixels) return;
    
    // Convert thread ID to pixel coordinates
    int box_width = max_x - min_x + 1;
    int local_x = thread_id % box_width;
    int local_y = thread_id / box_width;
    int pixel_x = min_x + local_x;
    int pixel_y = min_y + local_y;
    
    // Calculate distance from pixel to line segment
    float dx = x2 - x1;
    float dy = y2 - y1;
    float length_sq = dx * dx + dy * dy;
    
    float distance;
    if (length_sq < 0.0001f) {
        // Point line - distance to point
        float px = pixel_x + 0.5f - x1;
        float py = pixel_y + 0.5f - y1;
        distance = sqrtf(px * px + py * py);
    } else {
        // Line segment - distance to closest point on line
        float px = pixel_x + 0.5f - x1;
        float py = pixel_y + 0.5f - y1;
        float t = fmaxf(0.0f, fminf(1.0f, (px * dx + py * dy) / length_sq));
        float closest_x = x1 + t * dx;
        float closest_y = y1 + t * dy;
        float dist_x = pixel_x + 0.5f - closest_x;
        float dist_y = pixel_y + 0.5f - closest_y;
        distance = sqrtf(dist_x * dist_x + dist_y * dist_y);
    }
    
    // Anti-aliased coverage calculation
    float coverage = 1.0f - fmaxf(0.0f, fminf(1.0f, distance - radius + 0.5f));
    if (coverage > 0.0f) {
        int idx = (pixel_y * width + pixel_x) * 4;
        
        // Alpha blending with coverage
        float alpha = coverage;
        float inv_alpha = 1.0f - alpha;
        
        field[idx] = (unsigned char)(ch1 * alpha + field[idx] * inv_alpha);
        field[idx + 1] = (unsigned char)(ch2 * alpha + field[idx + 1] * inv_alpha);
        field[idx + 2] = (unsigned char)(ch3 * alpha + field[idx + 2] * inv_alpha);
        field[idx + 3] = (unsigned char)(ch4 * alpha + field[idx + 3] * inv_alpha);
    }
}
'''


# Unified multi-channel cellular automata kernel
UNIFIED_KERNEL = r'''
#include <curand_kernel.h>

struct Cell {
    unsigned char ch1, ch2, ch3, ch4;
};

// Lookup tables for binary B/S rules (stored in constant memory)
__constant__ unsigned char birth_table[90];   // 10 rules × 9 neighbors  
__constant__ unsigned char survive_table[90]; // 10 rules × 9 neighbors

extern "C" __global__
void unified_step(const struct Cell* current, struct Cell* next,
                  const int width, const int height, const int rule_type,
                  const int rule_id, const unsigned long long seed) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize random state for multichannel rules
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    for (int i = idx; i < width * height; i += stride) {
        int x = i % width;
        int y = i / width;
        
        struct Cell cell = current[i];
        struct Cell new_cell = {0, 0, 0, 0};
        
        // Count neighbors based on rule type
        if (rule_type == 0) { // BINARY_BS
            // Count alive neighbors (ch1 > 0)
            int count = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = (x + dx + width) % width;
                    int ny = (y + dy + height) % height;
                    count += (current[ny * width + nx].ch1 > 0) ? 1 : 0;
                }
            }
            
            // Apply binary B/S rule - DEBUG VERSION
            unsigned char birth = birth_table[rule_id * 9 + count];
            unsigned char survive = survive_table[rule_id * 9 + count] & (cell.ch1 > 0);
            unsigned char alive = birth | survive;
            new_cell.ch1 = alive ? 255 : 0;
            
            
        } else if (rule_type == 1) { // MULTISTATE
            if (rule_id == 0) { // Brian's Brain
                // Count alive neighbors for birth
                int count = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = (x + dx + width) % width;
                        int ny = (y + dy + height) % height;
                        count += (current[ny * width + nx].ch1 == 255) ? 1 : 0;
                    }
                }
                
                // Brian's Brain transitions: alive->dying->dead, birth with 2 neighbors
                if (cell.ch1 == 255) {
                    new_cell.ch1 = 128; // alive -> dying
                } else if (cell.ch1 == 128) {
                    new_cell.ch1 = 0;   // dying -> dead
                } else if (count == 2) {
                    new_cell.ch1 = 255; // birth
                } else {
                    new_cell.ch1 = 0;   // stay dead
                }
            }
            
        } else if (rule_type == 2) { // MULTICHANNEL
            if (rule_id == 0) { // Life Simulation
                // Collect neighbor health data
                int mental_sum = 0, body_sum = 0, social_sum = 0, money_sum = 0;
                int neighbor_count = 0;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = (x + dx + width) % width;
                        int ny = (y + dy + height) % height;
                        struct Cell neighbor = current[ny * width + nx];
                        
                        mental_sum += neighbor.ch1;
                        body_sum += neighbor.ch2;
                        social_sum += neighbor.ch3;
                        money_sum += neighbor.ch4;
                        neighbor_count++;
                    }
                }
                
                // Life simulation rules
                // Mental health: decreases in isolation, increases with social connections
                int mental = cell.ch1;
                if (social_sum < 8 * 50) mental = max(0, mental - 2); // isolation decay
                if (social_sum > 8 * 150) mental = min(255, mental + 1); // social boost
                
                // Body health: slow decay, recovers with good mental health
                int body = cell.ch2;
                body = max(0, body - 1); // natural decay
                if (mental > 200) body = min(255, body + 2); // mental health boost
                
                // Social health: spreads from neighbors, requires mental stability
                int social = cell.ch3;
                if (mental < 30) social = max(0, social - 3); // mental instability
                else social = (social + social_sum / 8) / 2; // neighbor influence
                
                // Money: accumulates with health, random spending
                int money = cell.ch4;
                if (mental + body > 300) money = min(255, money + 1); // health income
                if (curand_uniform(&state) < 0.1f) money = max(0, money - 5); // random expenses
                
                new_cell.ch1 = mental;
                new_cell.ch2 = body;
                new_cell.ch3 = social;
                new_cell.ch4 = money;
            }
        }
        
        // Preserve existing non-zero content if simulation didn't produce a result
        if (new_cell.ch1 == 0 && new_cell.ch2 == 0 && new_cell.ch3 == 0 && new_cell.ch4 == 0) {
            // If simulation produced empty cell, check if original had drawing content
            if (cell.ch1 > 0 || cell.ch2 > 0 || cell.ch3 > 0 || cell.ch4 > 0) {
                // Preserve original drawing content
                new_cell = cell;
            }
        }
        
        next[i] = new_cell;
    }
}
'''

# Kernel for drawing stroke chains
DRAW_STROKE_CHAIN_KERNEL = r'''
struct Cell {
    unsigned char ch1, ch2, ch3, ch4;
};

extern "C" __global__
void draw_stroke_chain(struct Cell* field, const float* points, const int num_points,
                      const int width, const int height, const float thickness,
                      const unsigned char val1, const unsigned char val2,
                      const unsigned char val3, const unsigned char val4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float radius = thickness * 0.5f;
    int total_pixels = width * height;
    
    for (int i = idx; i < total_pixels; i += stride) {
        int x = i % width;
        int y = i / width;
        
        // Check distance to any line segment in the stroke chain
        bool should_draw = false;
        
        for (int seg = 0; seg < num_points - 1; seg++) {
            float x1 = points[seg * 2];
            float y1 = points[seg * 2 + 1];
            float x2 = points[seg * 2 + 2];
            float y2 = points[seg * 2 + 3];
            
            // Distance from point to line segment
            float dx = x2 - x1;
            float dy = y2 - y1;
            float length_sq = dx * dx + dy * dy;
            
            if (length_sq < 0.0001f) {
                // Degenerate line, treat as point
                float dist = sqrtf((x - x1) * (x - x1) + (y - y1) * (y - y1));
                if (dist <= radius) {
                    should_draw = true;
                    break;
                }
            } else {
                // Project point onto line
                float t = ((x - x1) * dx + (y - y1) * dy) / length_sq;
                t = fmaxf(0.0f, fminf(1.0f, t));
                
                float proj_x = x1 + t * dx;
                float proj_y = y1 + t * dy;
                
                float dist = sqrtf((x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y));
                if (dist <= radius) {
                    should_draw = true;
                    break;
                }
            }
        }
        
        if (should_draw) {
            struct Cell new_cell = {val1, val2, val3, val4};
            field[i] = new_cell;
        }
    }
}
'''

# Kernel for drawing circles on the field  
DRAW_CIRCLE_KERNEL = r'''
struct Cell {
    unsigned char ch1, ch2, ch3, ch4;
};

extern "C" __global__
void draw_circle(struct Cell* field, const int width, const int height,
                 const int center_x, const int center_y, const int radius,
                 const unsigned char val_ch1, const unsigned char val_ch2,
                 const unsigned char val_ch3, const unsigned char val_ch4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int radius_sq = radius * radius;
    struct Cell value = {val_ch1, val_ch2, val_ch3, val_ch4};
    
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

struct Cell {
    unsigned char ch1, ch2, ch3, ch4;
};

extern "C" __global__
void add_noise(struct Cell* field, const int width, const int height,
               const float density, const unsigned long long seed,
               const int rule_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize random state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    for (int i = idx; i < width * height; i += stride) {
        float random = curand_uniform(&state);
        if (random < density) {
            if (rule_type == 0 || rule_type == 1) { // Binary or multistate
                field[i].ch1 = 255;
                field[i].ch2 = 0;
                field[i].ch3 = 0;
                field[i].ch4 = 0;
            } else { // Multichannel
                field[i].ch1 = (unsigned char)(curand_uniform(&state) * 255); // Mental
                field[i].ch2 = (unsigned char)(curand_uniform(&state) * 255); // Body
                field[i].ch3 = (unsigned char)(curand_uniform(&state) * 255); // Social
                field[i].ch4 = (unsigned char)(curand_uniform(&state) * 255); // Money
            }
        }
    }
}
'''

# Kernel for clearing the field
CLEAR_KERNEL = r'''
struct Cell {
    unsigned char ch1, ch2, ch3, ch4;
};

extern "C" __global__
void clear_field(struct Cell* field, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    struct Cell empty = {0, 0, 0, 0};
    
    for (int i = idx; i < size; i += stride) {
        field[i] = empty;
    }
}
'''

# Kernel for converting field to RGBA for display
FIELD_TO_RGBA_KERNEL = r'''
struct Cell {
    unsigned char ch1, ch2, ch3, ch4;
};

extern "C" __global__
void field_to_rgba(const struct Cell* field, unsigned char* rgba,
                   const int size, const int rule_type, const int display_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        struct Cell cell = field[i];
        int rgba_idx = i * 4;
        
        if (rule_type == 0) { // Binary rules - show as black/white
            unsigned char value = cell.ch1;
            rgba[rgba_idx + 0] = value;  // R
            rgba[rgba_idx + 1] = value;  // G
            rgba[rgba_idx + 2] = value;  // B
            rgba[rgba_idx + 3] = 255;    // A
        } else if (rule_type == 1) { // Multistate - show states as colors
            if (cell.ch1 == 255) {      // Alive - white
                rgba[rgba_idx + 0] = 255;
                rgba[rgba_idx + 1] = 255;
                rgba[rgba_idx + 2] = 255;
            } else if (cell.ch1 == 128) { // Dying - gray
                rgba[rgba_idx + 0] = 128;
                rgba[rgba_idx + 1] = 128;
                rgba[rgba_idx + 2] = 128;
            } else {                     // Dead - black
                rgba[rgba_idx + 0] = 0;
                rgba[rgba_idx + 1] = 0;
                rgba[rgba_idx + 2] = 0;
            }
            rgba[rgba_idx + 3] = 255;    // A
        } else { // Multichannel - display based on mode
            if (display_mode == 0) {     // RGB display
                rgba[rgba_idx + 0] = cell.ch1;  // Mental -> R
                rgba[rgba_idx + 1] = cell.ch2;  // Body -> G
                rgba[rgba_idx + 2] = cell.ch3;  // Social -> B
                rgba[rgba_idx + 3] = 255;       // A
            } else if (display_mode == 1) { // Money as brightness
                unsigned char avg = (cell.ch1 + cell.ch2 + cell.ch3) / 3;
                unsigned char brightness = (avg * cell.ch4) / 255;
                rgba[rgba_idx + 0] = brightness;
                rgba[rgba_idx + 1] = brightness;
                rgba[rgba_idx + 2] = brightness;
                rgba[rgba_idx + 3] = 255;
            }
        }
    }
}
'''


def compile_kernels():
    """Compile all CUDA kernels and return them."""
    # Use RawModule for unified kernel to access constant memory
    unified_module = cp.RawModule(code=UNIFIED_KERNEL)
    
    kernels = {
        'unified_step': unified_module.get_function('unified_step'),
        'unified_module': unified_module,  # Keep reference for constant memory access
        'draw_circle': cp.RawKernel(DRAW_CIRCLE_KERNEL, 'draw_circle'),
        'draw_stroke_chain': cp.RawKernel(DRAW_STROKE_CHAIN_KERNEL, 'draw_stroke_chain'),
        'draw_antialiased_line': cp.RawKernel(ANTIALIASED_LINE_KERNEL, 'draw_antialiased_line'),
        'add_noise': cp.RawKernel(NOISE_KERNEL, 'add_noise'),
        'clear_field': cp.RawKernel(CLEAR_KERNEL, 'clear_field'),
        'field_to_rgba': cp.RawKernel(FIELD_TO_RGBA_KERNEL, 'field_to_rgba')
    }
    return kernels


def get_bs_tables():
    """Get birth/survival lookup tables for all binary rules."""
    import numpy as np
    
    # Define B/S rules: (birth_neighbors, survive_neighbors)
    rules = {
        BinaryRule.CONWAY_LIFE: ([3], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        BinaryRule.HIGHLIFE: ([3, 6], [2, 3]),
        BinaryRule.SEEDS: ([2], []),
        BinaryRule.DAY_NIGHT: ([3, 6, 7, 8], [3, 4, 6, 7, 8]),
        BinaryRule.MAZE: ([3], [1, 2, 3, 4, 5]),
        BinaryRule.REPLICATOR: ([1, 3, 5, 7], [1, 3, 5, 7]),
        BinaryRule.DRYLIFE: ([3, 7], [2, 3]),
        BinaryRule.LIVE_FREE_DIE: ([2], [0]),
        BinaryRule.RULE_2X2: ([3, 6], [1, 2, 5]),
        BinaryRule.LIFE_WITHOUT_DEATH: ([3], [0, 1, 2, 3, 4, 5, 6, 7, 8])
    }
    
    birth_table = np.zeros((10, 9), dtype=np.uint8)
    survive_table = np.zeros((10, 9), dtype=np.uint8)
    
    for rule_id, (birth, survive) in rules.items():
        for count in birth:
            if 0 <= count <= 8:
                birth_table[rule_id][count] = 1
        for count in survive:
            if 0 <= count <= 8:
                survive_table[rule_id][count] = 1
    
    return birth_table, survive_table