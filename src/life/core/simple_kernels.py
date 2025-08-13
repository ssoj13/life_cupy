"""Simplified CUDA kernels that work with 4-channel arrays."""
import cupy as cp
import numpy as np


# Combined stroke resampling and drawing kernel
STROKE_KERNEL = r'''
extern "C" __global__
void draw_resampled_stroke(unsigned char* field,
                          const float* raw_points, const int num_raw_points,
                          const float step_distance, const int brush_radius,
                          const unsigned char ch1, const unsigned char ch2,
                          const unsigned char ch3, const unsigned char ch4,
                          const int width, const int height) {
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (num_raw_points < 2) {
        // Single point case
        if (thread_id == 0 && num_raw_points == 1) {
            float x = raw_points[0];
            float y = raw_points[1];
            int center_x = (int)x;
            int center_y = (int)y;
            
            // Draw circle at single point
            int radius_sq = brush_radius * brush_radius;
            for (int dy = -brush_radius; dy <= brush_radius; dy++) {
                for (int dx = -brush_radius; dx <= brush_radius; dx++) {
                    if (dx * dx + dy * dy <= radius_sq) {
                        int px = center_x + dx;
                        int py = center_y + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            int idx = (py * width + px) * 4;
                            field[idx] = ch1;
                            field[idx + 1] = ch2;
                            field[idx + 2] = ch3;
                            field[idx + 3] = ch4;
                        }
                    }
                }
            }
        }
        return;
    }
    
    // Calculate total stroke length and number of resampled points
    float total_length = 0.0f;
    for (int i = 0; i < num_raw_points - 1; i++) {
        float dx = raw_points[(i+1)*2] - raw_points[i*2];
        float dy = raw_points[(i+1)*2+1] - raw_points[i*2+1];
        total_length += sqrtf(dx * dx + dy * dy);
    }
    
    int num_resampled = (int)(total_length / step_distance) + 1;
    if (num_resampled <= 0) return;
    
    // Each thread handles one resampled point
    if (thread_id >= num_resampled) return;
    
    // Find the resampled point position
    float target_distance = thread_id * step_distance;
    float current_distance = 0.0f;
    
    float resample_x, resample_y;
    bool found = false;
    
    for (int i = 0; i < num_raw_points - 1 && !found; i++) {
        float x1 = raw_points[i*2];
        float y1 = raw_points[i*2+1];
        float x2 = raw_points[(i+1)*2];
        float y2 = raw_points[(i+1)*2+1];
        
        float dx = x2 - x1;
        float dy = y2 - y1;
        float segment_length = sqrtf(dx * dx + dy * dy);
        
        if (current_distance + segment_length >= target_distance) {
            // Target point is in this segment
            float t = (target_distance - current_distance) / segment_length;
            resample_x = x1 + dx * t;
            resample_y = y1 + dy * t;
            found = true;
        } else {
            current_distance += segment_length;
        }
    }
    
    if (!found) {
        // Use last point if we didn't find exact distance
        resample_x = raw_points[(num_raw_points-1)*2];
        resample_y = raw_points[(num_raw_points-1)*2+1];
    }
    
    // Draw circle at resampled point
    int center_x = (int)resample_x;
    int center_y = (int)resample_y;
    int radius_sq = brush_radius * brush_radius;
    
    for (int dy = -brush_radius; dy <= brush_radius; dy++) {
        for (int dx = -brush_radius; dx <= brush_radius; dx++) {
            if (dx * dx + dy * dy <= radius_sq) {
                int px = center_x + dx;
                int py = center_y + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int idx = (py * width + px) * 4;
                    field[idx] = ch1;
                    field[idx + 1] = ch2;
                    field[idx + 2] = ch3;
                    field[idx + 3] = ch4;
                }
            }
        }
    }
}
'''

# Simple unified kernel for all cellular automata rules
SIMPLE_UNIFIED_KERNEL = r'''
extern "C" __global__
void simple_unified_step(unsigned char* current, unsigned char* next,
                         const int width, const int height,
                         const int rule_type, const int rule_id) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = width * height;
    
    if (idx >= total_cells) return;
    
    int x = idx % width;
    int y = idx / width;
    
    // Index for 4-channel array (height, width, 4)
    int cell_idx = (y * width + x) * 4;
    
    if (rule_type == 0) { // Binary B/S rules
        // Count alive neighbors (using channel 0)
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                if (current[neighbor_idx] > 0) count++;
            }
        }
        
        // Apply different rules based on rule_id
        bool alive = current[cell_idx] > 0;
        bool next_alive = false;
        
        switch(rule_id) {
            case 0: // Conway's Life B3/S23
                next_alive = (count == 3) || (count == 2 && alive);
                break;
            case 1: // HighLife B36/S23
                next_alive = (count == 3 || count == 6) || (count == 2 && alive);
                break;
            case 2: // Seeds B2/S
                next_alive = (count == 2) && !alive;
                break;
            case 3: // Day & Night B3678/S34678
                next_alive = (!alive && (count == 3 || count == 6 || count == 7 || count == 8)) ||
                            (alive && (count == 3 || count == 4 || count == 6 || count == 7 || count == 8));
                break;
            case 4: // Maze B3/S12345
                next_alive = (!alive && count == 3) || 
                            (alive && (count >= 1 && count <= 5));
                break;
            case 5: // Replicator B1357/S1357
                next_alive = (count == 1 || count == 3 || count == 5 || count == 7);
                break;
            case 6: // DryLife B37/S23
                next_alive = (!alive && (count == 3 || count == 7)) ||
                            (alive && (count == 2 || count == 3));
                break;
            case 7: // Live Free or Die B2/S0
                next_alive = (!alive && count == 2);
                break;
            case 8: // 2x2 B36/S125
                next_alive = (!alive && (count == 3 || count == 6)) ||
                            (alive && (count == 1 || count == 2 || count == 5));
                break;
            case 9: // Life Without Death B3/S012345678
                next_alive = (!alive && count == 3) || alive;
                break;
            default:
                next_alive = (count == 3) || (count == 2 && alive);
        }
        
        next[cell_idx] = next_alive ? 255 : 0;
        next[cell_idx + 1] = 0;
        next[cell_idx + 2] = 0;
        next[cell_idx + 3] = 0;
        
    } else if (rule_type == 1 && rule_id == 0) { // Brian's Brain
        // Count alive neighbors (state 255)
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                if (current[neighbor_idx] == 255) count++;
            }
        }
        
        unsigned char state = current[cell_idx];
        unsigned char next_state;
        
        if (state == 255) { // Alive -> Dying
            next_state = 128;
        } else if (state == 128) { // Dying -> Dead
            next_state = 0;
        } else if (count == 2) { // Dead -> Alive (birth with 2 neighbors)
            next_state = 255;
        } else {
            next_state = 0;
        }
        
        next[cell_idx] = next_state;
        next[cell_idx + 1] = 0;
        next[cell_idx + 2] = 0;
        next[cell_idx + 3] = 0;
        
    } else if (rule_type == 2 && rule_id == 0) { // Life Simulation
        // Collect neighbor health data
        int mental_sum = 0, body_sum = 0, social_sum = 0;
        int neighbor_count = 0;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                
                mental_sum += current[neighbor_idx];
                body_sum += current[neighbor_idx + 1];
                social_sum += current[neighbor_idx + 2];
                neighbor_count++;
            }
        }
        
        // Current cell values
        int mental = current[cell_idx];
        int body = current[cell_idx + 1];
        int social = current[cell_idx + 2];
        int money = current[cell_idx + 3];
        
        // Mental health: decreases in isolation, increases with social connections
        if (social_sum < 8 * 50) mental = max(0, mental - 2);
        if (social_sum > 8 * 150) mental = min(255, mental + 1);
        
        // Body health: slow decay, recovers with good mental health
        body = max(0, body - 1);
        if (mental > 200) body = min(255, body + 2);
        
        // Social health: spreads from neighbors, requires mental stability
        if (mental < 30) {
            social = max(0, social - 3);
        } else {
            social = (social + social_sum / 8) / 2;
        }
        
        // Money: accumulates with health
        if (mental + body > 300) money = min(255, money + 1);
        // Simple decay for money
        if ((x + y + blockIdx.x) % 10 == 0) money = max(0, money - 5);
        
        next[cell_idx] = mental;
        next[cell_idx + 1] = body;
        next[cell_idx + 2] = social;
        next[cell_idx + 3] = money;
    }
}
'''


def compile_simple_kernels():
    """Compile simplified CUDA kernels."""
    return {
        'simple_unified_step': cp.RawKernel(SIMPLE_UNIFIED_KERNEL, 'simple_unified_step'),
        'draw_resampled_stroke': cp.RawKernel(STROKE_KERNEL, 'draw_resampled_stroke')
    }