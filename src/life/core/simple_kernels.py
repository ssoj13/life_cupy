"""Simplified CUDA kernels that work with 4-channel arrays."""
import cupy as cp
import numpy as np


# Anti-aliased line segment kernel (Option 2)
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

# Multi-segment stroke chain kernel (Option 4)
STROKE_CHAIN_KERNEL = r'''
extern "C" __global__
void draw_stroke_chain(unsigned char* field,
                      const float* points, const int num_points,
                      const float thickness,
                      const unsigned char ch1, const unsigned char ch2,
                      const unsigned char ch3, const unsigned char ch4,
                      const int width, const int height) {
    
    if (num_points < 2) return;
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate overall bounding box for entire stroke
    float radius = thickness * 0.5f;
    float min_x_f = points[0], max_x_f = points[0];
    float min_y_f = points[1], max_y_f = points[1];
    
    for (int i = 1; i < num_points; i++) {
        float x = points[i * 2];
        float y = points[i * 2 + 1];
        min_x_f = fminf(min_x_f, x);
        max_x_f = fmaxf(max_x_f, x);
        min_y_f = fminf(min_y_f, y);
        max_y_f = fmaxf(max_y_f, y);
    }
    
    int min_x = max(0, (int)floorf(min_x_f - radius - 1));
    int max_x = min(width - 1, (int)ceilf(max_x_f + radius + 1));
    int min_y = max(0, (int)floorf(min_y_f - radius - 1));
    int max_y = min(height - 1, (int)ceilf(max_y_f + radius + 1));
    
    int total_pixels = (max_x - min_x + 1) * (max_y - min_y + 1);
    if (thread_id >= total_pixels) return;
    
    // Convert thread ID to pixel coordinates
    int box_width = max_x - min_x + 1;
    int local_x = thread_id % box_width;
    int local_y = thread_id / box_width;
    int pixel_x = min_x + local_x;
    int pixel_y = min_y + local_y;
    
    // Find minimum distance to any line segment in the stroke
    float min_distance = 1e6f;
    
    for (int i = 0; i < num_points - 1; i++) {
        float x1 = points[i * 2];
        float y1 = points[i * 2 + 1];
        float x2 = points[(i + 1) * 2];
        float y2 = points[(i + 1) * 2 + 1];
        
        // Distance to this line segment
        float dx = x2 - x1;
        float dy = y2 - y1;
        float length_sq = dx * dx + dy * dy;
        
        float distance;
        if (length_sq < 0.0001f) {
            // Point segment
            float px = pixel_x + 0.5f - x1;
            float py = pixel_y + 0.5f - y1;
            distance = sqrtf(px * px + py * py);
        } else {
            // Line segment
            float px = pixel_x + 0.5f - x1;
            float py = pixel_y + 0.5f - y1;
            float t = fmaxf(0.0f, fminf(1.0f, (px * dx + py * dy) / length_sq));
            float closest_x = x1 + t * dx;
            float closest_y = y1 + t * dy;
            float dist_x = pixel_x + 0.5f - closest_x;
            float dist_y = pixel_y + 0.5f - closest_y;
            distance = sqrtf(dist_x * dist_x + dist_y * dist_y);
        }
        
        min_distance = fminf(min_distance, distance);
    }
    
    // Anti-aliased coverage calculation
    float coverage = 1.0f - fmaxf(0.0f, fminf(1.0f, min_distance - radius + 0.5f));
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

# Immediate drawing kernel for real-time mode
IMMEDIATE_DRAW_KERNEL = r'''
extern "C" __global__
void draw_immediate_circle(unsigned char* field,
                          const int center_x, const int center_y, const int radius,
                          const unsigned char ch1, const unsigned char ch2,
                          const unsigned char ch3, const unsigned char ch4,
                          const int width, const int height) {
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the bounding box of the circle
    int min_x = max(0, center_x - radius);
    int max_x = min(width - 1, center_x + radius);
    int min_y = max(0, center_y - radius);
    int max_y = min(height - 1, center_y + radius);
    
    int total_pixels = (max_x - min_x + 1) * (max_y - min_y + 1);
    
    if (thread_id >= total_pixels) return;
    
    // Convert thread ID to pixel coordinates within bounding box
    int box_width = max_x - min_x + 1;
    int local_x = thread_id % box_width;
    int local_y = thread_id / box_width;
    
    int pixel_x = min_x + local_x;
    int pixel_y = min_y + local_y;
    
    // Check if pixel is within circle
    int dx = pixel_x - center_x;
    int dy = pixel_y - center_y;
    int distance_sq = dx * dx + dy * dy;
    int radius_sq = radius * radius;
    
    if (distance_sq <= radius_sq) {
        // Pixel is inside circle, draw it
        int idx = (pixel_y * width + pixel_x) * 4;
        field[idx] = ch1;
        field[idx + 1] = ch2;
        field[idx + 2] = ch3;
        field[idx + 3] = ch4;
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
    
    // Unified RGBA system - all kernels work with RGBA input/output
    int cell_idx = (y * width + x) * 4;
    
    if (rule_type == 0) { // Binary B/S rules - 4 independent RGBA channels!
        // Process each RGBA channel independently
        for (int channel = 0; channel < 4; channel++) {
            // Count alive neighbors for this channel
            int count = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = (x + dx + width) % width;
                    int ny = (y + dy + height) % height;
                    int neighbor_idx = (ny * width + nx) * 4;
                    if (current[neighbor_idx + channel] > 128) count++; // Threshold for alive
                }
            }
            
            // Apply rule to this channel
            bool alive = current[cell_idx + channel] > 128;
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
            
            // Set this channel's value - preserve original color intensity
            if (next_alive) {
                // Keep the original color value when alive
                next[cell_idx + channel] = current[cell_idx + channel] > 0 ? current[cell_idx + channel] : 255;
            } else {
                // Dead = 0
                next[cell_idx + channel] = 0;
            }
        }
        
    } else if (rule_type == 1 && rule_id == 0) { // Brian's Brain - RGBA interpretation
        // Count alive neighbors (any component == 255 = alive)
        int count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                // Alive if any channel is at max (255)
                if (current[neighbor_idx] == 255 || current[neighbor_idx + 1] == 255 || 
                    current[neighbor_idx + 2] == 255 || current[neighbor_idx + 3] == 255) {
                    count++;
                }
            }
        }
        
        // Determine current state based on max RGBA value
        unsigned char max_val = max(max(current[cell_idx], current[cell_idx + 1]), 
                                   max(current[cell_idx + 2], current[cell_idx + 3]));
        unsigned char next_state;
        
        if (max_val == 255) { // Alive -> Dying
            next_state = 128;
        } else if (max_val == 128) { // Dying -> Dead
            next_state = 0;
        } else if (count == 2) { // Dead -> Alive (birth with 2 neighbors)
            next_state = 255;
        } else {
            next_state = 0;
        }
        
        // Apply state to all RGBA channels equally for Brian's Brain
        next[cell_idx] = next_state;
        next[cell_idx + 1] = next_state;
        next[cell_idx + 2] = next_state;
        next[cell_idx + 3] = next_state;
        
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
        
        next[cell_idx] = mental;
        next[cell_idx + 1] = body;
        next[cell_idx + 2] = social;
        next[cell_idx + 3] = money;
        
    } else if (rule_type == 3) { // Extended Classic - RGB layers with independent rules
        // Each RGB channel follows cellular automata independently
        // Red channel
        int r_count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                if (current[neighbor_idx] > 128) r_count++; // Red channel threshold
            }
        }
        
        // Green channel  
        int g_count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                if (current[neighbor_idx + 1] > 128) g_count++; // Green channel threshold
            }
        }
        
        // Blue channel
        int b_count = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                int neighbor_idx = (ny * width + nx) * 4;
                if (current[neighbor_idx + 2] > 128) b_count++; // Blue channel threshold
            }
        }
        
        // Apply Conway's Life rule to each channel independently
        bool r_alive = current[cell_idx] > 128;
        bool g_alive = current[cell_idx + 1] > 128;
        bool b_alive = current[cell_idx + 2] > 128;
        
        bool r_next = (r_count == 3) || (r_count == 2 && r_alive);
        bool g_next = (g_count == 3) || (g_count == 2 && g_alive);
        bool b_next = (b_count == 3) || (b_count == 2 && b_alive);
        
        next[cell_idx] = r_next ? 255 : 0;
        next[cell_idx + 1] = g_next ? 255 : 0;
        next[cell_idx + 2] = b_next ? 255 : 0;
        next[cell_idx + 3] = 255; // Alpha always full
    }
}
'''


def compile_simple_kernels():
    """Compile simplified CUDA kernels."""
    return {
        'simple_unified_step': cp.RawKernel(SIMPLE_UNIFIED_KERNEL, 'simple_unified_step'),
        'draw_antialiased_line': cp.RawKernel(ANTIALIASED_LINE_KERNEL, 'draw_antialiased_line'),
        'draw_stroke_chain': cp.RawKernel(STROKE_CHAIN_KERNEL, 'draw_stroke_chain'),
        'draw_immediate_circle': cp.RawKernel(IMMEDIATE_DRAW_KERNEL, 'draw_immediate_circle')
    }