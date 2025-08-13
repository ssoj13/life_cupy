"""Simplified CUDA kernels that work with 4-channel arrays."""
import cupy as cp
import numpy as np


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
        'simple_unified_step': cp.RawKernel(SIMPLE_UNIFIED_KERNEL, 'simple_unified_step')
    }