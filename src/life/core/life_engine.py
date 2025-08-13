"""Multi-channel cellular automata engine using CuPy and CUDA kernels."""
import cupy as cp
import numpy as np
from typing import Tuple, Optional
import math
import time

from .cuda_kernels import (RuleType, BinaryRule, MultiStateRule, MultiChannelRule, get_bs_tables)
from .simple_kernels import compile_simple_kernels
from ..utils.config import Config


class MultiChannelEngine:
    """GPU-accelerated multi-channel cellular automata simulation engine."""
    
    def __init__(self, width: int = Config.DEFAULT_FIELD_WIDTH, 
                 height: int = Config.DEFAULT_FIELD_HEIGHT):
        """Initialize the multi-channel engine.
        
        Args:
            width: Field width in cells
            height: Field height in cells
        """
        self.width = width
        self.height = height
        self.generation = 0
        self.is_running = False
        
        # Current rule settings
        self.rule_type = RuleType.BINARY_BS
        self.rule_id = BinaryRule.CONWAY_LIFE
        self.display_mode = 0  # 0=RGB, 1=Money as brightness
        
        # Compile simplified CUDA kernels
        self.kernels = compile_simple_kernels()
        
        # Initialize double buffer for simulation (4-channel cells as 4D arrays)
        self.current_buffer = 0
        self.buffers = [
            cp.zeros((height, width, 4), dtype=cp.uint8),
            cp.zeros((height, width, 4), dtype=cp.uint8)
        ]
        
        # RGBA buffer for display
        self.rgba_buffer = cp.zeros((height, width, 4), dtype=cp.uint8)
        
        # Clipboard buffers for save/restore
        self.clipboards = [
            cp.zeros((height, width, 4), dtype=cp.uint8) for _ in range(4)
        ]
        
        # Calculate grid dimensions for kernels
        self.threads_per_block = Config.THREADS_PER_BLOCK
        self.blocks_per_grid = self._calculate_grid_size(width * height)
        
    def _calculate_grid_size(self, total_elements: int) -> int:
        """Calculate optimal grid size for CUDA kernel launch.
        
        Args:
            total_elements: Total number of elements to process
            
        Returns:
            Number of blocks in grid
        """
        return math.ceil(total_elements / self.threads_per_block)
    
    def set_rule(self, rule_type: RuleType, rule_id: int):
        """Set the current cellular automata rule.
        
        Args:
            rule_type: Type of rule (binary, multistate, multichannel)
            rule_id: Specific rule within the type
        """
        self.rule_type = rule_type
        self.rule_id = rule_id
        
        # Reset field when switching rule types
        if rule_type != self.rule_type:
            self.reset()
    
    def step(self, steps: int = 1) -> None:
        """Advance simulation by specified number of steps.
        
        Args:
            steps: Number of simulation steps to perform
        """
        for _ in range(steps):
            current = self.buffers[self.current_buffer]
            next_buffer = self.buffers[1 - self.current_buffer]
            
            # Launch simplified unified kernel
            blocks = self._calculate_grid_size(self.width * self.height)
            self.kernels['simple_unified_step'](
                (blocks,), (self.threads_per_block,),
                (current.ravel(), next_buffer.ravel(), 
                 self.width, self.height, 
                 int(self.rule_type), int(self.rule_id))
            )
            
            # Swap buffers
            self.current_buffer = 1 - self.current_buffer
            self.generation += 1
    
    def reset(self) -> None:
        """Reset the field to empty state."""
        # Clear both buffers
        for buffer in self.buffers:
            buffer.fill(0)
        
        self.generation = 0
        self.current_buffer = 0
    
    def draw_circle(self, x: int, y: int, radius: int, value: Tuple[int, int, int, int] = (255, 0, 0, 0)) -> None:
        """Draw a filled circle on the field.
        
        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Circle radius in cells
            value: 4-channel values (ch1, ch2, ch3, ch4)
        """
        # Simple CPU-based circle drawing for now
        radius_sq = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius_sq:
                    px = x + dx
                    py = y + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.buffers[self.current_buffer][py, px, 0] = value[0]
                        self.buffers[self.current_buffer][py, px, 1] = value[1]
                        self.buffers[self.current_buffer][py, px, 2] = value[2]
                        self.buffers[self.current_buffer][py, px, 3] = value[3]
    
    def draw_line_segment(self, x1: float, y1: float, x2: float, y2: float, 
                         thickness: float, value: Tuple[int, int, int, int] = (255, 0, 0, 0)) -> None:
        """Draw an anti-aliased line segment using GPU.
        
        Args:
            x1, y1: Start point coordinates
            x2, y2: End point coordinates  
            thickness: Line thickness in pixels
            value: 4-channel values (ch1, ch2, ch3, ch4)
        """
        # Calculate bounding box for optimal thread allocation
        radius = thickness * 0.5
        min_x = max(0, int(min(x1, x2) - radius - 1))
        max_x = min(self.width - 1, int(max(x1, x2) + radius + 1))
        min_y = max(0, int(min(y1, y2) - radius - 1))
        max_y = min(self.height - 1, int(max(y1, y2) + radius + 1))
        
        total_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
        if total_pixels <= 0:
            return
            
        # Launch anti-aliased line kernel
        threads_per_block = min(256, total_pixels)
        blocks = max(1, (total_pixels + threads_per_block - 1) // threads_per_block)
        
        self.kernels['draw_antialiased_line'](
            (blocks,), (threads_per_block,),
            (self.buffers[self.current_buffer].ravel(),  # field
             cp.float32(x1),                             # x1
             cp.float32(y1),                             # y1
             cp.float32(x2),                             # x2
             cp.float32(y2),                             # y2
             cp.float32(thickness),                      # thickness
             cp.uint8(value[0]),                         # ch1
             cp.uint8(value[1]),                         # ch2
             cp.uint8(value[2]),                         # ch3
             cp.uint8(value[3]),                         # ch4
             cp.int32(self.width),                       # width
             cp.int32(self.height))                      # height
        )
    
    def draw_stroke_chain(self, stroke_points: list, thickness: float,
                         value: Tuple[int, int, int, int] = (255, 0, 0, 0)) -> None:
        """Draw an entire stroke chain with anti-aliasing using GPU batch processing.
        
        Args:
            stroke_points: List of (x, y) tuples defining the stroke path
            thickness: Line thickness in pixels
            value: 4-channel values (ch1, ch2, ch3, ch4)
        """
        if len(stroke_points) < 2:
            return
            
        # Convert stroke points to flat array format for GPU
        points_array = []
        for x, y in stroke_points:
            points_array.extend([float(x), float(y)])
        
        # Convert to CuPy array
        points_gpu = cp.array(points_array, dtype=cp.float32)
        
        # Calculate overall bounding box
        radius = thickness * 0.5
        min_x = min(p[0] for p in stroke_points) - radius - 1
        max_x = max(p[0] for p in stroke_points) + radius + 1
        min_y = min(p[1] for p in stroke_points) - radius - 1
        max_y = max(p[1] for p in stroke_points) + radius + 1
        
        min_x = max(0, int(min_x))
        max_x = min(self.width - 1, int(max_x))
        min_y = max(0, int(min_y))
        max_y = min(self.height - 1, int(max_y))
        
        total_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
        if total_pixels <= 0:
            return
            
        # Launch stroke chain kernel
        threads_per_block = min(256, total_pixels)
        blocks = max(1, (total_pixels + threads_per_block - 1) // threads_per_block)
        
        self.kernels['draw_stroke_chain'](
            (blocks,), (threads_per_block,),
            (self.buffers[self.current_buffer].ravel(),  # field
             points_gpu,                                  # points
             len(stroke_points),                         # num_points
             cp.float32(thickness),                      # thickness
             cp.uint8(value[0]),                         # ch1
             cp.uint8(value[1]),                         # ch2
             cp.uint8(value[2]),                         # ch3
             cp.uint8(value[3]),                         # ch4
             cp.int32(self.width),                       # width
             cp.int32(self.height))                      # height
        )
    
    def draw_circle_immediate(self, x: int, y: int, radius: int, 
                             value: Tuple[int, int, int, int] = (255, 0, 0, 0)) -> None:
        """Draw a circle immediately using the fast GPU kernel.
        
        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Circle radius in cells
            value: 4-channel values (ch1, ch2, ch3, ch4)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
            
        # Calculate number of threads needed (bounding box area)
        min_x = max(0, x - radius)
        max_x = min(self.width - 1, x + radius)
        min_y = max(0, y - radius)
        max_y = min(self.height - 1, y + radius)
        
        total_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
        
        if total_pixels <= 0:
            return
            
        # Launch immediate drawing kernel
        threads_per_block = min(256, total_pixels)
        blocks = max(1, (total_pixels + threads_per_block - 1) // threads_per_block)
        
        self.kernels['draw_immediate_circle'](
            (blocks,), (threads_per_block,),
            (self.buffers[self.current_buffer].ravel(),  # field
             cp.int32(x),                                # center_x
             cp.int32(y),                                # center_y
             cp.int32(radius),                           # radius
             cp.uint8(value[0]),                         # ch1
             cp.uint8(value[1]),                         # ch2
             cp.uint8(value[2]),                         # ch3
             cp.uint8(value[3]),                         # ch4
             cp.int32(self.width),                       # width
             cp.int32(self.height))                      # height
        )
    
    def add_noise(self, density: float = 0.3, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Add random noise pattern to the field.
        
        Args:
            density: Probability of cell being affected (0.0 to 1.0)
            region: Optional (x, y, width, height) to limit noise to region
        """
        current = self.buffers[self.current_buffer]
        
        # Simple CPU-based noise generation
        if self.rule_type == 0 or self.rule_type == 1:  # Binary or multistate
            # Random binary noise
            noise = cp.random.random((self.height, self.width)) < density
            current[:, :, 0] = cp.where(noise, 255, current[:, :, 0])
            current[:, :, 1] = 0
            current[:, :, 2] = 0
            current[:, :, 3] = 0
        else:  # Multichannel
            # Random values for all channels
            noise_mask = cp.random.random((self.height, self.width)) < density
            current[:, :, 0] = cp.where(noise_mask, cp.random.randint(150, 256, (self.height, self.width)), current[:, :, 0])
            current[:, :, 1] = cp.where(noise_mask, cp.random.randint(150, 256, (self.height, self.width)), current[:, :, 1])
            current[:, :, 2] = cp.where(noise_mask, cp.random.randint(150, 256, (self.height, self.width)), current[:, :, 2])
            current[:, :, 3] = cp.where(noise_mask, cp.random.randint(100, 201, (self.height, self.width)), current[:, :, 3])
    
    def get_rgba_array(self) -> cp.ndarray:
        """Get current field as RGBA array for display.
        
        Returns:
            RGBA array suitable for OpenGL texture
        """
        current = self.buffers[self.current_buffer]
        
        if self.display_mode == 0:  # RGB display
            if self.rule_type == 0 or self.rule_type == 1:  # Binary/multistate - show as black/white
                value = current[:, :, 0]
                self.rgba_buffer[:, :, 0] = value
                self.rgba_buffer[:, :, 1] = value
                self.rgba_buffer[:, :, 2] = value
                self.rgba_buffer[:, :, 3] = 255
            else:  # Multichannel
                self.rgba_buffer[:, :, 0] = current[:, :, 0]  # Mental -> R
                self.rgba_buffer[:, :, 1] = current[:, :, 1]  # Body -> G
                self.rgba_buffer[:, :, 2] = current[:, :, 2]  # Social -> B
                self.rgba_buffer[:, :, 3] = 255
                
        elif self.display_mode == 1:  # Money as brightness
            avg = (current[:, :, 0] + current[:, :, 1] + current[:, :, 2]) // 3
            brightness = (avg * current[:, :, 3]) // 255
            self.rgba_buffer[:, :, 0] = brightness
            self.rgba_buffer[:, :, 1] = brightness
            self.rgba_buffer[:, :, 2] = brightness
            self.rgba_buffer[:, :, 3] = 255
            
        elif self.display_mode == 2:  # Health heatmap
            # Average health as heatmap (red = low, yellow = medium, green = high)
            avg_health = (current[:, :, 0] + current[:, :, 1] + current[:, :, 2]) // 3
            self.rgba_buffer[:, :, 0] = 255 - avg_health  # Red decreases with health
            self.rgba_buffer[:, :, 1] = avg_health  # Green increases with health
            self.rgba_buffer[:, :, 2] = 0
            self.rgba_buffer[:, :, 3] = 255
            
        elif self.display_mode == 3:  # Channel 1 only
            value = current[:, :, 0]
            self.rgba_buffer[:, :, 0] = value
            self.rgba_buffer[:, :, 1] = 0
            self.rgba_buffer[:, :, 2] = 0
            self.rgba_buffer[:, :, 3] = 255
            
        elif self.display_mode == 4:  # Channel 2 only
            value = current[:, :, 1]
            self.rgba_buffer[:, :, 0] = 0
            self.rgba_buffer[:, :, 1] = value
            self.rgba_buffer[:, :, 2] = 0
            self.rgba_buffer[:, :, 3] = 255
            
        elif self.display_mode == 5:  # Channel 3 only
            value = current[:, :, 2]
            self.rgba_buffer[:, :, 0] = 0
            self.rgba_buffer[:, :, 1] = 0
            self.rgba_buffer[:, :, 2] = value
            self.rgba_buffer[:, :, 3] = 255
            
        elif self.display_mode == 6:  # Channel 4 only
            value = current[:, :, 3]
            self.rgba_buffer[:, :, 0] = value
            self.rgba_buffer[:, :, 1] = value
            self.rgba_buffer[:, :, 2] = value
            self.rgba_buffer[:, :, 3] = 255
        else:
            # Default to first channel
            value = current[:, :, 0]
            self.rgba_buffer[:, :, 0] = value
            self.rgba_buffer[:, :, 1] = value
            self.rgba_buffer[:, :, 2] = value
            self.rgba_buffer[:, :, 3] = 255
        
        return self.rgba_buffer
    
    def get_field_cpu(self) -> np.ndarray:
        """Get current field as NumPy array on CPU.
        
        Returns:
            Current field as structured NumPy array
        """
        return cp.asnumpy(self.buffers[self.current_buffer])
    
    def set_field(self, field: np.ndarray) -> None:
        """Set field from NumPy array.
        
        Args:
            field: NumPy structured array with field data
        """
        if field.shape != (self.height, self.width):
            raise ValueError(f"Field shape {field.shape} doesn't match engine size ({self.height}, {self.width})")
        
        self.buffers[self.current_buffer] = cp.asarray(field)
    
    def resize(self, width: int, height: int) -> None:
        """Resize the field, preserving existing cells where possible.
        
        Args:
            width: New width in cells
            height: New height in cells
        """
        # Save current state
        old_field = self.buffers[self.current_buffer]
        
        # Update dimensions
        self.width = width
        self.height = height
        
        # Recreate buffers
        self.buffers = [
            cp.zeros((height, width, 4), dtype=cp.uint8),
            cp.zeros((height, width, 4), dtype=cp.uint8)
        ]
        self.rgba_buffer = cp.zeros((height, width, 4), dtype=cp.uint8)
        
        # Recreate clipboards
        self.clipboards = [
            cp.zeros((height, width, 4), dtype=cp.uint8) for _ in range(4)
        ]
        
        # Copy old data (crop or pad as needed)
        min_h = min(old_field.shape[0], height)
        min_w = min(old_field.shape[1], width)
        self.buffers[0][:min_h, :min_w, :] = old_field[:min_h, :min_w, :]
        
        # Update grid dimensions
        self.blocks_per_grid = self._calculate_grid_size(width * height)
        self.current_buffer = 0
    
    def save_to_clipboard(self, slot: int) -> None:
        """Save current field to clipboard slot.
        
        Args:
            slot: Clipboard slot (0-3)
        """
        if 0 <= slot < 4:
            cp.copyto(self.clipboards[slot], self.buffers[self.current_buffer])
    
    def load_from_clipboard(self, slot: int) -> None:
        """Load field from clipboard slot.
        
        Args:
            slot: Clipboard slot (0-3)
        """
        if 0 <= slot < 4:
            cp.copyto(self.buffers[self.current_buffer], self.clipboards[slot])
    
    def set_display_mode(self, mode: int) -> None:
        """Set display mode for multichannel visualization.
        
        Args:
            mode: 0=RGB display, 1=Money as brightness
        """
        self.display_mode = mode
    
    @property
    def current_field(self) -> cp.ndarray:
        """Get current field buffer."""
        return self.buffers[self.current_buffer]
    
    @property
    def rule_name(self) -> str:
        """Get human-readable name of current rule."""
        if self.rule_type == RuleType.BINARY_BS:
            rule_names = {
                BinaryRule.CONWAY_LIFE: "Conway's Life (B3/S23)",
                BinaryRule.HIGHLIFE: "HighLife (B36/S23)",
                BinaryRule.SEEDS: "Seeds (B2/S)",
                BinaryRule.DAY_NIGHT: "Day & Night (B3678/S34678)",
                BinaryRule.MAZE: "Maze (B3/S12345)",
                BinaryRule.REPLICATOR: "Replicator (B1357/S1357)",
                BinaryRule.DRYLIFE: "DryLife (B37/S23)",
                BinaryRule.LIVE_FREE_DIE: "Live Free or Die (B2/S0)",
                BinaryRule.RULE_2X2: "2x2 (B36/S125)",
                BinaryRule.LIFE_WITHOUT_DEATH: "Life Without Death (B3/S012345678)"
            }
            return rule_names.get(self.rule_id, "Unknown Binary Rule")
        elif self.rule_type == RuleType.MULTISTATE:
            if self.rule_id == MultiStateRule.BRIANS_BRAIN:
                return "Brian's Brain"
            return "Unknown Multi-State Rule"
        elif self.rule_type == RuleType.MULTICHANNEL:
            if self.rule_id == MultiChannelRule.LIFE_SIMULATION:
                return "Life Simulation (Health/Money)"
            return "Unknown Multi-Channel Rule"
        return "Unknown Rule"


# Backward compatibility alias
GameOfLifeEngine = MultiChannelEngine