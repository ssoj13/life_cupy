"""Game of Life engine using CuPy and CUDA kernels."""
import cupy as cp
import numpy as np
from typing import Tuple, Optional
import math
import time

from .cuda_kernels import compile_kernels
from ..utils.config import Config


class GameOfLifeEngine:
    """GPU-accelerated Game of Life simulation engine."""
    
    def __init__(self, width: int = Config.DEFAULT_FIELD_WIDTH, 
                 height: int = Config.DEFAULT_FIELD_HEIGHT):
        """Initialize the Game of Life engine.
        
        Args:
            width: Field width in cells
            height: Field height in cells
        """
        self.width = width
        self.height = height
        self.generation = 0
        self.is_running = False
        
        # Compile CUDA kernels
        self.kernels = compile_kernels()
        
        # Initialize double buffer for simulation
        self.current_buffer = 0
        self.buffers = [
            cp.zeros((height, width), dtype=cp.uint8),
            cp.zeros((height, width), dtype=cp.uint8)
        ]
        
        # RGBA buffer for display
        self.rgba_buffer = cp.zeros((height, width, 4), dtype=cp.uint8)
        
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
    
    def step(self, steps: int = 1) -> None:
        """Advance simulation by specified number of steps.
        
        Args:
            steps: Number of simulation steps to perform
        """
        for _ in range(steps):
            current = self.buffers[self.current_buffer]
            next_buffer = self.buffers[1 - self.current_buffer]
            
            # Launch life step kernel
            self.kernels['life_step'](
                (self.blocks_per_grid,), (self.threads_per_block,),
                (current.ravel(), next_buffer.ravel(), 
                 self.width, self.height)
            )
            
            # Swap buffers
            self.current_buffer = 1 - self.current_buffer
            self.generation += 1
    
    def reset(self) -> None:
        """Reset the field to empty state."""
        size = self.width * self.height
        blocks = self._calculate_grid_size(size)
        
        for buffer in self.buffers:
            self.kernels['clear_field'](
                (blocks,), (self.threads_per_block,),
                (buffer.ravel(), size)
            )
        
        self.generation = 0
        self.current_buffer = 0
    
    def draw_circle(self, x: int, y: int, radius: int, value: bool = True) -> None:
        """Draw a filled circle on the field.
        
        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Circle radius in cells
            value: True to set cells alive, False to kill them
        """
        current = self.buffers[self.current_buffer]
        blocks = self._calculate_grid_size(self.width * self.height)
        
        self.kernels['draw_circle'](
            (blocks,), (self.threads_per_block,),
            (current.ravel(), self.width, self.height,
             x, y, radius, cp.uint8(1 if value else 0))
        )
    
    def add_noise(self, density: float = 0.3, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Add random noise pattern to the field.
        
        Args:
            density: Probability of cell being alive (0.0 to 1.0)
            region: Optional (x, y, width, height) to limit noise to region
        """
        current = self.buffers[self.current_buffer]
        
        if region:
            x, y, w, h = region
            # Create temporary buffer for region
            temp = cp.random.random((h, w)) < density
            current[y:y+h, x:x+w] = temp.astype(cp.uint8)
        else:
            # Use kernel for full field noise
            blocks = self._calculate_grid_size(self.width * self.height)
            seed = int(time.time() * 1000000) % (2**32)
            
            self.kernels['add_noise'](
                (blocks,), (self.threads_per_block,),
                (current.ravel(), self.width, self.height,
                 cp.float32(density), cp.uint64(seed))
            )
    
    def get_rgba_array(self) -> cp.ndarray:
        """Get current field as RGBA array for display.
        
        Returns:
            RGBA array suitable for OpenGL texture
        """
        current = self.buffers[self.current_buffer]
        size = self.width * self.height
        blocks = self._calculate_grid_size(size)
        
        self.kernels['field_to_rgba'](
            (blocks,), (self.threads_per_block,),
            (current.ravel(), self.rgba_buffer.ravel(), size)
        )
        
        return self.rgba_buffer
    
    def get_field_cpu(self) -> np.ndarray:
        """Get current field as NumPy array on CPU.
        
        Returns:
            Current field as NumPy array
        """
        return cp.asnumpy(self.buffers[self.current_buffer])
    
    def set_field(self, field: np.ndarray) -> None:
        """Set field from NumPy array.
        
        Args:
            field: NumPy array with field data
        """
        if field.shape != (self.height, self.width):
            raise ValueError(f"Field shape {field.shape} doesn't match engine size ({self.height}, {self.width})")
        
        self.buffers[self.current_buffer] = cp.asarray(field, dtype=cp.uint8)
    
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
            cp.zeros((height, width), dtype=cp.uint8),
            cp.zeros((height, width), dtype=cp.uint8)
        ]
        self.rgba_buffer = cp.zeros((height, width, 4), dtype=cp.uint8)
        
        # Copy old data (crop or pad as needed)
        min_h = min(old_field.shape[0], height)
        min_w = min(old_field.shape[1], width)
        self.buffers[0][:min_h, :min_w] = old_field[:min_h, :min_w]
        
        # Update grid dimensions
        self.blocks_per_grid = self._calculate_grid_size(width * height)
        self.current_buffer = 0
    
    def set_pattern(self, pattern: np.ndarray, x: int, y: int) -> None:
        """Place a pattern at specified position.
        
        Args:
            pattern: 2D array with pattern (1 for alive, 0 for dead)
            x: Top-left X coordinate
            y: Top-left Y coordinate
        """
        pattern_gpu = cp.asarray(pattern, dtype=cp.uint8)
        h, w = pattern.shape
        
        # Clip pattern to field boundaries
        x_end = min(x + w, self.width)
        y_end = min(y + h, self.height)
        x_start = max(0, x)
        y_start = max(0, y)
        
        # Calculate pattern offsets if placement is partially outside
        pattern_x_start = max(0, -x)
        pattern_y_start = max(0, -y)
        pattern_x_end = pattern_x_start + (x_end - x_start)
        pattern_y_end = pattern_y_start + (y_end - y_start)
        
        # Place pattern
        self.buffers[self.current_buffer][y_start:y_end, x_start:x_end] = \
            pattern_gpu[pattern_y_start:pattern_y_end, pattern_x_start:pattern_x_end]
    
    @property
    def current_field(self) -> cp.ndarray:
        """Get current field buffer."""
        return self.buffers[self.current_buffer]