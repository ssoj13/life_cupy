"""Configuration constants for Game of Life application."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Application configuration."""
    
    # Window settings
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800
    WINDOW_TITLE: str = "Conway's Game of Life - CUDA"
    
    # Field settings
    DEFAULT_FIELD_WIDTH: int = 800
    DEFAULT_FIELD_HEIGHT: int = 600
    MAX_FIELD_SIZE: int = 4096
    MIN_FIELD_SIZE: int = 100
    
    # Simulation settings
    DEFAULT_FPS: int = 60
    MAX_STEPS_PER_FRAME: int = 10
    DEFAULT_STEPS_PER_FRAME: int = 1
    
    # Drawing tools
    DEFAULT_BRUSH_RADIUS: int = 5
    MAX_BRUSH_RADIUS: int = 50
    MIN_BRUSH_RADIUS: int = 1
    DEFAULT_NOISE_DENSITY: float = 0.3
    
    # Colors (RGBA)
    ALIVE_COLOR: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    DEAD_COLOR: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    GRID_COLOR: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    
    # CUDA settings
    THREADS_PER_BLOCK: int = 256
    USE_SHARED_MEMORY: bool = True
    
    # Performance
    USE_DOUBLE_BUFFER: bool = True
    ENABLE_PROFILING: bool = False