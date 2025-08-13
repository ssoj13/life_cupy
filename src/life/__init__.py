"""Conway's Game of Life - CUDA/CuPy Implementation."""

__version__ = "0.1.0"
__author__ = "Life Game"

from .core.life_engine import GameOfLifeEngine
from .gui.main_window import MainWindow

__all__ = ['GameOfLifeEngine', 'MainWindow', '__version__']