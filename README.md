# Conway's Game of Life - CUDA/CuPy Implementation

A high-performance implementation of Conway's Game of Life using CUDA (via CuPy) for GPU acceleration and PySide6 for the GUI.

## Features

- **GPU Acceleration**: Uses CUDA kernels through CuPy for blazing-fast simulation
- **Interactive Drawing**: Left-click to draw, right-click to erase
- **Resizable Field**: Dynamic field sizing up to 4096×4096 cells
- **Drawing Tools**: Circle brush with adjustable radius and noise pattern generator
- **Real-time Controls**: Play/pause, step, reset with keyboard shortcuts
- **Smooth Navigation**: Pan (middle-click drag) and zoom (scroll wheel)
- **Performance Optimized**: Double buffering, minimal CPU-GPU transfers

## Requirements

- Python 3.8+
- CUDA-capable GPU with CUDA 11.2+ installed
- Windows/Linux/macOS

## Installation

1. Install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Note: Choose the appropriate CuPy version for your CUDA version:
- CUDA 11.x: `pip install cupy-cuda11x`
- CUDA 12.x: `pip install cupy-cuda12x`

## Usage

Run the application:
```bash
python main.py
```

### Controls

- **Left Click**: Draw living cells
- **Right Click**: Erase cells
- **Middle Click + Drag**: Pan view
- **Scroll Wheel**: Zoom in/out
- **Space**: Play/Pause simulation
- **S**: Single step
- **R**: Reset field
- **N**: Add random noise
- **C**: Clear field

### Toolbar

- **Play/Pause**: Start/stop simulation
- **Step**: Advance one generation
- **Reset**: Clear field and reset generation counter
- **Speed Slider**: Adjust simulation speed (steps per frame)
- **Add Noise**: Fill field with random pattern
- **Clear**: Clear all cells

### Drawing Tools Panel

- **Brush Radius**: Adjust drawing brush size (1-50 cells)
- **Noise Density**: Set density for random pattern (0-100%)
- **Field Size**: Resize the simulation field

## Architecture

The application uses:
- **CuPy RawKernel**: Custom CUDA kernels for Game of Life rules
- **Double Buffering**: Two GPU buffers for current/next state
- **OpenGL Integration**: Direct GPU texture rendering
- **Grid-Stride Loops**: Handles arbitrary field sizes efficiently

## Performance

- Supports fields up to millions of cells
- 60+ FPS on modern GPUs
- Minimal CPU-GPU memory transfers
- Optimized kernels without warp divergence
