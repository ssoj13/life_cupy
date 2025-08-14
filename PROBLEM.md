# Drawing Disappearance and Black Screen Problem

## Problem Description

User reported that when drawing shapes with the painter tool, they would appear temporarily but then disappear after a simulation step, resulting in a black screen despite debug info showing 255 values in the field data.

## Root Cause Analysis

The issue was **NOT** with the simulation kernel clearing drawings, but with the **display pipeline bypassing proper field-to-RGBA conversion**.

### Technical Details

1. **Field Storage**: Drawings were correctly stored in field buffers as RGBA values (e.g., `[255, 0, 0, 255]`)
2. **Simulation Preservation**: The CUDA simulation kernel was correctly preserving drawing data
3. **Display Problem**: The `get_rgba_array()` method was bypassing the `field_to_rgba` CUDA kernel

### The Broken Code Path

```python
# In life_engine.py - BROKEN VERSION
def get_rgba_array(self) -> cp.ndarray:
    current = self.buffers[self.current_buffer]
    
    if self.display_mode == 0:  # RGB display - direct RGBA passthrough
        return current  # ← PROBLEM: Returns raw field data
```

### Why This Failed

- **Binary Rules (rule_type=0)**: Expected field data to be converted to grayscale
- **Raw Field Data**: Contained RGBA values like `[255, 0, 0, 255]` (red channel only)
- **Display Expected**: White pixels `[255, 255, 255, 255]` for alive cells
- **Result**: Red pixels appeared as black in binary rule visualization

### The Fix

Modified `get_rgba_array()` to properly call the `field_to_rgba` CUDA kernel:

```python
# FIXED VERSION
def get_rgba_array(self) -> cp.ndarray:
    current = self.buffers[self.current_buffer]
    
    # Use CUDA kernel to convert field to proper RGBA based on rule type and display mode
    total_pixels = self.width * self.height
    blocks = self._calculate_grid_size(total_pixels)
    
    self.kernels['field_to_rgba'](
        (blocks,), (self.threads_per_block,),
        (current.ravel(), self.rgba_buffer.ravel(),
         total_pixels, int(self.rule_type), int(self.display_mode))
    )
    
    return self.rgba_buffer
```

### CUDA Kernel Logic

The `field_to_rgba` kernel properly handles different rule types:

```c
if (rule_type == 0) { // Binary rules - show as black/white
    unsigned char value = cell.ch1;  // Only use channel 1
    rgba[rgba_idx + 0] = value;      // R = ch1
    rgba[rgba_idx + 1] = value;      // G = ch1  
    rgba[rgba_idx + 2] = value;      // B = ch1
    rgba[rgba_idx + 3] = 255;        // A = 255
}
```

## Lesson Learned

The debug info showing "255 values" was misleading - the values were there, but the display pipeline wasn't converting them properly for the current rule type. Always trace through the entire data flow from storage → processing → display when debugging visualization issues.

## Verification

After fix:
- Field data: `[255, 0, 0, 255]` (red channel drawing)
- RGBA output: `[255, 255, 255, 255]` (white pixel for display)
- Result: Drawings now visible and persist through simulation steps

## Files Modified

- `src/life/core/life_engine.py` - Fixed `get_rgba_array()` method to use proper CUDA conversion