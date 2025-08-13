"""OpenGL widget for rendering Game of Life field."""
import numpy as np
import cupy as cp
from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QMouseEvent, QWheelEvent
from OpenGL.GL import *
from typing import Optional

from ..core.life_engine import MultiChannelEngine
from ..utils.config import Config


class LifeGLWidget(QOpenGLWidget):
    """OpenGL widget for rendering and interacting with Game of Life field."""
    
    generation_updated = Signal(int)
    field_resized = Signal(int, int)  # width, height
    
    def __init__(self, parent=None):
        """Initialize the OpenGL widget."""
        super().__init__(parent)
        
        # Engine
        self.engine = MultiChannelEngine()
        
        # Display settings
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Interaction state
        self.drawing = False
        self.erasing = False
        self.last_draw_pos: Optional[QPoint] = None
        self.stroke_points = []  # Store stroke points for batch drawing
        self.current_draw_value = None  # Value to draw with current stroke
        self.brush_radius = Config.DEFAULT_BRUSH_RADIUS
        self.noise_density = Config.DEFAULT_NOISE_DENSITY
        self.stroke_step_distance = 2.0  # Distance between resampled points
        
        # Animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.steps_per_frame = Config.DEFAULT_STEPS_PER_FRAME
        
        # OpenGL texture
        self.texture_id = None
        
    def initializeGL(self):
        """Initialize OpenGL context."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Create texture for field display
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
    def resizeGL(self, width: int, height: int):
        """Handle widget resize.
        
        Args:
            width: New widget width
            height: New widget height
        """
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
        # Resize field to match widget dimensions (with scale factor for performance)
        scale_factor = 2  # Adjust this to balance performance vs resolution
        field_width = max(width // scale_factor, Config.MIN_FIELD_SIZE)
        field_height = max(height // scale_factor, Config.MIN_FIELD_SIZE)
        
        # Only resize if dimensions changed significantly
        if (abs(field_width - self.engine.width) > 10 or 
            abs(field_height - self.engine.height) > 10):
            self.engine.resize(field_width, field_height)
            # Reset pan to center the field
            self.pan_x = 0
            self.pan_y = 0
            self.zoom = min(width / field_width, height / field_height)
            # Emit signal for UI updates
            self.field_resized.emit(field_width, field_height)
        
    def paintGL(self):
        """Render the Game of Life field."""
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Get RGBA data from engine
        rgba_gpu = self.engine.get_rgba_array()
        rgba_cpu = cp.asnumpy(rgba_gpu)
        
        # Update texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                    self.engine.width, self.engine.height, 
                    0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_cpu)
        
        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glPushMatrix()
        
        # Apply transformations
        glTranslatef(self.pan_x, self.pan_y, 0)
        glScalef(self.zoom, self.zoom, 1)
        
        # Draw quad with texture
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(self.engine.width, 0)
        glTexCoord2f(1, 1)
        glVertex2f(self.engine.width, self.engine.height)
        glTexCoord2f(0, 1)
        glVertex2f(0, self.engine.height)
        glEnd()
        
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)
        
        # Draw stroke overlay if currently drawing
        if (self.drawing or self.erasing) and len(self.stroke_points) > 1:
            self.draw_stroke_overlay()
        
    def update_simulation(self):
        """Update simulation and trigger redraw."""
        if self.engine.is_running:
            self.engine.step(self.steps_per_frame)
            self.generation_updated.emit(self.engine.generation)
            self.update()
    
    def start_simulation(self):
        """Start the simulation."""
        self.engine.is_running = True
        self.timer.start(1000 // Config.DEFAULT_FPS)
    
    def stop_simulation(self):
        """Stop the simulation."""
        self.engine.is_running = False
        self.timer.stop()
    
    def reset_simulation(self):
        """Reset the simulation."""
        self.stop_simulation()
        self.engine.reset()
        self.generation_updated.emit(0)
        self.update()
    
    def set_simulation_speed(self, steps_per_frame: int):
        """Set simulation speed.
        
        Args:
            steps_per_frame: Number of steps per frame
        """
        self.steps_per_frame = min(max(1, steps_per_frame), 
                                  Config.MAX_STEPS_PER_FRAME)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events.
        
        Args:
            event: Mouse event
        """
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.stroke_points = []
            # Determine draw value based on current rule type
            if self.engine.rule_type == 2:  # Multichannel
                import random
                self.current_draw_value = (random.randint(150, 255),  # Mental health
                                         random.randint(150, 255),   # Body health  
                                         random.randint(150, 255),   # Social health
                                         random.randint(100, 200))   # Money
            else:  # Binary or multistate
                self.current_draw_value = (255, 0, 0, 0)  # Standard alive cell
            self.add_stroke_point(event.pos())
        elif event.button() == Qt.RightButton:
            self.erasing = True
            self.stroke_points = []
            self.current_draw_value = (0, 0, 0, 0)  # Erase value
            self.add_stroke_point(event.pos())
        elif event.button() == Qt.MiddleButton:
            self.last_draw_pos = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events.
        
        Args:
            event: Mouse event
        """
        if self.drawing or self.erasing:
            self.add_stroke_point(event.pos())
            self.update()  # Update to show stroke overlay
        elif event.buttons() & Qt.MiddleButton and self.last_draw_pos:
            # Pan view
            delta = event.pos() - self.last_draw_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_draw_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events.
        
        Args:
            event: Mouse event
        """
        if self.drawing or self.erasing:
            # Draw the complete stroke at once
            self.draw_stroke()
            
        self.drawing = False
        self.erasing = False
        self.last_draw_pos = None
        self.stroke_points = []
        self.current_draw_value = None
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming.
        
        Args:
            event: Wheel event
        """
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10.0, self.zoom))
        self.update()
    
    def add_stroke_point(self, pos: QPoint):
        """Add a point to the current stroke.
        
        Args:
            pos: Mouse position in widget coordinates
        """
        # Convert widget coordinates to field coordinates
        x = (pos.x() - self.pan_x) / self.zoom
        y = (pos.y() - self.pan_y) / self.zoom
        
        # Only add if within field bounds
        if 0 <= x < self.engine.width and 0 <= y < self.engine.height:
            self.stroke_points.append((x, y))
    
    def resample_stroke(self, points, step_distance):
        """Resample stroke points with consistent distance between points.
        
        Args:
            points: List of (x, y) tuples
            step_distance: Distance between resampled points
            
        Returns:
            List of resampled (x, y) tuples
        """
        if len(points) < 2:
            return points
            
        resampled = [points[0]]  # Start with first point
        current_distance = 0
        
        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            
            segment_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            if segment_length == 0:
                continue
                
            # How many steps fit in this segment?
            remaining_distance = step_distance - current_distance
            
            while remaining_distance <= segment_length:
                # Calculate point at remaining_distance along this segment
                t = remaining_distance / segment_length
                new_x = x1 + (x2 - x1) * t
                new_y = y1 + (y2 - y1) * t
                resampled.append((new_x, new_y))
                
                # Move to next step
                remaining_distance += step_distance
                
            # Update current distance for next iteration
            current_distance = segment_length - (remaining_distance - step_distance)
            
        return resampled
    
    def draw_stroke(self):
        """Draw the complete stroke with resampling."""
        if not self.stroke_points or self.current_draw_value is None:
            return
            
        # If only one point, just draw a circle
        if len(self.stroke_points) == 1:
            x, y = self.stroke_points[0]
            self.engine.draw_circle(int(x), int(y), self.brush_radius, self.current_draw_value)
        else:
            # Resample the stroke for consistent spacing
            resampled_points = self.resample_stroke(self.stroke_points, self.stroke_step_distance)
            
            # Draw circles at each resampled point
            for x, y in resampled_points:
                int_x, int_y = int(x), int(y)
                if 0 <= int_x < self.engine.width and 0 <= int_y < self.engine.height:
                    self.engine.draw_circle(int_x, int_y, self.brush_radius, self.current_draw_value)
        
        # Update display after drawing complete stroke
        self.update()
    
    def draw_stroke_overlay(self):
        """Draw temporary stroke overlay to show current drawing path."""
        if len(self.stroke_points) < 2:
            return
            
        # Set up overlay drawing
        glPushMatrix()
        glTranslatef(self.pan_x, self.pan_y, 0)
        glScalef(self.zoom, self.zoom, 1)
        
        # Set stroke color based on drawing/erasing
        if self.erasing:
            glColor4f(1.0, 0.0, 0.0, 0.7)  # Red for erasing
        else:
            glColor4f(0.0, 1.0, 0.0, 0.7)  # Green for drawing
            
        # Draw stroke line
        glLineWidth(max(1.0, self.brush_radius * self.zoom / 4))
        glBegin(GL_LINE_STRIP)
        for x, y in self.stroke_points:
            glVertex2f(x, y)
        glEnd()
        
        # Draw brush circles at key points for better visualization
        glPointSize(max(2.0, self.brush_radius * self.zoom / 2))
        glBegin(GL_POINTS)
        for i, (x, y) in enumerate(self.stroke_points):
            if i % 3 == 0:  # Every 3rd point to avoid clutter
                glVertex2f(x, y)
        glEnd()
        
        glPopMatrix()
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color
    
    
    def add_noise(self):
        """Add noise pattern to the field."""
        self.engine.add_noise(self.noise_density)
        self.update()
    
    def set_brush_radius(self, radius: int):
        """Set brush radius.
        
        Args:
            radius: New brush radius
        """
        self.brush_radius = min(max(Config.MIN_BRUSH_RADIUS, radius), 
                               Config.MAX_BRUSH_RADIUS)
    
    def set_noise_density(self, density: float):
        """Set noise density.
        
        Args:
            density: Noise density (0.0 to 1.0)
        """
        self.noise_density = min(max(0.0, density), 1.0)
    
    def set_stroke_step_distance(self, distance: float):
        """Set the distance between resampled stroke points.
        
        Args:
            distance: Distance between points in field coordinates
        """
        self.stroke_step_distance = max(0.5, min(10.0, distance))