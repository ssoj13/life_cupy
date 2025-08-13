"""OpenGL widget for rendering Game of Life field."""
import numpy as np
import cupy as cp
from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QMouseEvent, QWheelEvent
from OpenGL.GL import *
from typing import Optional

from ..core.life_engine import GameOfLifeEngine
from ..utils.config import Config


class LifeGLWidget(QOpenGLWidget):
    """OpenGL widget for rendering and interacting with Game of Life field."""
    
    generation_updated = Signal(int)
    field_resized = Signal(int, int)  # width, height
    
    def __init__(self, parent=None):
        """Initialize the OpenGL widget."""
        super().__init__(parent)
        
        # Engine
        self.engine = GameOfLifeEngine()
        
        # Display settings
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Interaction state
        self.drawing = False
        self.erasing = False
        self.last_draw_pos: Optional[QPoint] = None
        self.brush_radius = Config.DEFAULT_BRUSH_RADIUS
        self.noise_density = Config.DEFAULT_NOISE_DENSITY
        
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
            self.draw_at_position(event.pos())
        elif event.button() == Qt.RightButton:
            self.erasing = True
            self.erase_at_position(event.pos())
        elif event.button() == Qt.MiddleButton:
            self.last_draw_pos = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events.
        
        Args:
            event: Mouse event
        """
        if self.drawing:
            self.draw_at_position(event.pos())
        elif self.erasing:
            self.erase_at_position(event.pos())
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
        self.drawing = False
        self.erasing = False
        self.last_draw_pos = None
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming.
        
        Args:
            event: Wheel event
        """
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10.0, self.zoom))
        self.update()
    
    def draw_at_position(self, pos: QPoint):
        """Draw at mouse position.
        
        Args:
            pos: Mouse position in widget coordinates
        """
        # Convert widget coordinates to field coordinates
        x = int((pos.x() - self.pan_x) / self.zoom)
        y = int((pos.y() - self.pan_y) / self.zoom)
        
        if 0 <= x < self.engine.width and 0 <= y < self.engine.height:
            self.engine.draw_circle(x, y, self.brush_radius, True)
            self.update()
    
    def erase_at_position(self, pos: QPoint):
        """Erase at mouse position.
        
        Args:
            pos: Mouse position in widget coordinates
        """
        # Convert widget coordinates to field coordinates
        x = int((pos.x() - self.pan_x) / self.zoom)
        y = int((pos.y() - self.pan_y) / self.zoom)
        
        if 0 <= x < self.engine.width and 0 <= y < self.engine.height:
            self.engine.draw_circle(x, y, self.brush_radius, False)
            self.update()
    
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