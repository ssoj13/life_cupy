"""Main application window for Game of Life."""
from PySide6.QtWidgets import (QMainWindow, QToolBar, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QSlider,
                              QSpinBox, QGroupBox, QDockWidget, QStatusBar)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

from .gl_widget import LifeGLWidget
from ..utils.config import Config


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.setGeometry(100, 100, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        
        # Create central widget
        self.gl_widget = LifeGLWidget()
        self.setCentralWidget(self.gl_widget)
        
        # Connect signals
        self.gl_widget.generation_updated.connect(self.update_generation_display)
        self.gl_widget.field_resized.connect(self.update_field_size_display)
        
        # Create UI elements
        self.create_toolbar()
        self.create_dock_widget()
        self.create_status_bar()
        self.setup_keyboard_shortcuts()
        
        # State
        self.is_playing = False
        
    def create_toolbar(self):
        """Create main toolbar with simulation controls."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Play/Pause button
        self.play_pause_action = QAction("▶ Play", self)
        self.play_pause_action.triggered.connect(self.toggle_simulation)
        toolbar.addAction(self.play_pause_action)
        
        # Step button
        step_action = QAction("⏭ Step", self)
        step_action.triggered.connect(self.step_simulation)
        toolbar.addAction(step_action)
        
        # Reset button
        reset_action = QAction("⏹ Reset", self)
        reset_action.triggered.connect(self.reset_simulation)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # Generation counter
        self.generation_label = QLabel("Generation: 0")
        self.generation_label.setStyleSheet("QLabel { padding: 0 10px; }")
        toolbar.addWidget(self.generation_label)
        
        toolbar.addSeparator()
        
        # Speed control
        toolbar.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, Config.MAX_STEPS_PER_FRAME)
        self.speed_slider.setValue(Config.DEFAULT_STEPS_PER_FRAME)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self.update_simulation_speed)
        toolbar.addWidget(self.speed_slider)
        
        self.speed_label = QLabel(f"{Config.DEFAULT_STEPS_PER_FRAME}")
        self.speed_label.setMinimumWidth(20)
        toolbar.addWidget(self.speed_label)
        
        toolbar.addSeparator()
        
        # Drawing tools
        noise_action = QAction("🎲 Add Noise", self)
        noise_action.triggered.connect(self.add_noise)
        toolbar.addAction(noise_action)
        
        clear_action = QAction("🗑 Clear", self)
        clear_action.triggered.connect(self.clear_field)
        toolbar.addAction(clear_action)
        
    def create_dock_widget(self):
        """Create dock widget with drawing tool controls."""
        dock = QDockWidget("Drawing Tools", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create widget for dock
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Brush settings group
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QVBoxLayout()
        
        # Brush radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setRange(Config.MIN_BRUSH_RADIUS, Config.MAX_BRUSH_RADIUS)
        self.radius_spinbox.setValue(Config.DEFAULT_BRUSH_RADIUS)
        self.radius_spinbox.valueChanged.connect(self.update_brush_radius)
        radius_layout.addWidget(self.radius_spinbox)
        brush_layout.addLayout(radius_layout)
        
        brush_group.setLayout(brush_layout)
        layout.addWidget(brush_group)
        
        # Noise settings group
        noise_group = QGroupBox("Noise Settings")
        noise_layout = QVBoxLayout()
        
        # Noise density
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(0, 100)
        self.density_slider.setValue(int(Config.DEFAULT_NOISE_DENSITY * 100))
        self.density_slider.valueChanged.connect(self.update_noise_density)
        density_layout.addWidget(self.density_slider)
        self.density_label = QLabel(f"{Config.DEFAULT_NOISE_DENSITY:.2f}")
        density_layout.addWidget(self.density_label)
        noise_layout.addLayout(density_layout)
        
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        # Field settings group
        field_group = QGroupBox("Field Settings")
        field_layout = QVBoxLayout()
        
        # Field size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Width:"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(Config.MIN_FIELD_SIZE, Config.MAX_FIELD_SIZE)
        self.width_spinbox.setValue(Config.DEFAULT_FIELD_WIDTH)
        size_layout.addWidget(self.width_spinbox)
        
        size_layout.addWidget(QLabel("Height:"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(Config.MIN_FIELD_SIZE, Config.MAX_FIELD_SIZE)
        self.height_spinbox.setValue(Config.DEFAULT_FIELD_HEIGHT)
        size_layout.addWidget(self.height_spinbox)
        
        field_layout.addLayout(size_layout)
        
        # Resize button
        resize_button = QPushButton("Resize Field")
        resize_button.clicked.connect(self.resize_field)
        field_layout.addWidget(resize_button)
        
        field_group.setLayout(field_layout)
        layout.addWidget(field_group)
        
        # Instructions
        instructions = QGroupBox("Controls")
        inst_layout = QVBoxLayout()
        inst_layout.addWidget(QLabel("• Left Click: Draw"))
        inst_layout.addWidget(QLabel("• Right Click: Erase"))
        inst_layout.addWidget(QLabel("• Middle Drag: Pan"))
        inst_layout.addWidget(QLabel("• Scroll: Zoom"))
        inst_layout.addWidget(QLabel("• Space: Play/Pause"))
        instructions.setLayout(inst_layout)
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Field size label
        self.field_size_label = QLabel(
            f"Field: {Config.DEFAULT_FIELD_WIDTH}×{Config.DEFAULT_FIELD_HEIGHT}"
        )
        self.status_bar.addPermanentWidget(self.field_size_label)
        
        # FPS label
        self.fps_label = QLabel("FPS: 0")
        self.status_bar.addPermanentWidget(self.fps_label)
        
        # FPS timer
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update every second
        self.frame_count = 0
        
    def setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Space for play/pause
        space_action = QAction(self)
        space_action.setShortcut(QKeySequence(Qt.Key_Space))
        space_action.triggered.connect(self.toggle_simulation)
        self.addAction(space_action)
        
        # R for reset
        reset_action = QAction(self)
        reset_action.setShortcut(QKeySequence(Qt.Key_R))
        reset_action.triggered.connect(self.reset_simulation)
        self.addAction(reset_action)
        
        # S for step
        step_action = QAction(self)
        step_action.setShortcut(QKeySequence(Qt.Key_S))
        step_action.triggered.connect(self.step_simulation)
        self.addAction(step_action)
        
        # N for noise
        noise_action = QAction(self)
        noise_action.setShortcut(QKeySequence(Qt.Key_N))
        noise_action.triggered.connect(self.add_noise)
        self.addAction(noise_action)
        
        # C for clear
        clear_action = QAction(self)
        clear_action.setShortcut(QKeySequence(Qt.Key_C))
        clear_action.triggered.connect(self.clear_field)
        self.addAction(clear_action)
        
    def toggle_simulation(self):
        """Toggle simulation play/pause."""
        if self.is_playing:
            self.gl_widget.stop_simulation()
            self.play_pause_action.setText("▶ Play")
            self.is_playing = False
        else:
            self.gl_widget.start_simulation()
            self.play_pause_action.setText("⏸ Pause")
            self.is_playing = True
            
    def step_simulation(self):
        """Perform single simulation step."""
        self.gl_widget.engine.step()
        self.gl_widget.update()
        self.update_generation_display(self.gl_widget.engine.generation)
        
    def reset_simulation(self):
        """Reset the simulation."""
        self.gl_widget.reset_simulation()
        self.play_pause_action.setText("▶ Play")
        self.is_playing = False
        
    def clear_field(self):
        """Clear the field."""
        self.gl_widget.engine.reset()
        self.gl_widget.update()
        self.update_generation_display(0)
        
    def add_noise(self):
        """Add noise to the field."""
        self.gl_widget.add_noise()
        
    def update_generation_display(self, generation: int):
        """Update generation counter display.
        
        Args:
            generation: Current generation number
        """
        self.generation_label.setText(f"Generation: {generation}")
        self.frame_count += 1
        
    def update_simulation_speed(self, value: int):
        """Update simulation speed.
        
        Args:
            value: Steps per frame
        """
        self.gl_widget.set_simulation_speed(value)
        self.speed_label.setText(str(value))
        
    def update_brush_radius(self, value: int):
        """Update brush radius.
        
        Args:
            value: New brush radius
        """
        self.gl_widget.set_brush_radius(value)
        
    def update_noise_density(self, value: int):
        """Update noise density.
        
        Args:
            value: Slider value (0-100)
        """
        density = value / 100.0
        self.gl_widget.set_noise_density(density)
        self.density_label.setText(f"{density:.2f}")
        
    def resize_field(self):
        """Resize the field based on spinbox values."""
        width = self.width_spinbox.value()
        height = self.height_spinbox.value()
        self.gl_widget.engine.resize(width, height)
        self.field_size_label.setText(f"Field: {width}×{height}")
        self.gl_widget.update()
        
    def update_fps(self):
        """Update FPS display."""
        self.fps_label.setText(f"FPS: {self.frame_count}")
        self.frame_count = 0
        
    def update_field_size_display(self, width: int, height: int):
        """Update field size display when field is resized.
        
        Args:
            width: New field width
            height: New field height
        """
        self.field_size_label.setText(f"Field: {width}×{height}")
        # Update spinboxes to reflect current field size
        self.width_spinbox.setValue(width)
        self.height_spinbox.setValue(height)