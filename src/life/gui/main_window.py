"""Main application window for Game of Life."""
from PySide6.QtWidgets import (QMainWindow, QToolBar, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QSlider,
                              QSpinBox, QGroupBox, QDockWidget, QStatusBar, QComboBox,
                              QColorDialog, QFrame, QCheckBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QColor

from .gl_widget import LifeGLWidget
from ..utils.config import Config
from ..core import RuleType, BinaryRule, MultiStateRule, MultiChannelRule


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.setGeometry(100, 100, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        
        # State - initialize first before creating UI
        self.is_playing = False
        self.current_paint_color = QColor(255, 255, 255)  # Default white
        
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
        
        # Rule selection combobox
        toolbar.addWidget(QLabel("Rule:"))
        self.rule_combobox = QComboBox()
        self.setup_rule_combobox()
        self.rule_combobox.currentTextChanged.connect(self.change_rule)
        toolbar.addWidget(self.rule_combobox)
        
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
        
        toolbar.addSeparator()
        
        # Color picker for painting
        toolbar.addWidget(QLabel("Color:"))
        self.color_button = QPushButton()
        self.color_button.setMaximumSize(30, 25)
        self.color_button.clicked.connect(self.open_color_picker)
        self.update_color_button()
        toolbar.addWidget(self.color_button)
        
        toolbar.addSeparator()
        
        # Clipboard buttons - grouped save/load pairs
        for i in range(4):
            # Add a small separator frame between groups
            if i > 0:
                separator = QFrame()
                separator.setFrameStyle(QFrame.VLine | QFrame.Sunken)
                separator.setMaximumWidth(2)
                toolbar.addWidget(separator)
            
            # Save button
            save_action = QAction(f"S{i+1}", self)
            save_action.setToolTip(f"Save to slot {i+1}")
            save_action.triggered.connect(lambda checked, slot=i: self.save_to_clipboard(slot))
            toolbar.addAction(save_action)
            
            # Load button (right next to save)
            load_action = QAction(f"L{i+1}", self)
            load_action.setToolTip(f"Load from slot {i+1}")
            load_action.triggered.connect(lambda checked, slot=i: self.load_from_clipboard(slot))
            toolbar.addAction(load_action)
        
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
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(1, 100)  # 1 to 100 pixels
        self.radius_slider.setValue(Config.DEFAULT_BRUSH_RADIUS)
        self.radius_slider.valueChanged.connect(self.update_brush_radius)
        radius_layout.addWidget(self.radius_slider)
        self.radius_label = QLabel(str(Config.DEFAULT_BRUSH_RADIUS))
        self.radius_label.setMinimumWidth(30)
        radius_layout.addWidget(self.radius_label)
        brush_layout.addLayout(radius_layout)
        
# Removed drawing mode checkbox - now always immediate mode
        
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
        
        # Display mode settings
        display_group = QGroupBox("Display Mode")
        display_layout = QVBoxLayout()
        
        # Display mode selection
        self.display_mode_combobox = QComboBox()
        self.display_mode_combobox.addItem("RGB Display", 0)
        self.display_mode_combobox.addItem("Money as Brightness", 1)
        self.display_mode_combobox.addItem("Health Heatmap", 2)
        self.display_mode_combobox.addItem("Channel 1 Only", 3)
        self.display_mode_combobox.addItem("Channel 2 Only", 4)
        self.display_mode_combobox.addItem("Channel 3 Only", 5)
        self.display_mode_combobox.addItem("Channel 4 Only", 6)
        self.display_mode_combobox.currentIndexChanged.connect(self.change_display_mode)
        display_layout.addWidget(self.display_mode_combobox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Instructions
        instructions = QGroupBox("Controls")
        inst_layout = QVBoxLayout()
        inst_layout.addWidget(QLabel("• Left Click: Draw"))
        inst_layout.addWidget(QLabel("• Right Click: Erase"))
        inst_layout.addWidget(QLabel("• Middle Drag: Pan"))
        inst_layout.addWidget(QLabel("• Scroll: Zoom"))
        inst_layout.addWidget(QLabel("• Space: Play/Pause"))
        inst_layout.addWidget(QLabel("• H/Home: Reset Viewport"))
        inst_layout.addWidget(QLabel("• [/]: Brush Size ±10%"))
        inst_layout.addWidget(QLabel("• ESC: Exit Application"))
        inst_layout.addWidget(QLabel("• GPU anti-aliased drawing"))
        inst_layout.addWidget(QLabel("• Color picker works in multichannel mode"))
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
        
        # H for home (reset viewport)
        home_action = QAction(self)
        home_action.setShortcut(QKeySequence(Qt.Key_H))
        home_action.triggered.connect(self.reset_viewport)
        self.addAction(home_action)
        
        # Home key (right side of keyboard) for reset viewport
        home_key_action = QAction(self)
        home_key_action.setShortcut(QKeySequence(Qt.Key_Home))
        home_key_action.triggered.connect(self.reset_viewport)
        self.addAction(home_key_action)
        
        # [ key for smaller brush size
        smaller_brush_action = QAction(self)
        smaller_brush_action.setShortcut(QKeySequence(Qt.Key_BracketLeft))
        smaller_brush_action.triggered.connect(self.decrease_brush_size)
        self.addAction(smaller_brush_action)
        
        # ] key for larger brush size
        larger_brush_action = QAction(self)
        larger_brush_action.setShortcut(QKeySequence(Qt.Key_BracketRight))
        larger_brush_action.triggered.connect(self.increase_brush_size)
        self.addAction(larger_brush_action)
        
        # ESC key to exit application
        exit_action = QAction(self)
        exit_action.setShortcut(QKeySequence(Qt.Key_Escape))
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)
        
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
        self.radius_label.setText(str(value))
    
# Removed toggle method - now always immediate mode
        
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
        
    def setup_rule_combobox(self):
        """Setup the rule selection combobox with all available rules."""
        self.rule_combobox.clear()
        
        # Add binary rules
        self.rule_combobox.addItem("Conway's Life (B3/S23)", (RuleType.BINARY_BS, BinaryRule.CONWAY_LIFE))
        self.rule_combobox.addItem("HighLife (B36/S23)", (RuleType.BINARY_BS, BinaryRule.HIGHLIFE))
        self.rule_combobox.addItem("Seeds (B2/S)", (RuleType.BINARY_BS, BinaryRule.SEEDS))
        self.rule_combobox.addItem("Day & Night (B3678/S34678)", (RuleType.BINARY_BS, BinaryRule.DAY_NIGHT))
        self.rule_combobox.addItem("Maze (B3/S12345)", (RuleType.BINARY_BS, BinaryRule.MAZE))
        self.rule_combobox.addItem("Replicator (B1357/S1357)", (RuleType.BINARY_BS, BinaryRule.REPLICATOR))
        self.rule_combobox.addItem("DryLife (B37/S23)", (RuleType.BINARY_BS, BinaryRule.DRYLIFE))
        self.rule_combobox.addItem("Live Free or Die (B2/S0)", (RuleType.BINARY_BS, BinaryRule.LIVE_FREE_DIE))
        self.rule_combobox.addItem("2x2 (B36/S125)", (RuleType.BINARY_BS, BinaryRule.RULE_2X2))
        self.rule_combobox.addItem("Life Without Death (B3/S012345678)", (RuleType.BINARY_BS, BinaryRule.LIFE_WITHOUT_DEATH))
        
        # Add multistate rules
        self.rule_combobox.addItem("Brian's Brain", (RuleType.MULTISTATE, MultiStateRule.BRIANS_BRAIN))
        
        # Add multichannel rules  
        self.rule_combobox.addItem("Life Simulation (Health/Money)", (RuleType.MULTICHANNEL, MultiChannelRule.LIFE_SIMULATION))
        
        # Set default to Conway's Life
        self.rule_combobox.setCurrentIndex(0)
        
    def change_rule(self):
        """Change the cellular automata rule based on combobox selection."""
        rule_data = self.rule_combobox.currentData()
        if rule_data:
            rule_type, rule_id = rule_data
            self.gl_widget.engine.set_rule(rule_type, rule_id)
            self.gl_widget.update()
            
    def save_to_clipboard(self, slot: int):
        """Save current field state to clipboard slot.
        
        Args:
            slot: Clipboard slot number (0-3)
        """
        self.gl_widget.engine.save_to_clipboard(slot)
        self.status_bar.showMessage(f"Field saved to clipboard slot {slot + 1}", 2000)
        
    def load_from_clipboard(self, slot: int):
        """Load field state from clipboard slot.
        
        Args:
            slot: Clipboard slot number (0-3)
        """
        self.gl_widget.engine.load_from_clipboard(slot)
        self.gl_widget.update()
        self.status_bar.showMessage(f"Field loaded from clipboard slot {slot + 1}", 2000)
        
    def change_display_mode(self):
        """Change the display mode for multichannel visualization."""
        mode = self.display_mode_combobox.currentData()
        if mode is not None:
            self.gl_widget.engine.set_display_mode(mode)
            self.gl_widget.update()
    
    def reset_viewport(self):
        """Reset viewport to original zoom and pan state (Home key)."""
        self.gl_widget.reset_viewport()
    
    def decrease_brush_size(self):
        """Decrease brush size by 10% ([ key)."""
        current_size = self.radius_slider.value()
        new_size = max(1, int(current_size * 0.9))  # Decrease by 10%, minimum 1
        self.radius_slider.setValue(new_size)
        # update_brush_radius will be called automatically by slider valueChanged signal
    
    def increase_brush_size(self):
        """Increase brush size by 10% (] key)."""
        current_size = self.radius_slider.value()
        new_size = min(100, int(current_size * 1.1) + 1)  # Increase by 10%+1, maximum 100
        self.radius_slider.setValue(new_size)
        # update_brush_radius will be called automatically by slider valueChanged signal
    
    def open_color_picker(self):
        """Open color picker dialog for painting."""
        color = QColorDialog.getColor(self.current_paint_color, self, "Choose Paint Color")
        if color.isValid():
            self.current_paint_color = color
            self.update_color_button()
            # Update the GL widget's paint color
            self.gl_widget.set_paint_color(color)
    
    def update_color_button(self):
        """Update the color button to show current paint color."""
        r, g, b = self.current_paint_color.red(), self.current_paint_color.green(), self.current_paint_color.blue()
        self.color_button.setStyleSheet(f"QPushButton {{ background-color: rgb({r}, {g}, {b}); border: 1px solid black; }}")
        self.color_button.setToolTip(f"Paint Color: RGB({r}, {g}, {b})")