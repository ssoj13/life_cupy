"""Main application window for Game of Life."""
from PySide6.QtWidgets import (QMainWindow, QToolBar, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QSlider,
                              QSpinBox, QGroupBox, QDockWidget, QStatusBar, QComboBox,
                              QColorDialog, QFrame, QCheckBox, QMenuBar, QFileDialog,
                              QMessageBox)
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence, QColor, QPixmap, QImage
from pathlib import Path
import numpy as np
import logging

# Set up global logger
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
LOG.addHandler(handler)

from .gl_widget import LifeGLWidget
from ..utils.config import Config
from ..core import RuleType, BinaryRule, MultiStateRule, MultiChannelRule, ExtendedClassicRule


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Initialize settings first
        self.settings = QSettings('LifeGame', 'CudaLife')
        
        self.setWindowTitle(Config.WINDOW_TITLE)
        
        # Restore window geometry
        geometry = self.settings.value('window_geometry')
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(100, 100, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        
        # State - initialize first before creating UI
        self.is_playing = False
        
        # Restore paint color and brush size
        saved_color = self.settings.value('paint_color', [255, 255, 255])
        self.current_paint_color = QColor(int(saved_color[0]), int(saved_color[1]), int(saved_color[2]))
        self.saved_brush_size = self.settings.value('brush_size', Config.DEFAULT_BRUSH_RADIUS)
        
        # Create central widget
        self.gl_widget = LifeGLWidget()
        self.setCentralWidget(self.gl_widget)
        
        # Connect signals
        self.gl_widget.generation_updated.connect(self.update_generation_display)
        self.gl_widget.field_resized.connect(self.update_field_size_display)
        
        # Initialize recent files before creating menu
        self.recent_files = self.settings.value('recent_files', [])
        self.max_recent_files = 20
        self.last_folder = self.settings.value('last_folder', '')
        
        # Store field for reset functionality
        self.stored_field = None
        
        # Create UI elements
        self.create_menu_bar()
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
        
        # Remove generation counter from toolbar - moved to status bar
        
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
        
        glider_action = QAction("🚀 Add Glider", self)
        glider_action.triggered.connect(self.add_glider)
        toolbar.addAction(glider_action)
        
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
        
        # Stored field restore button
        stored_action = QAction("📁 Stored", self)
        stored_action.setToolTip("Restore stored field")
        stored_action.triggered.connect(self.restore_stored_field)
        toolbar.addAction(stored_action)
        
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
        self.radius_slider.setValue(self.saved_brush_size)
        self.radius_slider.valueChanged.connect(self.update_brush_radius)
        radius_layout.addWidget(self.radius_slider)
        self.radius_label = QLabel(str(self.saved_brush_size))
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
        inst_layout.addWidget(QLabel("• Space/↑: Play/Pause"))
        inst_layout.addWidget(QLabel("• R/←: Reset"))
        inst_layout.addWidget(QLabel("• S/→: Single Step"))
        inst_layout.addWidget(QLabel("• C/Ctrl+R: Clear Field"))
        inst_layout.addWidget(QLabel("• H/Home: Reset Viewport"))
        inst_layout.addWidget(QLabel("• [/]: Brush Size ±10%"))
        inst_layout.addWidget(QLabel("• ESC: Exit Application"))
        inst_layout.addWidget(QLabel("• GPU anti-aliased drawing"))
        inst_layout.addWidget(QLabel("• Color picker works in all modes"))
        inst_layout.addWidget(QLabel("• RGBA unified system"))
        inst_layout.addWidget(QLabel("• Classic: 4 independent RGBA layers!"))
        inst_layout.addWidget(QLabel("• RGB Conway: 3 independent RGB layers"))
        inst_layout.addWidget(QLabel("• Life Sim: R=Mental G=Body B=Social"))
        instructions.setLayout(inst_layout)
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def create_menu_bar(self):
        """Create menu bar with File and Video options."""
        LOG.debug("Creating menu bar")
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Open image
        LOG.debug("Creating open action")
        open_action = QAction('&Open Image...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        LOG.debug("Open action added to menu")
        
        # Save image
        save_action = QAction('&Save Image...', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu('&Recent Files')
        self.update_recent_files_menu()
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Video menu
        video_menu = menubar.addMenu('&Video')
        
        # Start/Stop recording
        self.record_action = QAction('Start &Recording', self)
        self.record_action.triggered.connect(self.toggle_recording)
        video_menu.addAction(self.record_action)
        
        # Recording settings
        settings_action = QAction('Recording &Settings...', self)
        settings_action.triggered.connect(self.show_recording_settings)
        video_menu.addAction(settings_action)
        
        # Initialize recording state
        self.is_recording = False
        self.recording_fps = 30
        self.recording_quality = 'high'
        
        # Recording settings
        pass  # Recent files already initialized above
    
    def create_status_bar(self):
        """Create status bar with generation counter, field size, and FPS."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Generation counter (moved from toolbar)
        self.generation_label = QLabel("Generation: 0")
        self.status_bar.addWidget(self.generation_label)
        
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
        
        # Arrow Up for play
        up_play_action = QAction(self)
        up_play_action.setShortcut(QKeySequence(Qt.Key_Up))
        up_play_action.triggered.connect(self.toggle_simulation)
        self.addAction(up_play_action)
        
        # Arrow Left for reset
        left_reset_action = QAction(self)
        left_reset_action.setShortcut(QKeySequence(Qt.Key_Left))
        left_reset_action.triggered.connect(self.reset_simulation)
        self.addAction(left_reset_action)
        
        # Arrow Right for step forward
        right_step_action = QAction(self)
        right_step_action.setShortcut(QKeySequence(Qt.Key_Right))
        right_step_action.triggered.connect(self.step_simulation)
        self.addAction(right_step_action)
        
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
        
        # Ctrl+R for clear
        ctrl_clear_action = QAction(self)
        ctrl_clear_action.setShortcut(QKeySequence("Ctrl+R"))
        ctrl_clear_action.triggered.connect(self.clear_field)
        self.addAction(ctrl_clear_action)
        
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
            # Store field when starting simulation from generation 0
            if self.gl_widget.engine.generation == 0:
                self.stored_field = self.gl_widget.engine.get_field_cpu().copy()
                self.status_bar.showMessage("Field stored for reset", 1500)
            
            self.gl_widget.start_simulation()
            self.play_pause_action.setText("⏸ Pause")
            self.is_playing = True
            
    def step_simulation(self):
        """Perform single simulation step."""
        # Debug: Check field before step
        field_before = self.gl_widget.engine.get_field_cpu()
        alive_before = np.count_nonzero(field_before > 64)
        
        self.gl_widget.engine.step()
        
        # Debug: Check field after step
        field_after = self.gl_widget.engine.get_field_cpu()
        alive_after = np.count_nonzero(field_after > 64)
        
        LOG.info(f"STEP {self.gl_widget.engine.generation}: Alive cells before: {alive_before}, after: {alive_after}")
        
        # Debug: Check actual values in the field
        sample_values = field_after[field_after.shape[0]//2:field_after.shape[0]//2+5, 
                                   field_after.shape[1]//2:field_after.shape[1]//2+5]
        LOG.debug(f"Sample values after step: min={sample_values.min()}, max={sample_values.max()}")
        LOG.debug(f"Sample area (red channel):\n{sample_values[:,:,0]}")
        
        self.gl_widget.update()
        self.update_generation_display(self.gl_widget.engine.generation)
        
    def reset_simulation(self):
        """Reset the simulation."""
        self.gl_widget.reset_simulation()
        self.play_pause_action.setText("▶ Play")
        self.is_playing = False
        
        # Restore stored field if available
        if self.stored_field is not None:
            self.gl_widget.engine.set_field(self.stored_field)
            self.gl_widget.update()
            self.status_bar.showMessage("Restored stored field", 1500)
        
    def clear_field(self):
        """Clear the field."""
        self.gl_widget.engine.reset()
        self.gl_widget.update()
        self.update_generation_display(0)
        
    def add_noise(self):
        """Add noise to the field."""
        self.gl_widget.add_noise()
    
    def add_glider(self):
        """Add a glider pattern to the field for testing Conway's Life."""
        # Get current field
        field = self.gl_widget.engine.get_field_cpu()
        height, width = field.shape[:2]
        
        # Add glider pattern at center (classic glider pattern)
        center_x, center_y = width // 2, height // 2
        
        # Glider pattern (relative positions)
        glider_pattern = [
            (1, 0), (2, 1), (0, 2), (1, 2), (2, 2)
        ]
        
        # Apply glider pattern in current paint color
        r, g, b = self.current_paint_color.red(), self.current_paint_color.green(), self.current_paint_color.blue()
        
        for dx, dy in glider_pattern:
            x, y = center_x + dx, center_y + dy
            if 0 <= x < width and 0 <= y < height:
                field[y, x] = [r, g, b, 255]
        
        # Set the field back
        self.gl_widget.engine.set_field(field)
        self.gl_widget.update()
        self.status_bar.showMessage("Added glider pattern", 2000)
        
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
        self.rule_combobox.addItem("Conway's Life (B3/S2345678)", (RuleType.BINARY_BS, BinaryRule.CONWAY_LIFE))
        self.rule_combobox.addItem("HighLife (B36/S23)", (RuleType.BINARY_BS, BinaryRule.HIGHLIFE))
        self.rule_combobox.addItem("Seeds (B2/S)", (RuleType.BINARY_BS, BinaryRule.SEEDS))
        self.rule_combobox.addItem("Day & Night (B3678/S34678)", (RuleType.BINARY_BS, BinaryRule.DAY_NIGHT))
        self.rule_combobox.addItem("Maze (B3/S12345)", (RuleType.BINARY_BS, BinaryRule.MAZE))
        self.rule_combobox.addItem("Replicator (B1357/S1357)", (RuleType.BINARY_BS, BinaryRule.REPLICATOR))
        self.rule_combobox.addItem("DryLife (B37/S23)", (RuleType.BINARY_BS, BinaryRule.DRYLIFE))
        self.rule_combobox.addItem("Live Free or Die (B2/S0)", (RuleType.BINARY_BS, BinaryRule.LIVE_FREE_DIE))
        self.rule_combobox.addItem("2x2 (B36/S125)", (RuleType.BINARY_BS, BinaryRule.RULE_2X2))
        self.rule_combobox.addItem("Life Without Death (B3/S012345678)", (RuleType.BINARY_BS, BinaryRule.LIFE_WITHOUT_DEATH))
        self.rule_combobox.addItem("Gradual Conway (B3/S2345678 Smooth)", (RuleType.BINARY_BS, BinaryRule.GRADUAL_CONWAY))
        self.rule_combobox.addItem("Classic Conway (B3/S23)", (RuleType.BINARY_BS, BinaryRule.CLASSIC_CONWAY))
        
        # Add multistate rules
        self.rule_combobox.addItem("Brian's Brain", (RuleType.MULTISTATE, MultiStateRule.BRIANS_BRAIN))
        
        # Add multichannel rules  
        self.rule_combobox.addItem("Life Simulation (Health/Money)", (RuleType.MULTICHANNEL, MultiChannelRule.LIFE_SIMULATION))
        
        # Add extended classic rules
        self.rule_combobox.addItem("RGB Conway (3-Layer Life)", (RuleType.EXTENDED_CLASSIC, ExtendedClassicRule.RGB_CONWAY))
        
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
    
    def restore_stored_field(self):
        """Restore the stored field manually."""
        if self.stored_field is not None:
            self.gl_widget.engine.set_field(self.stored_field)
            self.gl_widget.engine.generation = 0  # Reset generation counter
            self.gl_widget.update()
            self.update_generation_display(0)
            self.status_bar.showMessage("Restored stored field", 1500)
        else:
            self.status_bar.showMessage("No field stored yet", 1500)
        
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
    
    def _load_image_file(self, file_path: str) -> bool:
        """Load an image file into the field. Returns True if successful."""
        try:
            LOG.debug("Loading image file")
            # Load image with Qt
            qimg = QImage(file_path)
            
            if qimg.isNull():
                raise ValueError("Failed to load image file")
            
            LOG.info(f"QImage loaded successfully: {not qimg.isNull()}")
            
            # Convert to RGBA if needed
            if qimg.format() != QImage.Format_RGBA8888:
                qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
            
            # Get current field dimensions
            field_width = self.gl_widget.engine.width
            field_height = self.gl_widget.engine.height
            
            # Scale image to fit field
            qimg = qimg.scaled(field_width, field_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            
            # Convert QImage to numpy array
            # QImage data is in RGBA format
            ptr = qimg.constBits()
            img_array = np.array(ptr).reshape(field_height, field_width, 4)  # Shape: (height, width, 4)
            
            # Debug: Check what values we're getting
            LOG.info(f"BEFORE SET_FIELD - Image array shape: {img_array.shape}")
            LOG.info(f"BEFORE SET_FIELD - Image array dtype: {img_array.dtype}")
            LOG.info(f"BEFORE SET_FIELD - Min/Max values: {img_array.min()}/{img_array.max()}")
            LOG.info(f"BEFORE SET_FIELD - Non-zero count: {np.count_nonzero(img_array)}")
            LOG.info(f"BEFORE SET_FIELD - Sample values at center: {img_array[field_height//2, field_width//2]}")
            
            self.gl_widget.engine.set_field(img_array)
            
            # Debug: Check what the engine actually stored
            stored_field = self.gl_widget.engine.get_field_cpu()
            LOG.info(f"AFTER SET_FIELD - shape: {stored_field.shape}")
            LOG.info(f"AFTER SET_FIELD - dtype: {stored_field.dtype}")
            LOG.info(f"AFTER SET_FIELD - min/max: {stored_field.min()}/{stored_field.max()}")
            LOG.info(f"AFTER SET_FIELD - sample: {stored_field[field_height//2, field_width//2]}")
            
            # Now test one simulation step to see what happens
            LOG.info("TESTING ONE STEP...")
            self.gl_widget.engine.step()
            after_step = self.gl_widget.engine.get_field_cpu()
            LOG.info(f"AFTER ONE STEP - min/max: {after_step.min()}/{after_step.max()}")
            LOG.info(f"AFTER ONE STEP - non-zero count: {np.count_nonzero(after_step)}")
            LOG.info(f"AFTER ONE STEP - sample: {after_step[field_height//2, field_width//2]}")
            
            # Reset to before step
            self.gl_widget.engine.set_field(img_array)
            self.gl_widget.engine.generation = 0
            self.gl_widget.update()
            
            # Store the loaded image as the reset field
            self.stored_field = img_array.copy()
            self.gl_widget.engine.generation = 0  # Reset generation counter
            self.update_generation_display(0)
            
            return True
            
        except Exception as e:
            LOG.error(f"Failed to load image {Path(file_path).name}: {str(e)}")
            QMessageBox.critical(self, "Error Loading Image", f"Failed to load image:\\n{str(e)}")
            return False
    
    def open_image(self):
        """Open PNG/JPEG image and load into field."""
        LOG.info("Open image dialog called")
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            self.last_folder, 
            "Image Files (*.png *.jpg *.jpeg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        )
        
        LOG.info(f"Selected file: {file_path}")
        if file_path:
            if self._load_image_file(file_path):
                # Add to recent files and save last folder
                self.add_recent_file(file_path)
                self.last_folder = str(Path(file_path).parent)
                self.settings.setValue('last_folder', self.last_folder)
                self.status_bar.showMessage(f"Loaded image: {Path(file_path).name} - Stored for reset", 3000)
    
    def save_image(self):
        """Save current field as PNG/JPEG image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            self.last_folder,
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if file_path:
            try:
                # Get current field as RGBA array
                field_array = self.gl_widget.engine.get_rgba_array()
                
                # Convert CuPy array to NumPy if needed
                if hasattr(field_array, 'get'):
                    field_array = field_array.get()
                
                height, width, channels = field_array.shape
                
                # Create QImage from numpy array
                qimg = QImage(field_array.data, width, height, QImage.Format_RGBA8888)
                
                # Convert to RGB for JPEG
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    qimg = qimg.convertToFormat(QImage.Format_RGB888)
                
                # Save image
                if not qimg.save(file_path):
                    raise ValueError("Failed to save image")
                
                self.status_bar.showMessage(f"Saved image: {Path(file_path).name}", 3000)
                
                # Save last folder
                self.last_folder = str(Path(file_path).parent)
                self.settings.setValue('last_folder', self.last_folder)
                
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Image", f"Failed to save image:\\n{str(e)}")
    
    def toggle_recording(self):
        """Toggle video recording on/off."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_action.setText('Stop &Recording')
            self.status_bar.showMessage("Video recording started", 2000)
            # TODO: Implement actual video recording
        else:
            # Stop recording
            self.is_recording = False
            self.record_action.setText('Start &Recording')
            self.status_bar.showMessage("Video recording stopped", 2000)
            # TODO: Implement actual video recording
    
    def show_recording_settings(self):
        """Show recording settings dialog."""
        # TODO: Implement recording settings dialog
        QMessageBox.information(
            self, 
            "Recording Settings", 
            f"Current Settings:\\n"
            f"FPS: {self.recording_fps}\\n"
            f"Quality: {self.recording_quality}\\n\\n"
            f"Settings dialog not yet implemented.\\n"
            f"Options will include:\\n"
            f"• Frame rate (15, 30, 60 FPS)\\n"
            f"• Quality (low, medium, high)\\n"
            f"• Output format (MP4, AVI, GIF)\\n"
            f"• Recording region (full field, viewport)"
        )
    
    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        # Remove if already in list
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        
        # Add to beginning
        self.recent_files.insert(0, file_path)
        
        # Limit to max files
        self.recent_files = self.recent_files[:self.max_recent_files]
        
        # Save to settings
        self.settings.setValue('recent_files', self.recent_files)
        
        # Update menu
        self.update_recent_files_menu()
    
    def update_recent_files_menu(self):
        """Update the recent files menu."""
        self.recent_menu.clear()
        
        if not self.recent_files:
            no_recent = QAction('No recent files', self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return
        
        for file_path in self.recent_files:
            if Path(file_path).exists():
                action = QAction(Path(file_path).name, self)
                action.setToolTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self.load_recent_file(path))
                self.recent_menu.addAction(action)
        
        if self.recent_files:
            self.recent_menu.addSeparator()
            clear_action = QAction('Clear Recent Files', self)
            clear_action.triggered.connect(self.clear_recent_files)
            self.recent_menu.addAction(clear_action)
    
    def load_recent_file(self, file_path: str):
        """Load a recent file."""
        LOG.info(f"Loading recent file: {file_path}")
        if Path(file_path).exists():
            if self._load_image_file(file_path):
                self.status_bar.showMessage(f"Loaded recent: {Path(file_path).name} - Stored for reset", 3000)
                # Move to top of recent list
                self.add_recent_file(file_path)
            else:
                # Remove invalid file from recent list
                if file_path in self.recent_files:
                    self.recent_files.remove(file_path)
                    self.settings.setValue('recent_files', self.recent_files)
                    self.update_recent_files_menu()
        else:
            QMessageBox.warning(self, "File Not Found", f"File no longer exists:\\n{file_path}")
            # Remove missing file from recent list
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self.settings.setValue('recent_files', self.recent_files)
                self.update_recent_files_menu()
    
    def clear_recent_files(self):
        """Clear the recent files list."""
        self.recent_files.clear()
        self.settings.setValue('recent_files', self.recent_files)
        self.update_recent_files_menu()
    
    def closeEvent(self, event):
        """Save settings when closing."""
        # Save window geometry
        self.settings.setValue('window_geometry', self.saveGeometry())
        
        # Save paint color
        color = self.current_paint_color
        self.settings.setValue('paint_color', [color.red(), color.green(), color.blue()])
        
        # Save brush size
        self.settings.setValue('brush_size', self.radius_slider.value())
        
        # Save last folder
        self.settings.setValue('last_folder', self.last_folder)
        
        super().closeEvent(event)