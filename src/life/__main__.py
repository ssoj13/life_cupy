"""Main entry point for Game of Life CUDA application."""
import sys
from PySide6.QtWidgets import QApplication
from .gui.main_window import MainWindow


def main():
    """Run the Game of Life application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()