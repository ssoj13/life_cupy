"""Custom button widget with separate left and right click signals."""
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtCore import Qt


class Button2(QPushButton):
    """Button that emits separate signals for left and right clicks."""
    
    left_clicked = Signal()
    right_clicked = Signal()
    
    def __init__(self, text: str = "", parent=None):
        """Initialize Button2.
        
        Args:
            text: Button text
            parent: Parent widget
        """
        super().__init__(text, parent)
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events.
        
        Args:
            event: Mouse press event
        """
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit()
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit()
        
        # Call parent implementation for visual feedback
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events.
        
        Args:
            event: Mouse release event
        """
        # Only process the standard clicked signal for left button
        if event.button() == Qt.LeftButton:
            super().mouseReleaseEvent(event)
        else:
            # For right button, just update visual state without emitting clicked()
            self.setDown(False)
            self.repaint()