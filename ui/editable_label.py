"""
Editable label widget with click-to-edit functionality.
"""

import json
import os
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class EditableLabel(QWidget):
    """A label that can be edited when edit mode is enabled."""
    
    textChanged = Signal(str)  # Emitted when text is changed
    
    def __init__(self, text, key, parent=None, bold=False, font_size=None):
        super().__init__(parent)
        self.key = key  # Unique key for saving/loading
        self.default_text = text
        self.bold = bold
        self.font_size = font_size
        self.is_editing = False
        
        # Load saved text if available
        self.text = self._load_text()
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Label to display text
        self.label = QLabel(self.text)
        if self.bold:
            self.label.setStyleSheet(f"font-weight: bold; font-size: {self.font_size or 12}px;")
        elif self.font_size:
            self.label.setStyleSheet(f"font-size: {self.font_size}px;")
        layout.addWidget(self.label)
        
        # Line edit (hidden by default)
        self.line_edit = QLineEdit(self.text)
        self.line_edit.hide()
        self.line_edit.returnPressed.connect(self._on_return_pressed)
        layout.addWidget(self.line_edit)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _on_return_pressed(self):
        """Handle return key press in edit mode."""
        # Just update the text, don't exit edit mode
        # The parent will handle exiting edit mode for all labels
        new_text = self.line_edit.text().strip()
        if new_text:
            self.text = new_text
    
    def enable_edit_mode(self):
        """Enable editing mode."""
        self.is_editing = True
        self.label.hide()
        self.line_edit.setText(self.text)
        self.line_edit.show()
    
    def disable_edit_mode(self, save=True):
        """Disable editing mode."""
        self.is_editing = False
        
        if save:
            new_text = self.line_edit.text().strip()
            if new_text:
                self.text = new_text
                self.label.setText(self.text)
                self._save_text()
                self.textChanged.emit(self.text)
        
        self.line_edit.hide()
        self.label.show()
    
    def _get_config_file(self):
        """Get path to config file."""
        return "ui_labels_config.json"
    
    def _load_text(self):
        """Load saved text from config file."""
        config_file = self._get_config_file()
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    return data.get(self.key, self.default_text)
            except:
                pass
        return self.default_text
    
    def _save_text(self):
        """Save text to config file."""
        config_file = self._get_config_file()
        data = {}
        
        # Load existing data
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
            except:
                pass
        
        # Update with new text
        data[self.key] = self.text
        
        # Save
        try:
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving label config: {e}")
    
    def get_text(self):
        """Get current text."""
        return self.text
    
    def set_text(self, text):
        """Set text programmatically."""
        self.text = text
        self.label.setText(text)
        self._save_text()
