"""
Dialog for saving and loading configurations.
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QListWidget, QMessageBox,
                               QGroupBox, QFileDialog)
from PySide6.QtCore import Qt

from core.configuration_manager import ConfigurationManager


class ConfigurationDialog(QDialog):
    """Dialog for managing saved configurations."""
    
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.config_manager = ConfigurationManager()
        
        self.setWindowTitle("Save/Load Custom Configuration")
        self.setMinimumSize(600, 500)
        
        self._init_ui()
        self._refresh_lists()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Save section
        save_group = QGroupBox("💾 Save Current Configuration")
        save_layout = QVBoxLayout()
        
        save_input_layout = QHBoxLayout()
        save_input_layout.addWidget(QLabel("Configuration Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., My Robot Commands")
        save_input_layout.addWidget(self.name_input)
        save_layout.addLayout(save_input_layout)
        
        save_btn = QPushButton("💾 Save Configuration")
        save_btn.clicked.connect(self._save_configuration)
        save_layout.addWidget(save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Recent section
        recent_group = QGroupBox("📂 Recent Configurations")
        recent_layout = QVBoxLayout()
        
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._load_recent)
        recent_layout.addWidget(self.recent_list)
        
        recent_btn_layout = QHBoxLayout()
        load_recent_btn = QPushButton("Load Selected")
        load_recent_btn.clicked.connect(self._load_recent)
        recent_btn_layout.addWidget(load_recent_btn)
        recent_layout.addLayout(recent_btn_layout)
        
        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)
        
        # All configurations section
        all_group = QGroupBox("📁 All Saved Configurations")
        all_layout = QVBoxLayout()
        
        self.all_list = QListWidget()
        self.all_list.itemDoubleClicked.connect(self._load_selected)
        all_layout.addWidget(self.all_list)
        
        all_btn_layout = QHBoxLayout()
        load_all_btn = QPushButton("Load Selected")
        load_all_btn.clicked.connect(self._load_selected)
        all_btn_layout.addWidget(load_all_btn)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._delete_configuration)
        all_btn_layout.addWidget(delete_btn)
        
        all_layout.addLayout(all_btn_layout)
        
        all_group.setLayout(all_layout)
        layout.addWidget(all_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def _refresh_lists(self):
        """Refresh configuration lists."""
        # Recent configurations
        self.recent_list.clear()
        recent = self.config_manager.get_recent_configurations()
        for config in recent:
            self.recent_list.addItem(f"{config['name']} - {config['accessed'][:10]}")
        
        # All configurations
        self.all_list.clear()
        all_configs = self.config_manager.get_all_configurations()
        for filepath in all_configs:
            import os
            name = os.path.basename(filepath).replace('.json', '').replace('_', ' ')
            self.all_list.addItem(name)
    
    def _save_configuration(self):
        """Save current configuration."""
        name = self.name_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a configuration name.")
            return
        
        success = self.config_manager.save_configuration(
            name,
            self.backend.gesture_controller,
            self.backend.voice_controller,
            self.backend.profile_manager
        )
        
        if success:
            QMessageBox.information(self, "Success", f"Configuration '{name}' saved successfully!")
            self.name_input.clear()
            self._refresh_lists()
        else:
            QMessageBox.critical(self, "Error", "Failed to save configuration.")
    
    def _load_recent(self):
        """Load selected recent configuration."""
        current_item = self.recent_list.currentItem()
        if not current_item:
            return
        
        # Get the corresponding recent config
        recent = self.config_manager.get_recent_configurations()
        if self.recent_list.currentRow() < len(recent):
            config = recent[self.recent_list.currentRow()]
            self._load_config_file(config['filepath'])
    
    def _load_selected(self):
        """Load selected configuration from all list."""
        current_item = self.all_list.currentItem()
        if not current_item:
            return
        
        # Get the corresponding file
        all_configs = self.config_manager.get_all_configurations()
        if self.all_list.currentRow() < len(all_configs):
            filepath = all_configs[self.all_list.currentRow()]
            self._load_config_file(filepath)
    
    def _load_config_file(self, filepath):
        """Load a configuration file."""
        success = self.config_manager.load_configuration(
            filepath,
            self.backend.gesture_controller,
            self.backend.voice_controller,
            self.backend.profile_manager
        )
        
        if success:
            QMessageBox.information(self, "Success", "Configuration loaded successfully!")
            self._refresh_lists()
            # Refresh parent dialog if it exists
            if self.parent():
                parent = self.parent()
                if hasattr(parent, '_refresh_custom_gestures'):
                    parent._refresh_custom_gestures()
                if hasattr(parent, '_refresh_custom_voices'):
                    parent._refresh_custom_voices()
                if hasattr(parent, '_load_voice_mapping'):
                    parent._load_voice_mapping()
                if hasattr(parent, '_load_gesture_mapping'):
                    parent._load_gesture_mapping()
        else:
            QMessageBox.critical(self, "Error", "Failed to load configuration.")
    
    def _delete_configuration(self):
        """Delete selected configuration."""
        current_item = self.all_list.currentItem()
        if not current_item:
            return
        
        name = current_item.text()
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete configuration '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            all_configs = self.config_manager.get_all_configurations()
            if self.all_list.currentRow() < len(all_configs):
                filepath = all_configs[self.all_list.currentRow()]
                try:
                    import os
                    os.remove(filepath)
                    QMessageBox.information(self, "Success", "Configuration deleted.")
                    self._refresh_lists()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to delete: {e}")