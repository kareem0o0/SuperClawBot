"""
Model configuration dialog for loading models and editing mappings.
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QListWidget, QLineEdit, QMessageBox,
                               QFileDialog, QGroupBox, QTabWidget, QWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt

from core.model_manager import ModelManager


class ModelConfigDialog(QDialog):
    """Dialog for configuring models and class-to-letter mappings."""
    
    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.model_manager = ModelManager(backend.signals)
        self.setWindowTitle("Model Configuration")
        self.setMinimumSize(700, 600)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Tabs for Voice and Gesture
        tabs = QTabWidget()
        tabs.addTab(self._create_voice_tab(), "Voice Models")
        tabs.addTab(self._create_gesture_tab(), "Gesture Models")
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def _create_voice_tab(self):
        """Create voice model configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Load new model section
        load_group = QGroupBox("Load New Voice Model")
        load_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.voice_load_btn = QPushButton("Load .tflite and labels.txt")
        self.voice_load_btn.clicked.connect(lambda: self._load_new_model("voice"))
        btn_layout.addWidget(self.voice_load_btn)
        load_layout.addLayout(btn_layout)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Current model section
        current_group = QGroupBox("Current Voice Model Mapping")
        current_layout = QVBoxLayout()
        
        self.voice_model_label = QLabel("No model loaded")
        current_layout.addWidget(self.voice_model_label)
        
        self.voice_table = QTableWidget()
        self.voice_table.setColumnCount(3)
        self.voice_table.setHorizontalHeaderLabels(["Class Name", "Assigned Letter", "Action"])
        self.voice_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.voice_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.voice_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        current_layout.addWidget(self.voice_table)
        
        save_btn = QPushButton("Save Mapping")
        save_btn.clicked.connect(lambda: self._save_mapping("voice"))
        current_layout.addWidget(save_btn)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # Load current mapping
        self._load_voice_mapping()
        
        widget.setLayout(layout)
        return widget
    
    def _create_gesture_tab(self):
        """Create gesture model configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Load new model section
        load_group = QGroupBox("Load New Gesture Model")
        load_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.gesture_load_btn = QPushButton("Load .tflite and labels.txt")
        self.gesture_load_btn.clicked.connect(lambda: self._load_new_model("gesture"))
        btn_layout.addWidget(self.gesture_load_btn)
        load_layout.addLayout(btn_layout)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Custom gestures section
        custom_group = QGroupBox("Custom Gestures (Auto-Learn)")
        custom_layout = QVBoxLayout()
        
        custom_btn_layout = QHBoxLayout()
        
        add_custom_btn = QPushButton("➕ Add Custom Gesture")
        add_custom_btn.clicked.connect(self._add_custom_gesture)
        custom_btn_layout.addWidget(add_custom_btn)
        
        remove_custom_btn = QPushButton("➖ Remove Custom Gesture")
        remove_custom_btn.clicked.connect(self._remove_custom_gesture)
        custom_btn_layout.addWidget(remove_custom_btn)
        
        custom_layout.addLayout(custom_btn_layout)
        
        self.custom_gesture_list = QListWidget()
        custom_layout.addWidget(self.custom_gesture_list)
        
        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)
        
        # Current model section
        current_group = QGroupBox("Current Gesture Model Mapping")
        current_layout = QVBoxLayout()
        
        self.gesture_model_label = QLabel("No model loaded")
        current_layout.addWidget(self.gesture_model_label)
        
        self.gesture_table = QTableWidget()
        self.gesture_table.setColumnCount(3)
        self.gesture_table.setHorizontalHeaderLabels(["Class Name", "Assigned Letter", "Action"])
        self.gesture_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.gesture_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.gesture_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        current_layout.addWidget(self.gesture_table)
        
        save_btn = QPushButton("Save Mapping")
        save_btn.clicked.connect(lambda: self._save_mapping("gesture"))
        current_layout.addWidget(save_btn)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # Load current mapping
        self._load_gesture_mapping()
        self._refresh_custom_gestures()
        
        widget.setLayout(layout)
        return widget
    
    def _load_new_model(self, model_type):
        """Load a new model with file dialogs."""
        # Select .tflite file
        tflite_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {model_type.capitalize()} Model (.tflite)",
            "",
            "TFLite Files (*.tflite)"
        )
        
        if not tflite_path:
            return
        
        # Select labels.txt file
        labels_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Labels File (labels.txt)",
            "",
            "Text Files (*.txt)"
        )
        
        if not labels_path:
            return
        
        # Load labels
        labels = self.model_manager.load_labels_from_file(labels_path)
        if not labels:
            QMessageBox.warning(self, "Error", "Could not load labels from file.")
            return
        
        # Install model
        if model_type == "voice":
            dest_dir = "resources/sound_classifier"
        else:
            dest_dir = "resources/gesture_classifier"
        
        model_name, success = self.model_manager.install_model(
            tflite_path, labels_path, model_type, dest_dir
        )
        
        if not success:
            QMessageBox.critical(self, "Error", "Failed to install model files.")
            return
        
        # Create default mapping
        default_mapping = self.model_manager.create_default_mapping(labels)
        
        # Show mapping editor dialog
        final_mapping = self._edit_mapping_dialog(labels, default_mapping, model_type)
        
        if final_mapping:
            # Save mapping
            self.model_manager.save_mapping(model_name, model_type, final_mapping)
            
            # Load model into controller
            if model_type == "voice":
                success = self.backend.voice_controller.load_new_model(model_name)
                if success:
                    self._load_voice_mapping()
                    QMessageBox.information(self, "Success", f"Voice model '{model_name}' loaded successfully!")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load voice model into controller.")
            else:
                success = self.backend.gesture_controller.load_new_model(model_name)
                if success:
                    self._load_gesture_mapping()
                    self._refresh_custom_gestures()
                    QMessageBox.information(self, "Success", f"Gesture model '{model_name}' loaded successfully!")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load gesture model into controller.")
    
    def _edit_mapping_dialog(self, labels, current_mapping, model_type):
        """Show dialog to edit class-to-letter mappings."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Configure {model_type.capitalize()} Class Mappings")
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        
        info_label = QLabel("Assign a unique single letter to each class:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Table for editing
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Class Name", "Assigned Letter"])
        table.setRowCount(len(labels))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        # Populate table
        for i, label in enumerate(labels):
            # Class name (read-only)
            class_item = QTableWidgetItem(label)
            class_item.setFlags(class_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, class_item)
            
            # Letter (editable)
            letter = current_mapping.get(label, label)
            letter_item = QTableWidgetItem(letter)
            table.setItem(i, 1, letter_item)
        
        layout.addWidget(table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        validate_btn = QPushButton("Validate Mapping")
        validate_btn.clicked.connect(lambda: self._validate_table_mapping(table, labels))
        btn_layout.addWidget(validate_btn)
        
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec() == QDialog.Accepted:
            # Extract mapping from table
            mapping = {}
            for i, label in enumerate(labels):
                letter = table.item(i, 1).text().strip()
                mapping[label] = letter
            
            # Final validation
            is_valid, dup_letter, dup_classes = self.model_manager.validate_mapping(mapping)
            if not is_valid:
                QMessageBox.warning(
                    self,
                    "Duplicate Letter",
                    f"Letter '{dup_letter}' is assigned to multiple classes: {', '.join(dup_classes)}\n\n"
                    "Please fix duplicates before saving."
                )
                return None
            
            return mapping
        
        return None
    
    def _validate_table_mapping(self, table, labels):
        """Validate mapping from table."""
        mapping = {}
        for i, label in enumerate(labels):
            letter = table.item(i, 1).text().strip()
            mapping[label] = letter
        
        is_valid, dup_letter, dup_classes = self.model_manager.validate_mapping(mapping)
        
        if is_valid:
            QMessageBox.information(self, "Valid", "All mappings are unique! ✓")
        else:
            QMessageBox.warning(
                self,
                "Duplicate Letter",
                f"Letter '{dup_letter}' is assigned to multiple classes:\n\n"
                f"{', '.join(dup_classes)}\n\n"
                "Please choose different letters."
            )
    
    def _load_voice_mapping(self):
        """Load current voice model mapping into table."""
        if not self.backend.voice_controller.model:
            self.voice_model_label.setText("No model loaded")
            self.voice_table.setRowCount(0)
            return
        
        model_name = self.backend.voice_controller.current_model_name
        self.voice_model_label.setText(f"Model: {model_name}")
        
        mapping = self.backend.voice_controller.get_current_mapping()
        labels = self.backend.voice_controller.model.get_labels()
        
        self.voice_table.setRowCount(len(labels))
        
        for i, label in enumerate(labels):
            # Class name
            class_item = QTableWidgetItem(label)
            class_item.setFlags(class_item.flags() & ~Qt.ItemIsEditable)
            self.voice_table.setItem(i, 0, class_item)
            
            # Letter (editable)
            letter = mapping.get(label, "")
            letter_item = QTableWidgetItem(letter)
            self.voice_table.setItem(i, 1, letter_item)
            
            # Edit button
            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(lambda checked, row=i: self._edit_cell("voice", row))
            self.voice_table.setCellWidget(i, 2, edit_btn)
    
    def _load_gesture_mapping(self):
        """Load current gesture model mapping into table."""
        if not self.backend.gesture_controller.model:
            self.gesture_model_label.setText("No model loaded")
            self.gesture_table.setRowCount(0)
            return
        
        model_name = self.backend.gesture_controller.current_model_name
        self.gesture_model_label.setText(f"Model: {model_name}")
        
        mapping = self.backend.gesture_controller.get_current_mapping()
        
        # Get all classes (regular + custom)
        regular_labels = self.backend.gesture_controller.model.get_labels()
        custom_gestures = self.backend.gesture_controller.get_custom_gestures()
        
        all_labels = list(regular_labels) + [f"[CUSTOM] {g}" for g in custom_gestures]
        
        self.gesture_table.setRowCount(len(all_labels))
        
        for i, label in enumerate(all_labels):
            # Class name
            class_item = QTableWidgetItem(label)
            class_item.setFlags(class_item.flags() & ~Qt.ItemIsEditable)
            self.gesture_table.setItem(i, 0, class_item)
            
            # Letter (editable)
            letter = mapping.get(label, "")
            letter_item = QTableWidgetItem(letter)
            self.gesture_table.setItem(i, 1, letter_item)
            
            # Edit button
            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(lambda checked, row=i: self._edit_cell("gesture", row))
            self.gesture_table.setCellWidget(i, 2, edit_btn)
    
    def _edit_cell(self, model_type, row):
        """Edit a specific mapping cell."""
        if model_type == "voice":
            table = self.voice_table
        else:
            table = self.gesture_table
        
        # Make cell editable temporarily
        table.editItem(table.item(row, 1))
    
    def _save_mapping(self, model_type):
        """Save mapping from table."""
        if model_type == "voice":
            table = self.voice_table
            controller = self.backend.voice_controller
        else:
            table = self.gesture_table
            controller = self.backend.gesture_controller
        
        if not controller.model:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return
        
        # Extract mapping from table
        mapping = {}
        
        for i in range(table.rowCount()):
            class_name = table.item(i, 0).text()
            letter = table.item(i, 1).text().strip()
            mapping[class_name] = letter
        
        # Validate
        is_valid, dup_letter, dup_classes = self.model_manager.validate_mapping(mapping)
        
        if not is_valid:
            QMessageBox.warning(
                self,
                "Duplicate Letter",
                f"Letter '{dup_letter}' is assigned to multiple classes:\n\n"
                f"{', '.join(dup_classes)}\n\n"
                "Please fix duplicates before saving."
            )
            return
        
        # Save
        success = controller.update_mapping(mapping)
        
        if success:
            QMessageBox.information(self, "Success", "Mapping saved successfully!")
        else:
            QMessageBox.critical(self, "Error", "Failed to save mapping.")
    
    def _add_custom_gesture(self):
        """Open dialog to add custom gesture."""
        from .custom_gesture_dialog import CustomGestureDialog
        
        controller = self.backend.gesture_controller
        
        if not controller.model or not controller.camera:
            QMessageBox.warning(self, "Not Available","Gesture model and camera must be loaded first.")
            return
        
        # Get existing letters
        existing_letters = set(controller.get_current_mapping().values())
        model_path = controller.model.model_dir + f"/{controller.current_model_name}.tflite"
        
        dialog = CustomGestureDialog(controller.camera, model_path, existing_letters, self)
        
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_gesture_data()
            controller.add_custom_gesture(data['name'], data['embeddings'], data['letter'])
            self._refresh_custom_gestures()
            self._load_gesture_mapping()
            QMessageBox.information(self, "Success", 
                                  f"Custom gesture '{data['name']}' added successfully!")
    
    def _remove_custom_gesture(self):
        """Remove selected custom gesture."""
        current_item = self.custom_gesture_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a custom gesture to remove.")
            return
        
        gesture_name = current_item.text()
        
        reply = QMessageBox.question(
            self, "Confirm Remove",
            f"Remove custom gesture '{gesture_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.backend.gesture_controller.remove_custom_gesture(gesture_name)
            self._refresh_custom_gestures()
            self._load_gesture_mapping()
    
    def _refresh_custom_gestures(self):
        """Refresh custom gesture list."""
        self.custom_gesture_list.clear()
        gestures = self.backend.gesture_controller.get_custom_gestures()
        for gesture in gestures:
            self.custom_gesture_list.addItem(gesture)