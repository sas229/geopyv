from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

class ImageSelector:

    def __init__(self):
        app = QApplication.instance()
        if app is None: 
            app = QApplication()
        self.widget = QWidget()
        
    def get_path(self, message, directory):
        path, _ = QFileDialog.getOpenFileName(self.widget, directory, message, "Image files (*.png *.bmp *.jpg)")
        return path