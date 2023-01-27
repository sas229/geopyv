from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

class ImageSelector:

    def __init__(self):
        """Graphical user interface to allow the user to select an image using the native file browser on the host OS."""
        app = QApplication.instance()
        if app is None: 
            app = QApplication()
        self.widget = QWidget()
        
    def get_path(self, message, directory):
        """Method to get the path of the selected file.
        
        Parameters
        ----------
        message : str
            Message to display on the file browser window header.
        directory : str
            Directory to start from.

        Returns
        -------
        path : str
            Path to selected image file.
            
        """
        path, _ = QFileDialog.getOpenFileName(self.widget, directory, message, "Image files (*.png *.bmp *.jpg)")
        return path