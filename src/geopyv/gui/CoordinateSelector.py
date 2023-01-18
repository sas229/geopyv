import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QPixmap
import numpy as np
import matplotlib.pyplot as plt
from geopyv.templates import Circle, Square

log = logging.getLogger(__name__)

class CoordinateSelector:

    def __init__(self):
        self.app = QApplication.instance()
        if self.app is None: 
            self.app = QApplication()
        self.window = QMainWindow()
        self.window.setWindowTitle("Select coordinate.")
        self.widget = QWidget()               
        self.window.setCentralWidget(self.widget)    
        self.layout = QVBoxLayout(self.widget)
        self.label = QLabel(self.window)
        self.layout.addWidget(self.label)
        self.app.exec()

    def select(self, f_img, template):
        print("Selecting coordinate.")
        self.pixmap = QPixmap(f_img.filepath)
        self.label.setPixmap(self.pixmap)
        self.window.resize(self.pixmap.width(), self.pixmap.height())
        print(f_img.filepath)
        self.window.show()
        f_coord = np.asarray([np.shape(f_img.image_gs)[0]/2, np.shape(f_img.image_gs)[1]/2])
        return f_coord

    
    # def select(self, f_img, template):
    #     """Method to select f_coord if not supplied by the user."""
    #     log.warning("No coordinate supplied. Please select the target coordinate for the subset.")
    #     self.f_img = f_img
    #     self.template = template
    #     self.x = None
    #     self.y = None
    #     f, ax = plt.subplots(num="Right click on the target coordinate for the subset and close to save")
    #     f.canvas.mpl_connect('button_press_event', self.on_click)
    #     f.canvas.mpl_connect('close_event', self.selected)
    #     ax.imshow(self.f_img.image_gs, cmap="gist_gray")
    #     plt.tight_layout()
    #     plt.show()
    #     self.coord = np.asarray([self.x, self.y])
    #     return self.coord

    # def on_click(self, event):
    #     """Method to store and plot the currently selected coordinate in self.f_coord."""
    #     if event.button==3:
    #         if event.xdata != None and event.ydata != None:
    #             if event.xdata > self.template.size and event.xdata < np.shape(self.f_img.image_gs)[0]-self.template.size:
    #                 if event.ydata > self.template.size and event.ydata < np.shape(self.f_img.image_gs)[1]-self.template.size:
    #                     self.x = np.round(event.xdata, 0)
    #                     self.y = np.round(event.ydata, 0)
    #                     self.f_coord = np.asarray([self.x, self.y])
    #                     ax = event.inaxes
    #                     f = ax.get_figure()
    #                     num_lines = len(ax.lines)
    #                     while num_lines > 0:
    #                         ax.lines.pop()
    #                         num_lines = len(ax.lines)
    #                     ax.plot(self.x, self.y, marker="+", color="y", zorder=10)
    #                     if type(self.template) == Circle:
    #                         theta = np.linspace(0, 2*np.pi, 150)
    #                         radius = self.template.size
    #                         x = self.x+radius*np.cos(theta)
    #                         y = self.y+radius*np.sin(theta)
    #                         ax.plot(x, y, color='y')
    #                     elif type(self.template) == Square:
    #                         x = [
    #                             self.x-self.template.size, 
    #                             self.x-self.template.size,
    #                             self.x+self.template.size,
    #                             self.x+self.template.size,
    #                             self.x-self.template.size,
    #                         ]
    #                         y = [
    #                             self.y-self.template.size, 
    #                             self.y+self.template.size,
    #                             self.y+self.template.size,
    #                             self.y-self.template.size,
    #                             self.y-self.template.size,
    #                         ]
    #                         ax.plot(x, y, color='y')
    #                     f.canvas.draw()
    #                     f.canvas.flush_events()

    # def selected(self, event):
    #     """Method to print the selected coordinates."""
    #     log.info("Coordinate selected: {}, {}".format(self.x, self.y))