import logging
import numpy as np
import matplotlib.pyplot as plt
import geopyv.templates as templates

log = logging.getLogger(__name__)


class CoordinateSelector:
    def select(self, f_img, template):
        """Method to select f_coord if not supplied by the user."""
        log.warning(
            "No coordinate supplied. Please select the target "
            "coordinate for the subset."
        )
        self.f_img = f_img
        self.template = template
        self.x = None
        self.y = None
        self.f, self.ax = plt.subplots(
            num="Right click on the target coordinate for the subset "
            "and close to save"
        )
        self.f.canvas.mpl_connect("button_press_event", self.on_click)
        self.f.canvas.mpl_connect("close_event", self.selected)
        self.ax.imshow(self.f_img.image_gs, cmap="gist_gray")
        self.ax.set_axis_off()
        self.point = None
        self.boundary = None
        plt.tight_layout()
        plt.show()
        self.coord = np.asarray([self.x, self.y])
        return self.coord

    def on_click(self, event):
        """

        Method to store and plot the currently selected coordinate in self.f_coord.

        """
        if event.button == 3:
            if event.xdata is not None and event.ydata is not None:
                if (
                    event.xdata > self.template.size
                    and event.xdata
                    < np.shape(self.f_img.image_gs)[0] - self.template.size
                ):
                    if (
                        event.ydata > self.template.size
                        and event.ydata
                        < np.shape(self.f_img.image_gs)[1] - self.template.size
                    ):
                        self.x = np.round(event.xdata, 0)
                        self.y = np.round(event.ydata, 0)
                        self.f_coord = np.asarray([self.x, self.y])
                        if self.point is not None:
                            self.point[0].remove()
                        if self.boundary is not None:
                            self.boundary[0].remove()
                        self.point = self.ax.plot(
                            self.x,
                            self.y,
                            marker="+",
                            color="b",
                            zorder=10,
                        )
                        if type(self.template) == templates.Circle:
                            theta = np.linspace(0, 2 * np.pi, 150)
                            radius = self.template.size
                            x = self.x + radius * np.cos(theta)
                            y = self.y + radius * np.sin(theta)
                            self.boundary = self.ax.plot(x, y, color="b")
                        elif type(self.template) == templates.Square:
                            x = [
                                self.x - self.template.size,
                                self.x - self.template.size,
                                self.x + self.template.size,
                                self.x + self.template.size,
                                self.x - self.template.size,
                            ]
                            y = [
                                self.y - self.template.size,
                                self.y + self.template.size,
                                self.y + self.template.size,
                                self.y - self.template.size,
                                self.y - self.template.size,
                            ]
                            self.boundary = self.ax.plot(x, y, color="b")
                        self.f.canvas.draw()
                        self.f.canvas.flush_events()

    def selected(self, event):
        """Method to print the selected coordinates."""
        log.info("Coordinate selected: {}, {}".format(self.x, self.y))
