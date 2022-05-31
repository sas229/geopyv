from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
import numpy as np


class modQGraphicsPathItem(QGraphicsPathItem):

    marker = QPainterPath()
    crosshair = QPolygonF([QPointF(1,10), QPointF(1,1), QPointF(10,1),
                        QPointF(10,-1), QPointF(1,-1), QPointF(1,-10), 
                        QPointF(-1,-10), QPoint(-1,-1), QPointF(-10,-1), 
                        QPointF(-10,1), QPointF(-1,1), QPointF(-1,10),
                        QPointF(1,10)])
    marker.addPolygon(crosshair)

    def __init__(self, annotation_item, index):
        super(modQGraphicsPathItem, self).__init__()
        self.m_item = annotation_item 
        self.m_index = index
        self.setPath(modQGraphicsPathItem.marker)
        self.setBrush(Qt.red)
        self.setPen(QPen(Qt.black, 0.5))
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(11)

    def hoverEnterEvent(self, event):
        self.setBrush(Qt.green)
        super(modQGraphicsPathItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(Qt.red)
        super(modQGraphicsPathItem, self).hoverLeaveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setSelected(False)
        super(modQGraphicsPathItem, self).mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.isEnabled():
            self.m_item.movePoint(self.m_index, value)
        return super(modQGraphicsPathItem, self).itemChange(change, value)


class modQGraphicsView(QGraphicsView):

    def __init__ (self, parent=None):
        super(modQGraphicsView, self).__init__(parent)
        self.setMouseTracking(True)
        self.zoom = 5
        self.scene = modQGraphicsScene(self)
        self.photo = QGraphicsPixmapItem()
        self.scene.addItem(self.photo)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.CrossCursor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded) # Scrollbar policy.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded) # Scrollbar policy. 

    def fitInView(self, scale=True):
        rect = QRectF(self.photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self.zoom = 0

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self.zoom += 1
        else:
            factor = 0.75
            self.zoom -= 1
        if self.zoom > 0:
            self.scale(factor, factor)
        elif self.zoom == 0:
            self.fitInView()
        else:
            self.zoom = 0

    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        QGraphicsView.mousePressEvent(self, event)
    
    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        elif event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.NoDrag)
        QGraphicsView.mouseReleaseEvent(self, event)


class modQGraphicsPolygonItem(QGraphicsPolygonItem):

    def __init__(self, mode, parent=None):
        super(modQGraphicsPolygonItem, self).__init__(parent)
        self.m_points = []
        self.m_items = []
        self.mode = mode
        self.modeInit()
        self.setZValue(10)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setFillRule(Qt.WindingFill)

    def modeInit(self):
        if self.mode == 0:
            self.polyColour = QColor(Qt.blue)
        elif self.mode == 1:
            self.polyColour = QColor(Qt.magenta)
        elif self.mode == 2:
            self.polyColour = QColor(Qt.yellow)
        elif self.mode == 3:
            self.polyColour = QColor(Qt.red)
        self.setPen(QPen(self.polyColour))
        self.polyColour.setAlpha(100)
        self.setBrush(QBrush(self.polyColour))

    def number_of_points(self):
        return len(self.m_items)

    def addPoint(self, p):
        self.m_points.append(p)
        self.setPolygon(QPolygonF(self.m_points))
        item = modQGraphicsPathItem(self, len(self.m_points) - 1)
        self.scene().addItem(item)
        self.m_items.append(item)
        item.setPos(p)

    def removePoint(self, index, check):
        if self.m_points:
            self.m_points.pop(index)
            self.setPolygon(QPolygonF(self.m_points))
            it = self.m_items.pop(index)
            self.scene().removeItem(it)
            del it
            if len(self.m_points)<3 and check==True:
                self.removePoint(-1, True)

    def movePoint(self, i, p):
        if 0 <= i < len(self.m_points):
            self.m_points[i] = self.mapFromScene(p)
            self.setPolygon(QPolygonF(self.m_points))

    def move_item(self, index, pos):
        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.m_points):
                self.move_item(i, self.mapToScene(point))
        return super(QGraphicsPolygonItem, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.polyColour.setAlpha(200)
        self.setBrush(self.polyColour)
        super(QGraphicsPolygonItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.polyColour.setAlpha(100)
        self.setBrush(self.polyColour)
        super(QGraphicsPolygonItem, self).hoverLeaveEvent(event)

class modQGraphicsEllipseItem(QGraphicsEllipseItem):
    def __init__(self, mode, parent=None):
        super(modQGraphicsEllipseItem, self).__init__(parent)
        self.mode = mode
        self.setPen(QPen(QColor(Qt.magenta)))
        self.setBrush(QBrush(QColor(Qt.magenta)))
        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

class modQGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(modQGraphicsScene, self).__init__(parent)
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setCursor(QCursor(Qt.CrossCursor))
        self.addItem(self.image_item)
        self.polygon_item = None
        self.seed_item = None
        self.mode = None

    def modeUpdate(self, ref):
        self.mode = ref
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton and self.mode is not None:
            if self.mode == 1:
                if self.seed_item is None:
                    self.seed_item = modQGraphicsEllipseItem(1)
                    self.seed_item.setRect(-20, -20, 40, 40)
                    self.seed_item.setPos(event.scenePos())
                    self.addItem(self.seed_item)
                else:
                    self.seed_item.setPos(event.scenePos())
            else: 
                if self.polygon_item is None: 
                    self.polygon_item = modQGraphicsPolygonItem(self.mode)
                    self.addItem(self.polygon_item)
                self.polygon_item.removePoint(-1, False)
                self.polygon_item.addPoint(event.scenePos())
                self.polygon_item.addPoint(event.scenePos())
        super(modQGraphicsScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.polygon_item is not None:
            self.polygon_item.movePoint(self.polygon_item.number_of_points()-1, event.scenePos())
        super(modQGraphicsScene, self).mouseMoveEvent(event)

    def keyPressEvent(self,event):
        if event.key() == Qt.Key_Return or event.key()==Qt.Key_Space:
            self.polygon_item.removePoint(-1, True)
            self.polygon_item.polygon()
            self.polygon_item = None
        if event.key() == Qt.Key_Delete:
            self.selectedItems().sort(key=lambda x: x.type())
            for item in self.selectedItems():
                if item.type() == 2:
                    for it in self.items():
                        if it.type() == 5:
                            if item in it.m_items:
                                it.removePoint(it.m_items.index(item), True)
                                break
                elif item.mode == 2:
                    self.seed_item = None
                elif item.type() == 5:
                    for it in item.m_items:
                        item.removePoint(item.m_items.index(it), True)
                self.removeItem(item)
                del item
        super(modQGraphicsScene, self).keyPressEvent(event)



class modQMainWindow(QMainWindow):

    def __init__(self, img):
        super().__init__()

        # General preparation.
        self.setWindowTitle("geopyv")
        #self.resize(1000, 700) # Set initial window size (changeable).
        
        # Initiate imwidget.
        self.view = modQGraphicsView()
        self.scene = modQGraphicsScene(self)
        self.view.setScene(self.scene)
        self.image = QGraphicsPixmapItem()
        self.image.setPixmap(QPixmap(img))
        self.scene.addItem(self.image)

        # Intiate button widgets.
        self.button1 = QRadioButton("RoI") # Create button.
        self.button1.setCheckable(True)
        self.button1.clicked.connect(self.clicked) # Triggers method with arg from setCheckable.

        self.button2 = QRadioButton("Seed") # Create button.
        self.button2.setCheckable(True)
        self.button2.clicked.connect(self.clicked) # Triggers method with arg from setCheckable.

        self.button3 = QRadioButton("Holes") # Create button.
        self.button3.setCheckable(True)
        self.button3.clicked.connect(self.clicked) # Triggers method with arg from setCheckable.

        self.button4 = QRadioButton("Objects") # Create button.
        self.button4.setCheckable(True)
        self.button4.clicked.connect(self.clicked) # Triggers method with arg from setCheckable.

        # Initiate label widget.
        self.label = QLabel("Select a mode...")

        # Initiate layout and container.
        app_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        app_layout.addWidget(self.view)
        app_layout.addLayout(button_layout)
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        button_layout.addWidget(self.button4)
        app_layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(app_layout)

        self.setCentralWidget(container) # Apply widget to the window. 

    def clicked(self, checked):
        self.label.setText(r"$\bf{Instructions:}$"+"\n"
                            +r"$\bf{Left~~Click~+~Drag: }$"+"Pan"+"\n"
                            +r"$\bf{Left~~Click: }$"+"Select/Deselect"+"\n"
                            +r"$\bf{Right~~Click~+~Drag: }$"+"Selection box"+"\n"
                            +r"$\bf{Scroll: }$"+"Zoom"+"\n"
                            +r"$\bf{Middle~~Click: }$"+"Add vertex"+"\n"
                            +r"$\bf{Space/Enter: }$"+"Finish defining polygon"+"\n"
                            +r"$\bf{Delete: }$"+"Delete selected items"+"\n")
        if self.button1.isChecked():
            self.scene.modeUpdate(0)
        elif self.button2.isChecked():
            self.scene.modeUpdate(1)
            self.label.setText(r"$\bf{Instructions:}$"+"\n"
                            +r"$\bf{Left~~Click~+~Drag: }$"+"Pan"+"\n"
                            +r"$\bf{Left~~Click: }$"+"Select/Deselect"+"\n"
                            +r"$\bf{Right~~Click~+~Drag: }$"+"Selection box"+"\n"
                            +r"$\bf{Scroll: }$"+"Zoom"+"\n"
                            +r"$\bf{Middle~~Click: }$"+"Relocate vertex"+"\n"
                            +r"$\bf{Delete: }$"+"Delete selected items"+"\n")
        elif self.button3.isChecked():
            self.scene.modeUpdate(2)
        elif self.button4.isChecked():
            self.scene.modeUpdate(3)
            

    def closeEvent(self,event):
        self.roi = []
        self.sed = []
        self.obj = []
        self.hls = []
        for item in self.scene.items():
            if item.type() == 5:
                poly = []
                for pnt in item.m_points:
                    poly.append([pnt.x(), pnt.y()])
                if item.mode == 0:
                    self.roi = np.asarray(poly)
                elif item.mode == 2:
                    self.obj.append(np.asarray(poly))
                elif item.mode == 3:
                    self.hls.append(np.asarray(poly))
            elif item.type() == 4:
                self.sed = np.asarray([item.x(), item.y()])
        if self.roi == []:
            close = QMessageBox.question(self, "EXIT", 
                                        "Exiting now will terminate the process due to problematic input (e.g. seed outside RoI, RoI not defined). Do you want to proceed?", 
                                        QMessageBox.Yes | QMessageBox.No)
            if close == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore
        if self.obj == []:
            self.obj = None
        else:
            self.obj = np.asarray(self.obj)

class Gui:
    def __init__(self, img):
        self.img = img

    def main(self):
        app = QApplication([]) # Create Application instance (required once per application).
        window = modQMainWindow(self.img) # Create Widget (can use any widget as window).
        window.show() # Show Window.
        app.exec() # Start event loop.
        return window.roi, window.sed, None, np.asarray(window.hls)

