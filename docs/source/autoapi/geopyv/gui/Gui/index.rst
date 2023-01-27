:py:mod:`geopyv.gui.Gui`
========================

.. py:module:: geopyv.gui.Gui


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.gui.Gui.modQGraphicsPathItem
   geopyv.gui.Gui.modQGraphicsView
   geopyv.gui.Gui.modQGraphicsPolygonItem
   geopyv.gui.Gui.modQGraphicsEllipseItem
   geopyv.gui.Gui.modQGraphicsScene
   geopyv.gui.Gui.modQMainWindow
   geopyv.gui.Gui.Gui




.. py:class:: modQGraphicsPathItem(annotation_item, index)

   Bases: :py:obj:`QGraphicsPathItem`

   .. py:method:: hoverEnterEvent(event)


   .. py:method:: hoverLeaveEvent(event)


   .. py:method:: mouseReleaseEvent(event)


   .. py:method:: itemChange(change, value)


   .. py:attribute:: marker

      

   .. py:attribute:: crosshair

      


.. py:class:: modQGraphicsView(parent=None)

   Bases: :py:obj:`QGraphicsView`

   .. py:method:: fitInView(scale=True)


   .. py:method:: wheelEvent(event)


   .. py:method:: mousePressEvent(event)

      Start mouse pan or zoom mode.



   .. py:method:: mouseReleaseEvent(event)

      Stop mouse pan or zoom mode (apply zoom if valid).




.. py:class:: modQGraphicsPolygonItem(mode, parent=None)

   Bases: :py:obj:`QGraphicsPolygonItem`

   .. py:method:: modeInit()


   .. py:method:: number_of_points()


   .. py:method:: addPoint(p)


   .. py:method:: removePoint(index, check)


   .. py:method:: movePoint(i, p)


   .. py:method:: move_item(index, pos)


   .. py:method:: itemChange(change, value)


   .. py:method:: hoverEnterEvent(event)


   .. py:method:: hoverLeaveEvent(event)



.. py:class:: modQGraphicsEllipseItem(mode, parent=None)

   Bases: :py:obj:`QGraphicsEllipseItem`


.. py:class:: modQGraphicsScene(parent=None)

   Bases: :py:obj:`QGraphicsScene`

   .. py:method:: modeUpdate(ref)


   .. py:method:: mousePressEvent(event)


   .. py:method:: mouseMoveEvent(event)


   .. py:method:: keyPressEvent(event)



.. py:class:: modQMainWindow(img)

   Bases: :py:obj:`QMainWindow`

   .. py:method:: clicked(checked)


   .. py:method:: closeEvent(event)



.. py:class:: Gui(img)

   .. py:method:: main()



