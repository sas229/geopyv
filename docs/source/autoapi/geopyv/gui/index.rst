:py:mod:`geopyv.gui`
====================

.. py:module:: geopyv.gui


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   CoordinateSelector/index.rst
   Gui/index.rst
   ImageSelector/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.gui.ImageSelector
   geopyv.gui.CoordinateSelector




.. py:class:: ImageSelector

   .. py:method:: get_path(message, directory)



.. py:class:: CoordinateSelector

   .. py:method:: select(f_img, template)

      Method to select f_coord if not supplied by the user.


   .. py:method:: on_click(event)

      Method to store and plot the currently selected coordinate in self.f_coord.


   .. py:method:: selected(event)

      Method to print the selected coordinates.



