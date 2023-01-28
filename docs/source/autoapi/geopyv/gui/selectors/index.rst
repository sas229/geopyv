:py:mod:`geopyv.gui.selectors`
==============================

.. py:module:: geopyv.gui.selectors


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   coordinate/index.rst
   file/index.rst
   image/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.gui.selectors.ImageSelector
   geopyv.gui.selectors.CoordinateSelector
   geopyv.gui.selectors.FileSelector




.. py:class:: ImageSelector

   Graphical user interface to allow the user to select an image using the native file browser on the host OS.


   .. py:method:: get_path(message, directory)

      Method to get the path of the selected file.

      :param message: Message to display on the file browser window header.
      :type message: str
      :param directory: Directory to start from.
      :type directory: str

      :returns: **path** -- Path to selected image file.
      :rtype: str



.. py:class:: CoordinateSelector

   .. py:method:: select(f_img, template)

      Method to select f_coord if not supplied by the user.


   .. py:method:: on_click(event)

      Method to store and plot the currently selected coordinate in self.f_coord.


   .. py:method:: selected(event)

      Method to print the selected coordinates.



.. py:class:: FileSelector

   Graphical user interface to allow the user to select a results file using the native file browser on the host OS.


   .. py:method:: get_path(message, directory)

      Method to get the path of the selected file.

      :param message: Message to display on the file browser window header.
      :type message: str
      :param directory: Directory to start from.
      :type directory: str

      :returns: **path** -- Path to selected image file.
      :rtype: str



