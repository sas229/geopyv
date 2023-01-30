:orphan:

:py:mod:`geopyv.gui.selectors.file`
===================================

.. py:module:: geopyv.gui.selectors.file


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.gui.selectors.file.FileSelector




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



