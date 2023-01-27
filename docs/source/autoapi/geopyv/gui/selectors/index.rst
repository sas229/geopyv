:orphan:

:py:mod:`geopyv.gui.selectors`
==============================

.. py:module:: geopyv.gui.selectors


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.gui.selectors.ImageSelector




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



