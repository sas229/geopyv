:py:mod:`geopyv.gui.selectors`
==============================

.. py:module:: geopyv.gui.selectors


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   coordinate/index.rst
   image/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.gui.selectors.ImageSelector
   geopyv.gui.selectors.CoordinateSelector




.. py:class:: ImageSelector

   .. py:method:: get_path(message, directory)



.. py:class:: CoordinateSelector

   .. py:method:: select(f_img, template)

      Method to select f_coord if not supplied by the user.


   .. py:method:: on_click(event)

      Method to store and plot the currently selected coordinate in self.f_coord.


   .. py:method:: selected(event)

      Method to print the selected coordinates.



