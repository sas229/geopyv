:py:mod:`geopyv.templates`
==========================

.. py:module:: geopyv.templates


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.templates.Template
   geopyv.templates.Circle
   geopyv.templates.Square




.. py:class:: Template(size)

   Class for geopyv template.

   :param size: Size of the subset.
   :type size: int

   .. attribute:: shape

      String describing the shape of the subset.

      :type: `str`

   .. attribute:: dimension

      String describing the meaning of the size attribute.

      :type: `str`

   .. attribute:: size

      Size of the subset.

      :type: `int`

   .. attribute:: n_px

      Number of pixels in the subset template.

      :type: `int`

   .. attribute:: coords

      2D array of subset template coordinates of type `float`.

      :type: `numpy.ndarray` (Nx, 2)

   .. py:method:: _check_size_and_type()

      Private method to check if size is a positive integer, and if not convert to
      a positive integer.


   .. py:method:: mask(centre, mask)

      Method to mask subset based on binary mask from mesh.



.. py:class:: Circle(radius=25)

   Bases: :py:obj:`Template`

   Class for circular subset template. Subclassed from Template.

   :param radius: Radius of the subset.
   :type radius: int

   .. attribute:: shape

      String describing the shape of the subset.

      :type: `str`

   .. attribute:: dimension

      String describing the meaning of the size attribute.

      :type: `str`

   .. attribute:: size

      Radius of the subset.

      :type: `int`

   .. attribute:: n_px

      Number of pixels in the subset template.

      :type: `int`

   .. attribute:: coords

      2D array of subset template coordinates of type `float`.

      :type: `numpy.ndarray` (Nx, 2)


.. py:class:: Square(length=50)

   Bases: :py:obj:`Template`

   Class for square subset template. Subclassed from Template.

   :param length: Half length of the side of the subset.
   :type length: int

   .. attribute:: shape

      String describing the shape of the subset.

      :type: `str`

   .. attribute:: dimension

      String describing the meaning of the size attribute.

      :type: `str`

   .. attribute:: size

      Half length of side of the subset.

      :type: `int`

   .. attribute:: n_px

      Number of pixels in the subset template.

      :type: `int`

   .. attribute:: coords

      2D array of subset template coordinates of type `float`.

      :type: `numpy.ndarray` (Nx, 2)


