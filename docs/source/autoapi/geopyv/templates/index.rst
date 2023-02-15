:py:mod:`geopyv.templates`
==========================

.. py:module:: geopyv.templates

.. autoapi-nested-parse::

   Templates module for geopyv.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.templates.Circle
   geopyv.templates.Square
   geopyv.templates.Template




.. py:class:: Circle(radius=25)

   Bases: :py:obj:`Template`

   Class for circular subset template. Subclassed from Template.

   :param radius: Radius of the subset. Defaults to a value of 25.
   :type radius: int, optional

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

   .. attribute:: subset_mask

      2D array of coordinates to mask of type `float`.

      :type: `numpy.ndarray` (Nx, 2)


.. py:class:: Square(length=25)

   Bases: :py:obj:`Template`

   Class for square subset template. Subclassed from Template.

   :param length: Half length of the side of the subset. Defaults to a value of 25.
   :type length: int, optional

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

   .. attribute:: subset_mask

      2D array of coordinates to mask of type `float`.

      :type: `numpy.ndarray` (Nx, 2)


.. py:class:: Template(size)

   Base class for geopyv subset template.

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

   .. attribute:: subset_mask

      2D array of coordinates to mask of type `float`.

      :type: `numpy.ndarray` (Nx, 2)

   .. py:method:: mask(centre, mask)

      Method to mask subset based on binary mask from mesh.

      :param centre: Centre of subset.
      :type centre: `numpy.ndarray` (x,y)
      :param mask: Mask to be applied to the mesh.
                   Value of 0 indicates pixels to mask in template.
      :type mask: `numpy.ndarray` (Nx,Ny)



