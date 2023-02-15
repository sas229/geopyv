:py:mod:`geopyv.sequence`
=========================

.. py:module:: geopyv.sequence

.. autoapi-nested-parse::

   Sequence module for geopyv.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.sequence.Sequence
   geopyv.sequence.SequenceBase
   geopyv.sequence.SequenceResults




.. py:class:: Sequence(*, image_folder='.', image_file_type='.jpg', target_nodes=1000, boundary=None, exclusions=[], size_lower_bound=1, size_upper_bound=1000)

   Bases: :py:obj:`SequenceBase`

   Initialisation of geopyv sequence object.


.. py:class:: SequenceBase

   Bases: :py:obj:`geopyv.object.Object`

   Base class object initialiser.

   :param object_type: Object type.
   :type object_type: str

   .. py:method:: contour(mesh_index=None, quantity='C_ZNCC', imshow=True, colorbar=True, ticks=None, mesh=False, alpha=0.75, levels=None, axis=None, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot the contours of a given measure.



   .. py:method:: convergence(mesh=None, subset=None, quantity=None, show=True, block=True, save=None)

      Method to plot the rate of convergence for a mesh or subset.


   .. py:method:: inspect(mesh=None, subset=None, show=True, block=True, save=None)

      Method to show the sequence and associated mesh and subset properties.


   .. py:method:: quiver(mesh_index=None, scale=1, imshow=True, mesh=False, axis=None, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot a quiver plot of the displacements.




.. py:class:: SequenceResults(data)

   Bases: :py:obj:`SequenceBase`

   Initialisation of geopyv SequenceResults class.


