:py:mod:`geopyv.sequence`
=========================

.. py:module:: geopyv.sequence


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.sequence.SequenceBase
   geopyv.sequence.Sequence




Attributes
~~~~~~~~~~

.. autoapisummary::

   geopyv.sequence.log


.. py:data:: log

   

.. py:class:: SequenceBase

   Sequence base class to be used as a mixin.

   .. py:method:: inspect(mesh=None, show=True, block=True, save=None)

      Method to show the sequence and associated mesh properties.



.. py:class:: Sequence(*, image_folder='.', image_file_type='.jpg', target_nodes=1000, boundary=None, exclusions=[], size_lower_bound=1, size_upper_bound=1000)

   Bases: :py:obj:`SequenceBase`

   Initialisation of geopyv sequence object.

   .. py:method:: solve(*, trace=False, seed_coord=None, template=gp.templates.Circle(50), max_iterations=15, max_norm=0.001, adaptive_iterations=0, method='ICGN', order=1, tolerance=0.7, alpha=0.5, beta=2)


   .. py:method:: particle(coords, vols)

      A method to propogate "particles" across the domain upon which strain path interpolation is performed.



