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




.. py:class:: Sequence(*, image_folder='.', image_file_type='.jpg', target_nodes=1000, boundary=None, exclusions=[], size_lower_bound=1, size_upper_bound=1000)

   Bases: :py:obj:`SequenceBase`

   Initialisation of geopyv sequence object.

   .. py:method:: particle(coords, vols)

      A method to propogate "particles" across the domain upon which strain path interpolation is performed.



.. py:class:: SequenceBase

   Sequence base class to be used as a mixin.

   .. py:method:: inspect(mesh=None, show=True, block=True, save=None)

      Method to show the sequence and associated mesh properties.
