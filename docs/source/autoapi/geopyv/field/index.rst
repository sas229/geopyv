:py:mod:`geopyv.field`
======================

.. py:module:: geopyv.field

.. autoapi-nested-parse::

   Field module for geopyv.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.field.Field
   geopyv.field.FieldBase




.. py:class:: Field(*, series=None, target_particles=1000, moving=True, boundary=None, exclusions=[])

   Bases: :py:obj:`FieldBase`

   Base class object initialiser.

   :param object_type: Object type.
   :type object_type: str


.. py:class:: FieldBase

   Bases: :py:obj:`geopyv.object.Object`

   Base class object initialiser.

   :param object_type: Object type.
   :type object_type: str

   .. py:method:: inspect(mesh=True, show=True, block=True, save=None)

      Method to show the particles and associated representative areas.


   .. py:method:: volume_divergence(show=True, block=True, save=None)

      Method to show the volumetric error in the particle field.
