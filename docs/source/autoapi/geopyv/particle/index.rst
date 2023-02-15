:py:mod:`geopyv.particle`
=========================

.. py:module:: geopyv.particle

.. autoapi-nested-parse::

   Particle module for geopyv.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.particle.Particle
   geopyv.particle.ParticleBase
   geopyv.particle.ParticleResults




.. py:class:: Particle(*, series=None, coordinate_0=np.zeros(2), warp_0=np.zeros(12), volume_0=0.0, moving=True)

   Bases: :py:obj:`ParticleBase`

   Initialisation of geopyv particle object.

   :param meshes: Sequence for the particle object to track.
   :type meshes: `numpy.ndarray` of geopyv.mesh.Mesh objects
   :param coordinate_0: Initial particle coordinate (x,y)
   :type coordinate_0: numpy.ndarray (2)
   :param p_init: Initial warp vector.
   :type p_init: `numpy.ndarray` (12), optional
   :param vol: Volume represented by the particle.
   :type vol: float
   :param moving: Boolean for Lagrangian (False) or Eulerian (True) specification. Defaults to False.
   :type moving: bool

   .. py:method:: solve()

      Method to calculate the strain path of the particle from the
      mesh sequence and optionally the stress path employing the
      model specified by the input parameters.




.. py:class:: ParticleBase

   Bases: :py:obj:`geopyv.object.Object`

   Base class object initialiser.

   :param object_type: Object type.
   :type object_type: str


.. py:class:: ParticleResults(data)

   Bases: :py:obj:`ParticleBase`

   Initialisation of geopyv SequenceResults class.


