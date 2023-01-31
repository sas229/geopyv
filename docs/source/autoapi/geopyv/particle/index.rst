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




.. py:class:: Particle(meshes, update_register=None, coord=np.zeros(2), p_init=np.zeros(12), vol=None)

   Initialisation of geopyv particle object.

   :param coord: 1D array of the particle coordinates (x,y).
   :type coord: numpy.ndarray (2)
   :param strain: 1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx).
   :type strain: numpy.ndarray (3)
   :param vol: Volume represented by the particle.
   :type vol: float
   :param ref_coord_ref: 1D array of the particle coordinates (x,y) at an updatable reference time.
   :type ref_coord_ref: numpy.ndarray (2)
   :param ref_strain_ref: 1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx) at an updatable reference time.
   :type ref_strain_ref: numpy.ndarray (3)
   :param ref_vol: Volume represented by the particle at an updatable reference time.
   :type ref_vol: float

   .. py:method:: solve()

      Method to calculate the strain path of the particle from the mesh sequence and optionally the stress path
      employing the model specified by the input parameters.
