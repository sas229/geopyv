:py:mod:`geopyv.particle`
=========================

.. py:module:: geopyv.particle


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.particle.Particle




.. py:class:: Particle(meshes, update_register=None, coord=np.zeros(2), p_init=np.zeros(12), vol=None)

   Particle class for geopyv.

   .. attribute:: coord

      1D array of the particle coordinates (x,y).

      :type: `numpy.ndarray` (2)

   .. attribute:: strain

      1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx).

      :type: `numpy.ndarray` (3)

   .. attribute:: vol

      Volume represented by the particle.

      :type: `float`

   .. attribute:: coord_ref

      1D array of the particle coordinates (x,y) at an updatable reference time.

      :type: `numpy.ndarray` (2)

   .. attribute:: strain_ref

      1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx) at an updatable reference time.

      :type: `numpy.ndarray` (3)

   .. attribute:: vol_ref

      Volume represented by the particle at an updatable reference time.

      :type: `float`

   .. py:method:: solve()

      Method to calculate the strain path of the particle from the mesh sequence and optionally the stress path
      employing the model specified by the input parameters.


   .. py:method:: _triangulation_locator(m)

      Method to locate the numerical particle within the mesh, returning the current element index.

      :param mesh: The relevant mesh.
      :type mesh: `geopyv.Mesh` object


   .. py:method:: _N_T(m, tri_idx)

      Private method to calculate the element shape functions for position and strain calculations.

      :param mesh: The relevant mesh.
      :type mesh: `geopyv.Mesh` object
      :param tri_idx: The index of the relevant element within mesh.
      :type tri_idx: `int`


   .. py:method:: _p_inc(m, tri_idx)


   .. py:method:: _strain_path()

      Method to calculate and store stress path data for the particle object.


   .. py:method:: _stress_path(model, statev, props)

      Method to calculate and store stress path data for the particle object. Input taken as compression negative.

      :param model: Identifies the constitutive model to implement.
      :type model: str
      :param statev:
                     - State environment variables relevant for the selected model.
      :type statev: numpy.ndarray(N)
      :param props:
                    - Material properties relevant for the selected model.
      :type props: numpy.ndarray(M)
      :param Configuration overview:
      :param Mohr Coulomb:
      :param - model = "MC":
      :param - statev = [sigma0_xx sigma0_yy sigma0_zz tau0_yz tau0_xz tau0_xy]:
      :param - statev = [E G nu sphi spsi cohs tens]:



