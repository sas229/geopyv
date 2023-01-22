:py:mod:`geopyv.sequence`
=========================

.. py:module:: geopyv.sequence


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.sequence.Sequence



Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.sequence.PolyArea



.. py:class:: Sequence(img_sequence, target_nodes=1000, boundary=None, exclusions=[])

   Sequence class for geopyv.

   .. attribute:: img_sequence

      1D array of the image sequence, `geopyv.Image` class objects.

      :type: `numpy.ndarray` (N)

   .. attribute:: SETUP PARAMETERS



   .. attribute:: meshes

      1D array of the mesh sequence, `geopyv.Mesh` class objects.

      :type: `numpy.ndarray` (N)

   .. attribute:: update_register

      1D array of the reference image indexes for the mesh sequence.

      :type: `numpy.ndarray` (N)s

   .. attribute:: ppp

      3D array of the numerical particle position paths (ppp).
      N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied), 2: (x,y) coordinates.
      Computed by method :meth:`~particle`.

      :type: `numpy.ndarray` (N,M,2)

   .. attribute:: pep

      3D array of the numerical particle strain paths (pep) (total strain).
      N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied), 3: (du/dx, dv/dy, du/dy+dv/dx).
      Computed by method :meth:`~particle`.

      :type: `numpy.ndarray` (N,M,3)

   .. attribute:: pvp

      2D array of the numerical particle volume paths (pvp).
      N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied).
      Computed by method :meth:`~particle`.

      :type: `numpy.ndarray` (N,M)

   .. py:method:: solve(seed_coord=None, template=Circle(50), max_iterations=15, max_norm=0.001, adaptive_iterations=0, method='ICGN', order=1, tolerance=0.7, alpha=0.5, beta=2, size_lower_bound=25, size_upper_bound=250)

      A method to generate a mesh sequence for the image sequence input at initiation. A reliability guided (RG) approach is implemented,
      updating the reference image according to correlation coefficient threshold criteria. An elemental shear strain-based mesh adaptivity is implemented.
      The meshes are stored in self.meshes and the mesh-image index references are stored in self.update_register.

      .. note::
              * For more details on the RG approach implemented, see:
                Stanier, S.A., Blaber, J., Take, W.A. and White, D.J. (2016) Improved image-based deformation measurment for geotechnical applications.
                Can. Geotech. J. 53:727-739 dx.doi.org/10.1139/cgj-2015-0253.
              * For more details on the adaptivity method implemented, see:
                Tapper, L. (2013) Bearing capacity of perforated offshore foundations under combined loading, University of Oxford PhD Thesis p.73-74.


   .. py:method:: particle(key='AUTO', coords=None, vols=None, opt=0)

      A method to generate strain paths using interpolation from the meshes to a distribution of numerical particles.

      :param key:
                  "AUTO" : Particle positions defined according to a kernel density estimation using all meshes in the sequence,
                          volumes defined using Voronoi method.
                          inp : None.
                  "MANUAL" : Particle positions defined by the user via "inp", volumes defined using Voronoi method.
                          inp : np.ndarray, (N,2).
                  "MESH" : Particle positions defined using a user-selected mesh via "inp", volumes defined using Voronoi method.
                          inp : geopyv.Mesh object.
      :type key: str
      :param opt: 0 - Combined kde particle distribution with size function 0:
                  1 - Combined kde particle distribution with size function 1:
                  2 - Combined kde particle distribution with size function 2:
                  3 - Combined kde particle distribution with size function 3:
      :type opt: int


   .. py:method:: _distribution(opt=None)

      Internal method for distributing particles across the region of interest (RoI).

      :param opt: Specifies the size function for the kde produced mesh:
                  0 - Combined kde particle distribution with size function 0:
                  1 - Combined kde particle distribution with size function 1:
                  2 - Combined kde particle distribution with size function 2:
                  3 - Combined kde particle distribution with size function 3:
      :type opt: int


   .. py:method:: _combination_mesh(opt)



.. py:function:: PolyArea(pts)

   A function that returns the area of the input polygon.

   :param pts: Clockwise/anti-clockwise ordered coordinates.
   :type pts: numpy.ndarray (N,2)


