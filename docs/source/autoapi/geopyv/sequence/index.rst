:orphan:

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

   Initialisation of geopyv sequence object.

   :param img_sequence: 1D array of the image sequence of type `geopyv.Image` objects.
   :type img_sequence: numpy.ndarray (N)

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


   .. py:method:: particle(coords, vols)

      A method to propogate "particles" across the domain upon which strain path interpolation is performed.



.. py:function:: PolyArea(pts)

   A function that returns the area of the input polygon.

   :param pts: Clockwise/anti-clockwise ordered coordinates.
   :type pts: numpy.ndarray (N,2)


