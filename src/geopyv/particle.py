"""

Particle module for geopyv.

"""
import numpy as np
import matplotlib.path as path

log = logging.getLogger(__name__)

class ParticleBase(Object):
    """
    Particle base class to be used as a mixin. 
    """

    def __init__(self):
        super().__init__(object_type="Particle")
        """

        Particle base class initialiser

        """

class Particle(ParticleBase):
    """Particle class for geopyv.

    Parameters
    ----------
    
    
    
    vol : float
        Particle representative volume. 
    """

    def __init__(
        self, 
        *,
        meshes,
        update_register=None,
        coord=np.zeros(2),
        p_init=np.zeros(12),
        vol=None,
        fixed = False
    ):
        """Initialisation of geopyv particle object.

        Parameters
        ----------
        meshes : `numpy.ndarray` of geopyv.mesh.Mesh objects
            Sequence for the particle object to track. 
        update_register : `numpy.ndarray` (N)
            Record of reference image updates.
        coord : numpy.ndarray (2)
            Initial particle coordinate (x,y)
        p_init : `numpy.ndarray` (12), optional
            Initial warp vector.
        vol : float
            Volume represented by the particle.
        fixed : bool
            Boolean for Lagrangian (False) or Eulerian (True) specification. Defaults to False.
        """

        self.meshes = meshes
        self.length = len(meshes)
        if update_register is None:
            self.update_register = np.zeros(self.length)
        else:
            self.update_register = update_register
        self.coords = np.zeros((self.length + 1, 2))
        self.ps = np.zeros((self.length + 1, 12))
        self.vols = np.zeros(self.length + 1)
        self.stress_path = np.zeros((self.length + 1, 6))

        self.coords[0] = coord
        self.ps[0] = p_init
        self.vols[0] = vol

        self.ref_coord = coord
        self.ref_p = p_init
        self.ref_vol = vol

    def solve(self):  # model, statev, props):
        """

        Method to calculate the strain path of the particle from the
        mesh sequence and optionally the stress path employing the
        model specified by the input parameters.

        """
        self._strain_path()
        # self._stress_path(model, statev, props)

    def _triangulation_locator(self, m):
        """

        Method to locate the numerical particle within the mesh,
        returning the current element index.


        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.

        """
        diff = (
            self.meshes[m].nodes - self.ref_coord[:2]
        )  # Particle-mesh node positional vector.
        dist = np.einsum(
            "ij,ij->i", diff, diff
        )  # Particle-mesh node "distances" (truly distance^2).
        tri_idxs = np.argwhere(
            np.any(self.meshes[m].elements == np.argmin(dist), axis=1) is True
        ).flatten()  # Retrieve relevant element indices.
        for i in range(len(tri_idxs)):
            if path.Path(
                self.meshes[m].nodes[self.meshes[m].elements[tri_idxs[i]]]
            ).contains_point(
                self.ref_coord[:2]
            ):  # Check if the element includes the particle coordinates.
                break  # If the correct element is identified, stop the search.
        return tri_idxs[i]  # Return the element index.

    def _N_T(self, m, tri_idx):
        """

        Private method to calculate the element shape functions for
        position and strain calculations.


        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        tri_idx: `int`
            The index of the relevant element within mesh.

        """
        M = np.ones((4, 3))
        M[0, 1:] = self.ref_coord
        M[1:, 1:] = self.meshes[m].nodes[self.meshes[m].elements[tri_idx]]
        area = self.meshes[m].areas[tri_idx]

        self.W = np.ones(3)
        self.W[0] = abs(np.linalg.det(M[[0, 2, 3]])) / (2 * abs(area))
        self.W[1] = abs(np.linalg.det(M[[0, 1, 3]])) / (2 * abs(area))
        self.W[2] -= self.W[0] + self.W[1]

    def _p_inc(self, m, tri_idx):
        self.p_inc = np.zeros(12)
        element = self.meshes[m].nodes[self.meshes[m].elements[tri_idx]]
        displacements = self.meshes[m].displacements[self.meshes[m].elements[tri_idx]]

        # Local coordinates
        A = np.ones((3, 4))
        A[1:, 0] = self.ref_coord
        A[1:, 1:] = element[:3, :2].transpose()
        zeta = np.linalg.det(A[:, [0, 2, 3]]) / np.linalg.det(A[:, [1, 2, 3]])
        eta = np.linalg.det(A[:, [0, 3, 1]]) / np.linalg.det(A[:, [1, 2, 3]])
        theta = 1 - zeta - eta

        # Weighting function (and derivatives to 2nd order)
        N = np.asarray(
            [
                zeta * (2 * zeta - 1),
                eta * (2 * eta - 1),
                theta * (2 * theta - 1),
                4 * zeta * eta,
                4 * eta * theta,
                4 * theta * zeta,
            ]
        )
        dN = np.asarray(
            [
                [4 * zeta - 1, 0, 1 - 4 * theta, 4 * eta, -4 * eta, 4 * (theta - zeta)],
                [0, 4 * eta - 1, 1 - 4 * theta, 4 * zeta, 4 * (theta - eta), -4 * zeta],
            ]
        )
        d2N = np.asarray(
            [[4, 0, 4, 0, 0, -8], [0, 0, 4, 4, -4, -4], [0, 4, 4, 0, -8, 0]]
        )

        # Displacements
        self.p_inc[:2] = N @ displacements

        # 1st Order Strains
        J_x_T = dN @ element
        J_u_T = dN @ displacements
        self.p_inc[2:6] = (np.linalg.inv(J_x_T) @ J_u_T).flatten()

        # 2nd Order Strains
        d2udzeta2 = d2N @ displacements
        J_zeta = np.zeros((2, 2))
        J_zeta[0, 0] = element[1, 1] - element[2, 1]
        J_zeta[0, 1] = element[2, 0] - element[1, 0]
        J_zeta[1, 0] = element[2, 1] - element[0, 1]
        J_zeta[1, 1] = element[0, 0] - element[2, 0]
        J_zeta /= np.linalg.det(A[:, [1, 2, 3]])
        self.p_inc[6] = (
            d2udzeta2[0, 0] * J_zeta[0, 0] ** 2
            + 2 * d2udzeta2[1, 0] * J_zeta[0, 0] * J_zeta[1, 0]
            + d2udzeta2[2, 0] * J_zeta[1, 0] ** 2
        )
        self.p_inc[7] = (
            d2udzeta2[0, 1] * J_zeta[0, 0] ** 2
            + 2 * d2udzeta2[1, 1] * J_zeta[0, 0] * J_zeta[1, 0]
            + d2udzeta2[2, 1] * J_zeta[1, 0] ** 2
        )
        self.p_inc[8] = (
            d2udzeta2[0, 0] * J_zeta[0, 0] * J_zeta[0, 1]
            + d2udzeta2[1, 0]
            * (J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1])
            + d2udzeta2[2, 0] * J_zeta[1, 0] * J_zeta[1, 1]
        )
        self.p_inc[9] = (
            d2udzeta2[0, 1] * J_zeta[0, 0] * J_zeta[0, 1]
            + d2udzeta2[1, 1]
            * (J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1])
            + d2udzeta2[2, 1] * J_zeta[1, 0] * J_zeta[1, 1]
        )
        self.p_inc[10] = (
            d2udzeta2[0, 0] * J_zeta[0, 1] ** 2
            + 2 * d2udzeta2[1, 0] * J_zeta[0, 1] * J_zeta[1, 1]
            + d2udzeta2[2, 0] * J_zeta[1, 1] ** 2
        )
        self.p_inc[11] = (
            d2udzeta2[0, 1] * J_zeta[0, 1] ** 2
            + 2 * d2udzeta2[1, 1] * J_zeta[0, 1] * J_zeta[1, 1]
            + d2udzeta2[2, 1] * J_zeta[1, 1] ** 2
        )

    def _strain_path(self):
        """Method to calculate and store stress path data for the particle object."""
        for m in range(self.length):
            if self.update_register[m]:  # Check whether to update the reference values.
                self.ref_coords = self.coords[m]
                self.ref_p = self.ps[m]
                self.ref_vol = self.vols[m]
            tri_idx = self._triangulation_locator(
                m
            )  # Identify the relevant element of the mesh.
            self._p_inc(m, tri_idx)  # Calculate the nodal weightings.
            if not self._fixed:
                self.coords[m + 1] = (
                    self.ref_coord + self.p_inc[:2]
                )  # Update the particle positional coordinate.
            # i.e. (reference + mesh interpolation).
            self.ps[m + 1] = self.ref_p + self.p_inc
            self.vols[m + 1] = self.ref_vol * (
                1 + self.ps[m + 1, 3] + self.ps[m + 1, 4]
            )  # Update the particle volume.
            # i.e. (reference*(1 + volume altering strain components)).
