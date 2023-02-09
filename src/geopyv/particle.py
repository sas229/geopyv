"""

Particle module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re
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
    
    def plot(self, 
        quantity="warps", 
        component = 0,
        start_frame = None,
        end_frame = None,
        imshow = True,
        colorbar=True, 
        ticks=None,
        alpha=0.75,
        axis=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None
        ):

        
        if quantity is not None:
            fig,ax = gp.plots.plot_particle(
                data=self.data,
                quantity=quantity,
                component = component,
                start_frame = start_frame,
                end_frame = end_frame,
                imshow = imshow,
                colorbar=True,
                ticks=ticks,
                alpha=alpha,
                axis=axis,
                xlim=xlim,
                ylim=ylim,
                show=show,
                block=block,
                save=save
            )
            return fig, ax

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
        series = None,
        coordinate_0 = np.zeros(2),
        warp_0= np.zeros(12),
        vol_0 = 0.,
        moving = True
    ):
        """Initialisation of geopyv particle object.

        Parameters
        ----------
        meshes : `numpy.ndarray` of geopyv.mesh.Mesh objects
            Sequence for the particle object to track. 
        coordinate_0 : numpy.ndarray (2)
            Initial particle coordinate (x,y)
        p_init : `numpy.ndarray` (12), optional
            Initial warp vector.
        vol : float
            Volume represented by the particle.
        moving : bool
            Boolean for Lagrangian (False) or Eulerian (True) specification. Defaults to False.
        """

        self._initialised = False
        # Check types.
        if series.data["type"] != "Sequence" and series.data["type"] != "Mesh":
            log.error("Invalid series type. Must be gp.sequence.Sequence or gp.mesh.Mesh.")
        if type(coordinate_0) != np.ndarray:
            log.error("Invalid coordinate type. Must be numpy.ndarray.")
        elif np.shape(coordinate_0) != np.shape(np.empty(2)):
            log.error("Invalid coordinate shape. Must be (2).")
        if type(warp_0) != np.ndarray:
            log.error("Invalid initial warp type. Must be numpy.ndarray.")
        elif np.shape(warp_0) != np.shape(np.empty(12)):
            log.error("Invalid initial warp shape. Must be (12).")
        if type(vol_0) != float:
            log.error("Invalid initial volume type. Must be a float.")
        if type(moving) != bool:
            log.error("Invalid moving type. Must be a bool.")
            
        if series.data["type"] == "Sequence": 
            self._series_type = "Sequence"
            self._series = series.data["meshes"]
        else:
            self._series_type = "Mesh"
            self._series = np.asarray([series.data])
        self._moving = moving

        if self._series[0]["mask"][int(coordinate_0[0]), int(coordinate_0[1])] == 0:
            log.error("Particle coordinate lies outside the mesh.")
        
        
        self._coordinates = np.zeros((len(self._series)+ 1, 2))
        self._warps = np.zeros((len(self._series)+ 1, 12))
        self._volumes = np.zeros(len(self._series)+ 1)
        self._stresses = np.zeros((len(self._series)+ 1, 6))

        self._coordinates[0] = coordinate_0
        self._warps[0] = warp_0
        self._volumes[0] = vol_0

        self._reference_index = 0
        self.solved = False

        self._initialised = True
        self.data = {
            "type": "Particle",
            "solved": self.solved,
            "series_type": self._series_type,
            "moving" : self._moving,
            "coordinate_0" : self._coordinates[0],
            "warp_0": self._warps[0],
            "volume_0": self._volumes[0],
            "image_0": self._series[0]["images"]["f_img"]
        }
        #"series": self._series,
        

    def solve(self):  # model, statev, props):
        """

        Method to calculate the strain path of the particle from the
        mesh sequence and optionally the stress path employing the
        model specified by the input parameters.

        """

        self.solved += self._strain_path()
        #self.solved += self._stress_path(model, statev, props)
        self._results = {
            "coordinates" : self._coordinates,
            "warps" : self._warps,
            "volumes" : self._volumes,
            "stresses" : self._stresses
        }
        self.data.update({"results" : self._results})
        self.data["solved"] = bool(self.solved)
        return self.solved

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
            self._series[m]["nodes"] - self._coordinates[self._reference_index]
        )  # Particle-mesh node positional vector.
        dist = np.einsum(
            "ij,ij->i", diff, diff
        )  # Particle-mesh node "distances" (truly distance^2).
        tri_idxs = np.argwhere(np.any(self._series[m]["elements"] == np.argmin(dist), axis=1)).flatten()  # Retrieve relevant element indices.
        for i in range(len(tri_idxs)):
            if path.Path(
                self._series[m]["nodes"][self._series[m]["elements"][tri_idxs[i]]]
            ).contains_point(
                self._coordinates[self._reference_index]
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
        M[0, 1:] = self._coordinates[self._reference_index]
        M[1:, 1:] = self._series[m]["nodes"][self._series[m]["elements"][tri_idx]]
        area = self._series[m]["areas"][tri_idx]

        self.W = np.ones(3)
        self.W[0] = abs(np.linalg.det(M[[0, 2, 3]])) / (2 * abs(area))
        self.W[1] = abs(np.linalg.det(M[[0, 1, 3]])) / (2 * abs(area))
        self.W[2] -= self.W[0] + self.W[1]

    def _warp_increment(self, m, tri_idx):
        self._warp_inc = np.zeros(12)
        element = self._series[m]["nodes"][self._series[m]["elements"][tri_idx]]
        displacements = self._series[m]["results"]["displacements"][self._series[m]["elements"][tri_idx]]

        # Local coordinates
        A = np.ones((3, 4))
        A[1:, 0] = self._coordinates[self._reference_index]
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
        self._warp_inc[:2] = N @ displacements

        # 1st Order Strains
        J_x_T = dN @ element
        J_u_T = dN @ displacements
        self._warp_inc[2:6] = (np.linalg.inv(J_x_T) @ J_u_T).flatten()

        # 2nd Order Strains
        d2udzeta2 = d2N @ displacements
        J_zeta = np.zeros((2, 2))
        J_zeta[0, 0] = element[1, 1] - element[2, 1]
        J_zeta[0, 1] = element[2, 0] - element[1, 0]
        J_zeta[1, 0] = element[2, 1] - element[0, 1]
        J_zeta[1, 1] = element[0, 0] - element[2, 0]
        J_zeta /= np.linalg.det(A[:, [1, 2, 3]])
        self._warp_inc[6] = (
            d2udzeta2[0, 0] * J_zeta[0, 0] ** 2
            + 2 * d2udzeta2[1, 0] * J_zeta[0, 0] * J_zeta[1, 0]
            + d2udzeta2[2, 0] * J_zeta[1, 0] ** 2
        )
        self._warp_inc[7] = (
            d2udzeta2[0, 1] * J_zeta[0, 0] ** 2
            + 2 * d2udzeta2[1, 1] * J_zeta[0, 0] * J_zeta[1, 0]
            + d2udzeta2[2, 1] * J_zeta[1, 0] ** 2
        )
        self._warp_inc[8] = (
            d2udzeta2[0, 0] * J_zeta[0, 0] * J_zeta[0, 1]
            + d2udzeta2[1, 0]
            * (J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1])
            + d2udzeta2[2, 0] * J_zeta[1, 0] * J_zeta[1, 1]
        )
        self._warp_inc[9] = (
            d2udzeta2[0, 1] * J_zeta[0, 0] * J_zeta[0, 1]
            + d2udzeta2[1, 1]
            * (J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1])
            + d2udzeta2[2, 1] * J_zeta[1, 0] * J_zeta[1, 1]
        )
        self._warp_inc[10] = (
            d2udzeta2[0, 0] * J_zeta[0, 1] ** 2
            + 2 * d2udzeta2[1, 0] * J_zeta[0, 1] * J_zeta[1, 1]
            + d2udzeta2[2, 0] * J_zeta[1, 1] ** 2
        )
        self._warp_inc[11] = (
            d2udzeta2[0, 1] * J_zeta[0, 1] ** 2
            + 2 * d2udzeta2[1, 1] * J_zeta[0, 1] * J_zeta[1, 1]
            + d2udzeta2[2, 1] * J_zeta[1, 1] ** 2
        )

    def _strain_path(self):
        """Method to calculate and store stress path data for the particle object."""
        for m in range(len(self._series)):
            if  int(re.findall(r"\d+", self._series[self._reference_index]["images"]["f_img"])[-1]) != int(re.findall(r"\d+", self._series[m]["images"]["f_img"])[-1]):
                self._reference_index = m
            tri_idx = self._triangulation_locator(
                m
            )  # Identify the relevant element of the mesh.
            self._warp_increment(m, tri_idx)  # Calculate the nodal weightings.
            self._coordinates[m + 1] = (
                self._coordinates[self._reference_index] + self._warp_inc[:2]*int(self._moving)
            )  # Update the particle positional coordinate.
            # i.e. (reference + mesh interpolation).
            self._warps[m + 1] = self._warps[self._reference_index] + self._warp_inc
            self._volumes[m + 1] = self._volumes[self._reference_index] * (
                1 + self._warps[m + 1, 3] + self._warps[m + 1, 4]
            )  # Update the particle volume.
            # i.e. (reference*(1 + volume altering strain components)).
        return True

class ParticleResults(ParticleBase):
    """
    ParticleResults class for geopyv. 

    Parameters
    ----------
    data : dict
        geopyv data dict from Particle object.

    Attributes
    ----------
    data : dict
        geopyv data dict from Particle object.
    """

    def __init__(self, data):
        """Initialisation of geopyv SequenceResults class."""
        self.data = data
