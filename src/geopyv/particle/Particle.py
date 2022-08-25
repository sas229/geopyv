import numpy as np
import matplotlib.path as path
#import umat

class Particle:
    """Particle class for geopyv.

    Attributes
    ----------
    coord : `numpy.ndarray` (2)
        1D array of the particle coordinates (x,y).
    strain : `numpy.ndarray` (3)
        1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx).
    vol : `float`
        Volume represented by the particle. 
    coord_ref : `numpy.ndarray` (2)
        1D array of the particle coordinates (x,y) at an updatable reference time.
    strain_ref : `numpy.ndarray` (3)
        1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx) at an updatable reference time.
    vol_ref : `float`
        Volume represented by the particle at an updatable reference time. 
    """

    def __init__(self, coord, strain = np.zeros(3),vol=None):
        """Initialisation of geopyv particle object.
        
        Parameters
        ----------
        coord : numpy.ndarray (2)
            1D array of the particle coordinates (x,y).
        strain : numpy.ndarray (3)
            1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx).
        vol : float
            Volume represented by the particle. 
        coord_ref : numpy.ndarray (2)
            1D array of the particle coordinates (x,y) at an updatable reference time.
        strain_ref : numpy.ndarray (3)
            1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx) at an updatable reference time.
        vol_ref : float
            Volume represented by the particle at an updatable reference time. 
        """

        self.coord = coord
        self.strain = strain
        self.vol = vol
        self.coord_ref = coord
        self.strain_ref = strain
        self.vol_ref = vol

    def triloc(self, mesh):
        """Method to locate the numerical particle within the mesh, returning the current element index.

        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        """

        diff = mesh.nodes - self.coord # Particle-mesh node positional vector.
        dist = np.einsum('ij,ij->i',diff, diff) # Particle-mesh node "distances" (truly distance^2).
        tri_idxs = np.argwhere(np.any(mesh.triangulation == np.argmin(dist), axis=1)==True).flatten() # Retrieve relevant element indices.
        for i in range(len(tri_idxs)):  
            if path.Path(mesh.nodes[mesh.triangulation[tri_idxs[i]]]).contains_point(self.coord): # Check if the element includes the particle coordinates.
                break # If the correct element is identified, stop the search. 
        return tri_idxs[i] # Return the element index. 

    def corloc(self, mesh, tri_idx):
        """Method to calculate the element shape functions for position and strain calculations.
        
        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        tri_idx: `int`
            The index of the relevant element within mesh."""

        self.B = mesh.Bs[tri_idx] # Retrieve the element B matrix from the mesh object (where epsilon = Bu).
        self.W = self._W(mesh.nodes[mesh.triangulation[tri_idx]], mesh.areas[tri_idx]) # Calculate the element shape function matrix.

    def _W(self, tri, area):
        """Private method to calculate the element shape functions.
        
        Parameters
        ----------
        tri : numpy.ndarray (N,2)
            Clockwise/anti-clockwise ordered coordinates for the mesh element.
        area : float
            The area of the mesh element.
        """
        
        WM = np.asarray([[0,1,-1],[-1,0,1],[1,-1,0]]) # Matrix multiplier.
        diff = WM@tri # Create [[x2-x3, y2-y3],[x3-x1, y3-y1],[x1-x2, y1-y2]] matrix. 
        W = np.ones(3) # Initiate nodal weighting vector.
        W[:2] = (diff[:2,1]*(self.coord[0]-tri[2,0])-diff[:2,0]*(self.coord[1]-tri[2,1]))/(2*area) # Calculate nodal weightings (W1 and W2).
        W[2] -= W[0] + W[1] # Calculate nodal weightings (W3).
        return W
    
    def update(self, mesh, ref_flag = False):
        """Method to update the particle using the mesh.
        
        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        ref_flag : `bool`
            Indicates whether the reference image has been updated for this mesh. True: "reference" attributes are updated, False: nothing. 
        """
        
        tri_idx = self.triloc(mesh) # Identify the relevant element of the mesh.
        self.corloc(mesh, tri_idx) # Calculate the B and W matrices.
        if ref_flag: # Check whether to update the reference values.
            self.coord_ref = self.coord
            self.strain_ref = self.strain
            self.vol_ref = self.vol
        self.coord = self.coord_ref+self.W@mesh.p[mesh.triangulation[tri_idx], :2] # Update the particle positional coordinate (reference + mesh interpolation).
        self.strain = self.strain_ref+self.B@(mesh.p[mesh.triangulation[tri_idx],:2].flatten()) # Update the particle strain (reference + mesh interpolation).
        self.vol = self.vol_ref*(1 + self.strain[0] + self.strain[1]) # Update the particle volume (reference*(1 + volume altering strain components)).






        