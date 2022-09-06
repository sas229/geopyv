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

    def __init__(self, coord, meshes, update_register, strain = np.zeros(3), vol=None):
        """Initialisation of geopyv particle object.
        
        Parameters
        ----------
        coord : numpy.ndarray (2)
            1D array of the particle coordinates (x,y).
        strain : numpy.ndarray (3)
            1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx).
        vol : float
            Volume represented by the particle. 
        ref_coord_ref : numpy.ndarray (2)
            1D array of the particle coordinates (x,y) at an updatable reference time.
        ref_strain_ref : numpy.ndarray (3)
            1D array of the particle total strain (du/dx, dv/dy, du/dy+dv/dx) at an updatable reference time.
        ref_vol : float
            Volume represented by the particle at an updatable reference time. 
        """

        self.meshes = meshes
        self.length = len(meshes)
        self.update_register = update_register

        self.coords = np.empty((self.length+1, 2))
        self.strains = np.empty((self.length+1, 3))
        self.vols = np.empty(self.length+1)

        self.coords[0] = coord
        self.strains[0] = strain
        self.vols[0] = vol

        self.ref_coord = coord
        self.ref_strain = strain
        self.ref_vol = vol


    def solve(self):
        """Method to calculate the strain path of the particle from the mesh sequence and optionally the stress path
        employing the model specified by the input parameters."""

        self._strain_path()
        self._stress_path()


    def _triangulation_locator(self, m):
        """Method to locate the numerical particle within the mesh, returning the current element index.

        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        """

        diff = self.meshes[m].nodes - self.ref_coord # Particle-mesh node positional vector.
        dist = np.einsum('ij,ij->i',diff, diff) # Particle-mesh node "distances" (truly distance^2).
        tri_idxs = np.argwhere(np.any(self.meshes[m].triangulation == np.argmin(dist), axis=1)==True).flatten() # Retrieve relevant element indices.
        for i in range(len(tri_idxs)):  
            if path.Path(self.meshes[m].nodes[self.meshes[m].triangulation[tri_idxs[i]]]).contains_point(self.ref_coord): # Check if the element includes the particle coordinates.
                break # If the correct element is identified, stop the search. 
        return tri_idxs[i] # Return the element index. 

    def _shape(self, m, tri_idx):
        """Method to calculate the element shape functions for position and strain calculations.
        
        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        tri_idx: `int`
            The index of the relevant element within mesh."""

        self.B = self.meshes[m].Bs[tri_idx] # Retrieve the element B matrix from the mesh object (where epsilon = Bu).
        self.W = self._weight(self.meshes[m].nodes[self.meshes[m].triangulation[tri_idx]], self.meshes[m].areas[tri_idx]) # Calculate the element shape function matrix.

    def _weight(self, tri, area):
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
        W[:2] = (diff[:2,1]*(self.ref_coord[0]-tri[2,0])-diff[:2,0]*(self.ref_coord[1]-tri[2,1]))/(2*area) # Calculate nodal weightings (W1 and W2).
        W[2] -= W[0] + W[1] # Calculate nodal weightings (W3).
        return W
    
    def _strain_path(self):
        """Method to calculate and store stress path data for the particle object."""
        
        for m in range(self.length):
            tri_idx = self._triangulation_locator(m) # Identify the relevant element of the mesh.
            self._shape(m, tri_idx) # Calculate the B and W matrices.
            if self.update_register[m]: # Check whether to update the reference values.
                self.ref_coord = self.coords[m]
                self.ref_strain = self.strains[m]
                self.ref_vol = self.vols[m]
            self.coords[m+1] = self.ref_coord+self.W@self.meshes[m].p[self.meshes[m].triangulation[tri_idx], :2] # Update the particle positional coordinate (reference + mesh interpolation).
            self.strains[m+1] = self.ref_strain+self.B@(self.meshes[m].p[self.meshes[m].triangulation[tri_idx],:2].flatten()) # Update the particle strain (reference + mesh interpolation).
            self.vols[m+1] = self.ref_vol*(1 + self.strains[m+1,0] + self.strains[m+1,1]) # Update the particle volume (reference*(1 + volume altering strain components)).

    def _stress_path(self, **kwargs):
        """Method to calculate and store stress path data for the particle object.
        
        Parameters
        ----------
        model : str
            Identifies the constitutive model to implement:
                - "MC": Mohr-Coulomb.
                - "SMCC": Structurally modified Cam-Clay (according to Singh et al. (2021))."""

        pass
#
#        model = kwargs["model"] 
#        if model == "MC":
#            # Search for relevant variables and define defaults if non-specified. 
#            pass
#        elif model == "SMCC":
#            try: 
#                M = kwargs["M"]
#            except KeyError as error:
#                raise Exception("The slope of the CSL in q-p' space, M, must be provided.").with_traceback(error.__traceback__)
#            try: 
#                lam_star = kwargs["lam_star"]
#            except KeyError as error:
#                raise Exception("The slope of the NCL/ICL in ln(1+e)-ln(p') space, lam_star, must be provided.").with_traceback(error.__traceback__)
#            try: 
#                kap_star = kwargs["kap_star"]
#            except KeyError as error:
#                raise Exception("The slope of the URL in ln(1+e)-ln(p') space, kap_star, must be provided.").with_traceback(error.__traceback__)
#            try: 
#                N_star = kwargs["N_star"]
#            except KeyError as error:
#                raise Exception("The voids ratio at p' = 0 kPa, N_star, must be provided.").with_traceback(error.__traceback__)
#            try: 
#                nu = kwargs["nu"]
#            except KeyError as error:
#                raise Exception("Poisson's ratio, nu, must be provided.").with_traceback(error.__traceback__)
#            try: 
#                s_f = kwargs["s_f"]
#            except KeyError as error:
#                print("Warning: the reference sensitivity, s_f, has not been given. Proceeding with s_f = 1.")
#                s_f = 1
#            try:
#                k = kwargs["k"]
#            except KeyError as error:
#                print("Warning: the rate of structure degradation parameter, k, has not been given. Proceeding with k = 0.")
#                k = 0
#            try:
#                A = kwargs["A"]
#            except KeyError as error:
#                print("Warning: the deviatoric-volumetric strain proportion parameter, A, has not been given. Proceeding with A = 1.")
#                A = 1
#            try:
#                ocr = kwargs["ocr"]
#            except KeyError as error:
#                print("Warning: the initial overconsolidation ratio, ocr, has not been given. Proceeding with ocr = 1.")
#                ocr = 1
#            try: 
#                k_w = kwargs["k_w"]
#            except KeyError as error:
#                print("Warning: the pore water bulk modulus, k_w, has not been given. Proceeding with k_w = 1.2667 N/mm^2?.")
#                k_w = 1.2667
#            try: 
#                void = kwargs["void"]
#                p_c = np.exp((N_star-kap_star*np.log(p)-np.log(1+void))/(lam_star-kap_star))
#            except KeyError as error:
#                print("Warning: the initial void ratio, void, has not been given. Proceeding by estimating void ratio using p_c...")
#                try: 
#                    void = np.exp(N-kap_star*np.log(p))-(kap_star-lam_star)*np.log(p_c/p_r)
#                except NameError as error:
#                    try:
#
#                    print("Warning: the yield surface size has not been given.")









        