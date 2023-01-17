import numpy as np
import matplotlib.path as path
# from geopyv.umats import umat_mc

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

    def __init__(self, meshes, update_register=None, coord = np.zeros(2), p_init = np.zeros(12), vol=None):
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
        if update_register is None:
            self.update_register = np.zeros(self.length)
        else:
            self.update_register = update_register
        self.coords = np.zeros((self.length+1,2))
        self.ps = np.zeros((self.length+1, 12))
        self.vols = np.zeros(self.length+1)
        self.stress_path = np.zeros((self.length+1,6))

        self.coords[0] = coord
        self.ps[0] = p_init
        self.vols[0] = vol

        self.ref_coord = coord
        self.ref_p = p_init
        self.ref_vol = vol

    def solve(self): #model, statev, props):
        """Method to calculate the strain path of the particle from the mesh sequence and optionally the stress path
        employing the model specified by the input parameters."""

        self._strain_path()
        #self._stress_path(model, statev, props)

    def _triangulation_locator(self, m):
        """Method to locate the numerical particle within the mesh, returning the current element index.

        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        """

        diff = self.meshes[m].nodes - self.ref_coord[:2] # Particle-mesh node positional vector.
        dist = np.einsum('ij,ij->i',diff, diff) # Particle-mesh node "distances" (truly distance^2).
        tri_idxs = np.argwhere(np.any(self.meshes[m].elements == np.argmin(dist), axis=1)==True).flatten() # Retrieve relevant element indices.
        for i in range(len(tri_idxs)):  
            if path.Path(self.meshes[m].nodes[self.meshes[m].elements[tri_idxs[i]]]).contains_point(self.ref_coord[:2]): # Check if the element includes the particle coordinates.
                break # If the correct element is identified, stop the search. 
        return tri_idxs[i] # Return the element index. 

    def _N_T(self, m, tri_idx):
        """Private method to calculate the element shape functions for position and strain calculations.
        
        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.
        tri_idx: `int`
            The index of the relevant element within mesh."""
        
        M =np.ones((4,3))
        M[0,1:] = self.ref_coord
        M[1:,1:] = self.meshes[m].nodes[self.meshes[m].elements[tri_idx]]
        area = self.meshes[m].areas[tri_idx]

        self.W = np.ones(3)
        self.W[0] = abs(np.linalg.det(M[[0,2,3]]))/(2*abs(area))
        self.W[1] = abs(np.linalg.det(M[[0,1,3]]))/(2*abs(area))
        self.W[2] -= self.W[0]+self.W[1]

    def _p_inc(self, m, tri_idx):
        
        self.p_inc = np.zeros(12)
        element = self.meshes[m].nodes[self.meshes[m].elements[tri_idx]]
        displacements = self.meshes[m].displacements[self.meshes[m].elements[tri_idx]]

        # Local coordinates
        A = np.ones((3,4))
        A[1:,0] = self.ref_coord
        A[1:,1:] = element[:3,:2].transpose()
        zeta = np.linalg.det(A[:,[0,2,3]])/np.linalg.det(A[:,[1,2,3]])
        eta  = np.linalg.det(A[:,[0,3,1]])/np.linalg.det(A[:,[1,2,3]])
        theta = 1-zeta-eta

        # Weighting function (and derivatives to 2nd order)
        N = np.asarray([zeta*(2*zeta-1), eta*(2*eta-1), theta*(2*theta-1), 4*zeta*eta, 4*eta*theta, 4*theta*zeta])
        dN = np.asarray([[4*zeta-1, 0, 1-4*theta, 4*eta, -4*eta, 4*(theta-zeta)],
                            [0, 4*eta-1, 1-4*theta, 4*zeta, 4*(theta-eta), -4*zeta]])
        d2N = np.asarray([[4,0,4,0,0,-8],
                            [0,0,4,4,-4,-4],
                            [0,4,4,0,-8,0]])

        # Displacements
        self.p_inc[:2] = N@self.meshes[m].nodes[self.meshes[m].elements[tri_idx]]

        # 1st Order Strains
        J_x_T = dN@element
        J_u_T = dN@displacements
        self.p_inc[2:6] = (np.linalg.inv(J_x_T)@J_u_T).flatten()

        # 2nd Order Strains
        d2udzeta2 = d2N@displacements  
        J_zeta = np.zeros((2,2))
        J_zeta[0,0] = element[1,1]-element[2,1]
        J_zeta[0,1] = element[2,0]-element[1,0]
        J_zeta[1,0] = element[2,1]-element[0,1]
        J_zeta[1,1] = element[0,0]-element[2,0]
        J_zeta /= np.linalg.det(A[:,[1,2,3]])
        self.p_inc[6] = d2udzeta2[0,0]*J_zeta[0,0]**2+2*d2udzeta2[1,0]*J_zeta[0,0]*J_zeta[1,0]+d2udzeta2[2,0]*J_zeta[1,0]**2
        self.p_inc[7] = d2udzeta2[0,1]*J_zeta[0,0]**2+2*d2udzeta2[1,1]*J_zeta[0,0]*J_zeta[1,0]+d2udzeta2[2,1]*J_zeta[1,0]**2
        self.p_inc[8] = d2udzeta2[0,0]*J_zeta[0,0]*J_zeta[0,1]+d2udzeta2[1,0]*(J_zeta[0,0]*J_zeta[1,1]+J_zeta[1,0]*J_zeta[0,1])+d2udzeta2[2,0]*J_zeta[1,0]*J_zeta[1,1]
        self.p_inc[9] = d2udzeta2[0,1]*J_zeta[0,0]*J_zeta[0,1]+d2udzeta2[1,1]*(J_zeta[0,0]*J_zeta[1,1]+J_zeta[1,0]*J_zeta[0,1])+d2udzeta2[2,1]*J_zeta[1,0]*J_zeta[1,1]
        self.p_inc[10] = d2udzeta2[0,0]*J_zeta[0,1]**2+2*d2udzeta2[1,0]*J_zeta[0,1]*J_zeta[1,1]+d2udzeta2[2,0]*J_zeta[1,1]**2
        self.p_inc[11] = d2udzeta2[0,1]*J_zeta[0,1]**2+2*d2udzeta2[1,1]*J_zeta[0,1]*J_zeta[1,1]+d2udzeta2[2,1]*J_zeta[1,1]**2

    def _strain_path(self):
        """Method to calculate and store stress path data for the particle object."""
        print("Strain path")
        for m in range(self.length):
            if self.update_register[m]: # Check whether to update the reference values.
                self.ref_coords = self.coords[m]
                self.ref_p = self.ps[m]
                self.ref_vol = self.vols[m]
            tri_idx = self._triangulation_locator(m) # Identify the relevant element of the mesh.
            self._p_inc(m, tri_idx) # Calculate the nodal weightings.
            self.coords[m+1] = self.ref_coord+self.p_inc[:2] # Update the particle positional coordinate (reference + mesh interpolation).
            self.ps[m+1] = self.ref_p + self.p_inc
            self.vols[m+1] = self.ref_vol*(1 + self.ps[m+1,3] + self.ps[m+1,4]) # Update the particle volume (reference*(1 + volume altering strain components)).

    def _stress_path(self, model, statev, props):
        """Method to calculate and store stress path data for the particle object. Input taken as compression negative. 
        
        Parameters
        ----------
        model : str
            Identifies the constitutive model to implement.
        statev : numpy.ndarray(N)
            - State environment variables relevant for the selected model.
        props : numpy.ndarray(M)
            - Material properties relevant for the selected model. 

        Configuration overview:
        Mohr Coulomb:
        - model = "MC"
        - statev = [sigma0_xx sigma0_yy sigma0_zz tau0_yz tau0_xz tau0_xy]
        - statev = [E G nu sphi spsi cohs tens]"""

        model_list = ["MC"]
        if model not in model_list:
            raise ValueError("ValueError: constitutive model mis-named or unsupported. Ensure the model given is in: {}".format(model_list))
        else: 
            self.model = model
        
        if self.model == "MC":
            self.stress_path = statev
            self.props = props
            nstatev = len(statev)
            nprops = len(props)
            ddsdde = np.asarray([[1/props[0], -props[2]/props[0], -props[2]/props[0], 0, 0, 0],
                                [-props[2]/props[0], 1/props[0], -props[2]/props[0], 0, 0, 0],
                                [-props[2]/props[0], -props[2]/props[0], 1/props[0], 0, 0, 0],
                                [0, 0, 0, 1/props[1], 0, 0],
                                [0, 0, 0, 0, 1/props[1], 0],
                                [0, 0, 0, 0, 0, 1/props[1]]])
            for i in range(self.length):
                stran = self.ps[i, 2:]
                dstran = self.ps[i+1, 2:] - self.ps[i, 2:]
                coords = self.coords[i]
                stress = self.stress_path[i]
                # umat_mc.umat(stress = stress, statev=np.zeros(6), ddsdde = ddsdde, sse = 0, 
                #                 spd = 0, scd = 0, rpl = 0, ddsddt = np.zeros(6), drplde = np.zeros(6), drpldt = 0, 
                #                 stran = stran, dstran= dstran, time = np.zeros(2), dtime = 0, temp = 0, dtemp = 0,
                #                 predef = np.zeros(1), dpred = np.zeros(1), cmname = 0, ndi = 0, nshr = 0, ntens = 6, 
                #                 nstatev=nstatev, props=props, nprops=nprops, coords = coords, drot = np.zeros((3,3)), pnewdt = 0,
                #                 celent = 0, dfgrd0 = np.zeros((3,3)), dfgrd1 = np.zeros((3,3)), noel = 0, npt = 0, layer = 0, 
                #                 kspt = 0, kstep = 0, kinc = 0)
                self.stress_path[i+1] = stress


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
