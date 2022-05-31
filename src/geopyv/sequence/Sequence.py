import numpy as np
import scipy.stats as spst
import scipy.special as spsp
from geopyv.templates import Circle
from geopyv.image import Image
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv.gui import Gui
from geopyv.particle import Particle
import matplotlib.pyplot as plt
import gmsh
import cv2
# from ._subset_extensions import _init_reference, _solve_ICGN, _solve_FAGN, _solve_WFAGN

class Sequence:
    """Sequence class for geopyv.

    Attributes
    ----------
    img_sequence : `numpy.ndarray` (N) 
        1D array of the image sequence, `geopyv.Image` class objects.
    stps_num : `int`
        Number of images in the image sequence.
    SETUP PARAMETERS
    meshes : `numpy.ndarray` (N)
        1D array of the mesh sequence, `geopyv.Mesh` class objects.
    miref : `numpy.ndarray` (N)s
        1D array of the reference image indexes for the mesh sequence.
    ppp : `numpy.ndarray` (N,M,2)
        3D array of the numerical particle position paths (ppp). 
        N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied), 2: (x,y) coordinates.
        Computed by method :meth:`~particle`. 
    psp : `numpy.ndarray` (N,M,3)
        3D array of the numerical particle strain paths (psp) (total strain). 
        N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied), 3: (du/dx, dv/dy, du/dy+dv/dx).
        Computed by method :meth:`~particle`. 
    pvp : `numpy.ndarray` (N,M)
        2D array of the numerical particle volume paths (pvp). 
        N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied).
        Computed by method :meth:`~particle`. 
    """

    def __init__(self, img_sequence, aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)):
        """Initialisation of geopyv sequence object.

        Parameters
        ----------
        img_sequence : numpy.ndarray (N) 
            1D array of the image sequence of type `geopyv.Image` objects.
        """

        self.img_sequence = img_sequence
        self.ref = self.img_sequence[0]
        self.aruco_dict = aruco_dict
        self.stps_num = len(self.img_sequence)
        self.mesh_adaptivity_setup() # Default.
        self.mesh_piv_setup() # Default.

    def mesh_adaptivity_setup(self, max_iterations_adaptivity = 10, max_pts_num = 7500, alpha = 0.5, beta = 2, verbose = True):
        """A method to set mesh adaptivity input variables.
        
        Parameters
        ----------
        max_iterations_adaptivity : int
            Maximum number of adaptivity iterations to be performed.
        max_pts_num : int
            Maximum number of mesh nodes to be generated (note, exceeding this value triggers the end of adaptivity).
        alpha : float
            Mesh adaptivity element area weighting limit (1/beta^2 <= W <= 1/alpha^2, A_i,new = W*A_i,old).
        beta : float
            Mesh adaptivity element area weighting limit (1/beta^2 <= W <= 1/alpha^2, A_i,new = W*A_i,old).
        verbose : bool
            Toggle terminal ouput (False: no output, True: output). 
        """
        
        self.max_iterations_adaptivity = max_iterations_adaptivity
        self.max_pts_num = max_pts_num
        self.alpha = alpha
        self.beta = beta 
        self.verbose = verbose 
    
    def mesh_geometry_setup(self, area = None, roi = None, hls = None, obj = None, sed = None, manual = False):
        """A method to set mesh geometry input variables.
        
        Parameters
        ----------
        area : float
            Target element area pre-adaptivity.
        roi : numpy.ndarray (N,2)
            Coordinate array defining the edge of the mesh region.
        hls : numpy.ndarray (M,N,2)
            Exclusion regions within the roi. M hls, N, coordinates defining each. 
        sed : numpy.ndarray (2)
            Reliability-guided initial PIV coordinate specified in the far-field.
        manual : bool
            Toggle to manually select the roi, hls and sed (False: use input variables, True: use GUI).
        """
        
        self.manual = manual
        self.area = area
        self.roi = roi
        self.hls = hls
        self.obj = obj
        self.sed = sed
        if self.manual == True:
            self._manual()
            self.manual = False

    def _manual(self):
        """Private method to manually select the roi, hls and sed.
        """
        manual_selection = Gui(self.ref.filepath)
        self.roi, self.sed, self.obj, self.hls = manual_selection.main() # Run GUI and extract geometry data.

    def mesh_piv_setup(self, template = Circle(50), max_norm=1e-5, max_iterations_piv=50, p_0 = np.zeros(6), sed_tol = 0.9, tol = 0.75, method="ICGN"):
        """A method to set mesh PIV input variables.
        
        Parameters
        ----------
        template : object, optional
            Subset template. Defaults to Circle(50).
        max_norm : float, optional
            Exit criterion for norm of increment in warp function. Defaults to value of :math:`1 \cdot 10^{-5}`.
        max_iterations_piv: int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to value of 50.
        p_0: numpy.ndarray, optional
            1D array of warp function parameters with `float` type.
        sed_tol : float, optional
            Correlation coefficient threshold for sed subset (0<=sed_tol<=1). Defaults to value of 0.9.
        tol : float, optional
            Correlation coefficient thresholf for sed subset (0<=tol<=1). Defaults to value of 0.75.
        method : str, optional
            Solution method. Options are FAGN, WFAGN and ICGN. Default is ICGN since it is faster.

        .. note::
            * Any unspecified parameter will default in calling this method, even if specified in a previous call.  

        """
        self.template = template
        self.max_norm = max_norm
        self.max_iterations_piv = max_iterations_piv
        self.p_0 = p_0
        self.sed_tol = sed_tol
        self.tol = tol
        self.method = method
        
    def mesh(self, retain = False):
        """A method to generate a mesh sequence for the image sequence input at initiation. A reliability guided (RG) approach is implemented, 
        updating the reference image according to correlation coefficient threshold criteria. An elemental shear strain-based mesh adaptivity is implemented.
        The meshes are stored in self.meshes and the mesh-image index references are stored in self.miref. 

        .. note::
                * For more details on the RG approach implemented, see:
                  Stanier, S.A., Blaber, J., Take, W.A. and White, D.J. (2016) Improved image-based deformation measurment for geotechnical applications.
                  Can. Geotech. J. 53:727-739 dx.doi.org/10.1139/cgj-2015-0253.
                * For more details on the adaptivity method implemented, see:
                  Tapper, L. (2013) Bearing capacity of perforated offshore foundations under combined loading, University of Oxford PhD Thesis p.73-74.
        """
        
        ref = self.ref
        self.meshes = np.empty(self.stps_num-1, dtype=object) # Adapted meshes. 
        self.miref = np.zeros(self.stps_num-1, dtype=int) # Mesh-image reference.
        it = 1 # Initial target image index (note, initial reference image index is implicitly 0). 
        update_flag = True
        while it < self.stps_num:
            tar = self.img_sequence[it] # Select target image from the image sequence.
            if self.verbose == True:
                print("Image Pair: {}-{}".format(self.miref[-1], it)) # Print to terminal.
            mesh = Mesh(ref = ref, tar = tar, area = self.area, roi = self.roi, hls = self.hls, obj = self.obj, sed = self.sed, manual = self.manual) # Initialise mesh object.
            mesh.piv_setup(template = self.template, max_norm=self.max_norm, max_iterations_piv=self.max_iterations_piv, p_0 = self.p_0, sed_tol = self.sed_tol, tol = self.tol, method=self.method) # Update PIV parameters within the mesh object.
            mesh.adaptivity_setup(max_iterations_adaptivity = self.max_iterations_adaptivity, max_pts_num=self.max_pts_num, alpha=self.alpha, beta=self.beta, verbose=self.verbose) # Update adaptivity parameters within the mesh object.
            if update_flag == False and retain == True:
                mesh.pts = self.meshes[-1].pts
                mesh.tri = self.meshes[-1].tri
            mesh.strain_adapt(update_flag = update_flag) # Perform mesh adaptivity.
            if mesh.update: # Correlation coefficient thresholds not met (consequently no mesh generated).  
                update_flag = False
                if self.obj is not None:
                    self._obj_track(ref, self.img_sequence[it-1]) # Track object.
                ref = self.img_sequence[it-1] # Update reference image. 
                self.miref[it-1:] = it-1 # Update recorded reference image for future meshes.
            else:
                update_flag = True
                self.meshes[it-1] = mesh # Store the generated mesh.
                it += 1 # Iterate the target image index. 

    def _obj_track(self,ref,tar):
        """A private method which updates the object reference position by tracking a ChArUco marker.
        
        Parameters
        ----------
        ref :
            Reference image. 
        tar :
            Target image. 
        """
        r_c, r_id, r_rej = cv2.aruco.detectMarkers(ref.image_gs, self.aruco_dict)
        t_c, t_id, t_rej = cv2.aruco.detectMarkers(tar.image_gs, self.aruco_dict)
        r_c = r_c[0]
        t_c = t_c[0]
        r_id = r_id.flatten()
        t_id = t_id.flatten()
        for j in range(len(self.obj)):
            r_c_obj = []
            t_c_obj = []
            for i in range(len(r_c)):
                if path.Path(self.obj[j]).contains_points(r_c[i]).all() and (r_id[i] in t_id): # Marker within object and in both images.
                        r_c_obj.append(r_c[i])
                        t_c_obj.append(t_c[np.where(t_id==r_id[i])])
            r_c_obj = np.reshape(r_c_obj, (-1,2))
            t_c_obj = np.reshape(t_c_obj, (-1,2))
            self.obj[j] = affine(r_c_obj, t_c_obj, self.obj[j])
        
    def particle(self, key = 1, f = 0, par_pts = None, par_vols = None):
        """A method to generate strain paths using interpolation from the meshes to a distribution of numerical particles.
        
        Parameters
        ----------
        key : str
            Particles defined at the centroid of the element, 0, or the centroids of the incircle sub-divisions, 1. 
        coords : numpy.ndarray (N,2)
            User-specified initial numerical particle positions.
        """
        self.par_pts = par_pts
        self.par_vols = par_vols

        if key < 2:
            self.kde_dist(f = f)
            particles = np.empty(len(self.comb_mesh.tri)*3**key, dtype=object)
            self.ppp = np.empty((len(self.comb_mesh.tri)*3**key, self.stps_num, 2)) # Particle Position Path
            self.psp = np.zeros((len(self.comb_mesh.tri)*3**key, self.stps_num, 3)) # Particle Strain Path
            self.pvp = np.empty((len(self.comb_mesh.tri)*3**key, self.stps_num)) # Particle Volume Path
            self.ppp[:,0], self.pvp[:,0] = self.particle_distribution(mesh = self.comb_mesh, key = key) # Initial positions and volumes.
        else: 
            particles = np.empty(len(par_pts), dtype=object)
            self.ppp = np.empty((len(par_pts), self.stps_num, 2)) # Particle Position Path
            self.psp = np.zeros((len(par_pts), self.stps_num, 3)) # Particle Strain Path
            self.pvp = np.empty((len(par_pts), self.stps_num)) # Particle Volume Path
            self.ppp[:,0] = self.par_pts
            self.pvp[:,0] = self.par_vols
        for i in range(len(particles)): # Create matrix of particle objects.
            particles[i] = Particle(self.ppp[i,0], self.psp[i,0], self.pvp[i,0])
            for j in range(self.stps_num-1):
                ref_flag = False
                if self.miref[j-1] != self.miref[j] or i == 0:
                    ref_flag = True
                particles[i].update(self.meshes[j], ref_flag=ref_flag)
                self.ppp[i,j+1] = particles[i].coord
                self.psp[i,j+1] = particles[i].strain
                self.pvp[i,j+1] = particles[i].vol

    def particle_distribution(self, mesh, key = 0):
        """A method to distribute the numerical particles and calculate attributal volumes.
        
        Parameters
        ----------
        mesh : object
            The mesh to base the distribution of numerical particles upon.
        key : str
            Particles defined at the centroid of the element, "E", or the centroids of the incircle sub-divisions, "S". 
        coords : numpy.ndarray (N,2)
            User-specified initial numerical particle positions.
        """

        if key == 0:
            par_pts = np.mean(mesh.pts[mesh.tri], axis = 1)
            M = np.ones((len(mesh.tri),3,3))
            M[:,1] = mesh.pts[mesh.tri][:,:,0]
            M[:,2] = mesh.pts[mesh.tri][:,:,1]
            par_vols = abs(0.5*np.linalg.det(M))
        elif key == 1:
            # Find the element incentres
            BM = np.asarray([[0,1,-1],[-1,0,1],[1,-1,0]])
            MM = np.asarray([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])
            diff = np.einsum('ij,njk->nik', BM, mesh.pts[mesh.tri]) #BM@mesh.pts[mesh.tri]
            mid = np.einsum('ij,njk->nik', MM, mesh.pts[mesh.tri])
            dist = np.sqrt(np.einsum('nij,nij->ni', diff, diff))
            ctrs = np.einsum('ij,ijk->ik', dist, mesh.pts[mesh.tri])/np.einsum('ij->i', dist)[:,None]
            par_vols = np.zeros((len(mesh.tri),3))
            par_pts = np.empty((len(mesh.tri),3,2))
            for i in range(len(mesh.tri)):
                for j in range(3):
                    sub = np.asarray([ctrs[i], mid[i,j-2], mesh.pts[mesh.tri[i]][j], mid[i,j-1]])
                    par_vols[i,j] = PolyArea(sub)
                    par_pts[i,j] = np.mean(sub, axis=0)
            par_pts = par_pts.reshape(-1,2)
            par_vols = par_vols.flatten()
        return par_pts, par_vols
    
    def kde_dist(self, f = 0):
        """A method that combines the points through the mesh sequence to generate a Kernel Density Estimate (KDE)
        used as a background field to generate the numerical particle control mesh.

        f  : int
            Target element size function (f==0: 0.5*erfc(3.6*(Z_{bar}-0.5)), f==1: 100*10**(-4*Z_{bar})).
        """
        self.comb_mesh = Mesh(ref = self.img_sequence[0], tar = self.img_sequence[1], area = self.area, roi = self.meshes[0].init_roi, hls = self.meshes[0].hls, sed = self.meshes[0].sed) # Create mesh object with roi corresponding to the first image.
        self.comb_mesh.pts = self.comb_mesh.roi # Overwrite mesh points with roi.
        for i in range(len(self.meshes)):
            self.comb_mesh.pts = np.append(self.comb_mesh.pts, self.meshes[i].pts[len(self.comb_mesh.roi):], axis=0) # Append non-roi mesh points.
        self.comb_mesh.tri = np.reshape(gmsh.model.mesh.triangulate(self.comb_mesh.pts.flatten())-1, (-1,3)) # Define triangles for combined points.
        kernel = spst.gaussian_kde(self.comb_mesh.pts.T) # Create kde.
        Z = kernel(np.mean(self.comb_mesh.pts[self.comb_mesh.tri], axis=1).T) # Sample kde at element centroids.
        if f == 0:
            self.comb_mesh.areas = 0.5*spsp.erfc(3.6*((Z-np.min(Z))/(np.max(Z)-np.min(Z))-0.5))*self.area # Set target element areas.
        elif f == 1:
            self.comb_mesh.areas = self.area*10**(-2*(Z-np.min(Z))/(np.max(Z)-np.min(Z)))
        elif f == 2:
            self.comb_mesh.areas = self.area*(1-((Z-np.min(Z))/(np.max(Z)-np.min(Z))))
        elif f == 3:
            self.comb_mesh.areas = self.area*(1-((Z-np.min(Z))/(np.max(Z)-np.min(Z))))**2
        self.comb_mesh._bg_mesh_adapt() # Generate background field. 
        self.comb_mesh._mesh_extract() # Extract the numerical particle control mesh. 

def PolyArea(pts):
    """A function that returns the area of the input polygon.

    Parameters
    ----------
    pts : numpy.ndarray (N,2)
        Clockwise/anti-clockwise ordered coordinates.
    """

    x = pts[:,0]
    y = pts[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def affine(p,s, obj_pts):
    
    pad = lambda x: np.hstack([x, np.ones((x.shape[0],1))])
    unpad = lambda x: x[:,:-1]
    X = pad(p)
    Y = pad(s)
    A, res, rank, k = np.linalg.lstsq(X,Y, rcond = None)
    transform = lambda x: unpad(np.dot(pad(x),A))
    new_obj_pts = transform(obj_pts)
    print("Max error: {}".format(np.abs(s-transform(p)).max()))
    return new_obj_pts