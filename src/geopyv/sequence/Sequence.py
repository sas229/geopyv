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
    SETUP PARAMETERS
    meshes : `numpy.ndarray` (N)
        1D array of the mesh sequence, `geopyv.Mesh` class objects.
    f_img_index : `numpy.ndarray` (N)s
        1D array of the reference image indexes for the mesh sequence.
    ppp : `numpy.ndarray` (N,M,2)
        3D array of the numerical particle position paths (ppp). 
        N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied), 2: (x,y) coordinates.
        Computed by method :meth:`~particle`. 
    pep : `numpy.ndarray` (N,M,3)
        3D array of the numerical particle strain paths (pep) (total strain). 
        N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied), 3: (du/dx, dv/dy, du/dy+dv/dx).
        Computed by method :meth:`~particle`. 
    pvp : `numpy.ndarray` (N,M)
        2D array of the numerical particle volume paths (pvp). 
        N: particle index, M: time step (M=0 initiation step, M=1 1st mesh applied).
        Computed by method :meth:`~particle`. 
    """

    def __init__(self, img_sequence, target_nodes=1000, boundary=None, exclusions=[]):
        """Initialisation of geopyv sequence object.

        Parameters
        ----------
        img_sequence : numpy.ndarray (N) 
            1D array of the image sequence of type `geopyv.Image` objects.
        """
        self.initialised = False
        # Check types.
        if type(img_sequence) != np.ndarray:
            raise TypeError("Image sequence array of invalid type. Cannot initialise sequence.")
        elif type(target_nodes) != int:
            raise TypeError("Maximum number of elements not of integer type.")
        elif target_nodes <= 0:
            raise ValueError("Invalid maximum number of elements.")
        elif type(boundary) != np.ndarray:
            raise TypeError("Boundary coordinate array of invalid type. Cannot initialise mesh.")
        elif type(exclusions) != list:
            raise TypeError("Exclusion coordinate array of invalid type. Cannot initialise mesh.")
        for img in img_sequence:
            if type(img) != Image:
                raise TypeError("Sequence image not geopyv.image.Image type.")
        for exclusion in exclusions:
            if np.shape(exclusion)[1] != 2:
                raise ValueError("Exclusion coordinate array of invalid shape. Must be numpy.ndarray of size (n, 2).")

        
        # Store variables.
        self.img_sequence = img_sequence
        self.target_nodes = target_nodes
        self.boundary = boundary
        self.exclusions = exclusions
        
    def solve(self, seed_coord=None, template=Circle(50), max_iterations=15, max_norm=1e-3, adaptive_iterations=0, method="ICGN", order=1, tolerance=0.7, alpha=0.5, beta=2, size_lower_bound = 25, size_upper_bound = 250):
        """A method to generate a mesh sequence for the image sequence input at initiation. A reliability guided (RG) approach is implemented, 
        updating the reference image according to correlation coefficient threshold criteria. An elemental shear strain-based mesh adaptivity is implemented.
        The meshes are stored in self.meshes and the mesh-image index references are stored in self.f_img_index. 

        .. note::
                * For more details on the RG approach implemented, see:
                  Stanier, S.A., Blaber, J., Take, W.A. and White, D.J. (2016) Improved image-based deformation measurment for geotechnical applications.
                  Can. Geotech. J. 53:727-739 dx.doi.org/10.1139/cgj-2015-0253.
                * For more details on the adaptivity method implemented, see:
                  Tapper, L. (2013) Bearing capacity of perforated offshore foundations under combined loading, University of Oxford PhD Thesis p.73-74.
        """

        # Check inputs.
        if type(seed_coord) != np.ndarray:
            raise TypeError("Coordinate is not of numpy.ndarray type. Cannot initiate solver.")
        elif type(adaptive_iterations) != int:
            raise TypeError("Number of adaptive iterations of invalid type. Must be an integer greater than or equal to zero.")
        
        # Store variables.
        self.seed_coord = seed_coord
        self.template = template
        self.max_iterations = max_iterations
        self.max_norm = max_norm
        self.adaptive_iterations = adaptive_iterations
        self.method = method
        self.order = order
        self.tolerance = tolerance
        self.alpha = alpha
        self.beta = beta
        self.size_lower_bound = size_lower_bound
        self.size_upper_bound = size_upper_bound
        if self.order == 1 and self.method != "WFAGN":
            self.p_0 = np.zeros(6)
        elif self.order == 1 and self.method == "WFAGN":
            self.p_0 = np.zeros(7)
        elif self.order == 2 and self.method != "WFAGN":
            self.p_0 = np.zeros(12)

        # Prepare output. 
        self.meshes = np.empty(len(self.img_sequence)-1, dtype=object) # Adapted meshes. 
        self.f_img_index = np.zeros(len(self.img_sequence)-1, dtype=int) # Mesh-image reference.

        # Solve. 
        iteration = 1 # Initial target image index (note, initial reference image index is implicitly 0). 
        update_flag = True
        while iteration < len(self.img_sequence):
            print("Solving for image pair {}-{}".format(self.f_img_index[iteration-1], iteration))
            mesh = Mesh(f_img = self.img_sequence[self.f_img_index[iteration-1]], g_img = self.img_sequence[iteration], target_nodes = self.target_nodes, boundary = self.boundary, exclusions = self.exclusions, size_lower_bound = self.size_lower_bound, size_upper_bound = self.size_upper_bound) # Initialise mesh object.
            mesh.solve(seed_coord=self.seed_coord, template=self.template, max_iterations=self.max_iterations, max_norm=self.max_norm, adaptive_iterations=self.adaptive_iterations, method=self.method, order=self.order, tolerance=self.tolerance, alpha=self.alpha, beta=self.beta) # Solve mesh.
            if mesh.update and update_flag: # Correlation coefficient thresholds not met (consequently no mesh generated).  
                update_flag = False 
                self.f_img_index[iteration-1:] = iteration-1 # Update recorded reference image for future meshes.
            else:
                update_flag = True
                self.meshes[iteration-1] = mesh # Store the generated mesh.
                iteration += 1 # Iterate the target image index. 
        
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
            particles = np.empty(len(self.comb_mesh.triangulation)*3**key, dtype=object)
            self.ppp = np.empty((len(self.comb_mesh.triangulation)*3**key, len(self.img_sequence), 2)) # Particle Position Path
            self.pep = np.zeros((len(self.comb_mesh.triangulation)*3**key, len(self.img_sequence), 3)) # Particle Strain Path
            self.pvp = np.empty((len(self.comb_mesh.triangulation)*3**key, len(self.img_sequence))) # Particle Volume Path
            self.ppp[:,0], self.pvp[:,0] = self.particle_distribution(mesh = self.comb_mesh, key = key) # Initial positions and volumes.
        else: 
            particles = np.empty(len(par_pts), dtype=object)
            self.ppp = np.empty((len(par_pts), len(self.img_sequence), 2)) # Particle Position Path
            self.pep = np.zeros((len(par_pts), len(self.img_sequence), 3)) # Particle Strain Path
            self.pvp = np.empty((len(par_pts), len(self.img_sequence))) # Particle Volume Path
            self.ppp[:,0] = self.par_pts
            self.pvp[:,0] = self.par_vols
        for i in range(len(particles)): # Create matrix of particle objects.
            particles[i] = Particle(self.ppp[i,0], self.pep[i,0], self.pvp[i,0])
            for j in range(len(self.img_sequence)-1):
                ref_flag = False
                if self.f_img_index[j-1] != self.f_img_index[j] or i == 0:
                    ref_flag = True
                particles[i].update(self.meshes[j], ref_flag=ref_flag)
                self.ppp[i,j+1] = particles[i].coord
                self.pep[i,j+1] = particles[i].strain
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
            par_pts = np.mean(mesh.nodes[mesh.triangulation], axis = 1)
            M = np.ones((len(mesh.triangulation),3,3))
            M[:,1] = mesh.nodes[mesh.triangulation][:,:,0]
            M[:,2] = mesh.nodes[mesh.triangulation][:,:,1]
            par_vols = abs(0.5*np.linalg.det(M))
        elif key == 1:
            # Find the element incentres
            BM = np.asarray([[0,1,-1],[-1,0,1],[1,-1,0]])
            MM = np.asarray([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])
            diff = np.einsum('ij,njk->nik', BM, mesh.nodes[mesh.triangulation]) #BM@mesh.nodes[mesh.triangulation]
            mid = np.einsum('ij,njk->nik', MM, mesh.nodes[mesh.triangulation])
            dist = np.sqrt(np.einsum('nij,nij->ni', diff, diff))
            ctrs = np.einsum('ij,ijk->ik', dist, mesh.nodes[mesh.triangulation])/np.einsum('ij->i', dist)[:,None]
            par_vols = np.zeros((len(mesh.triangulation),3))
            par_pts = np.empty((len(mesh.triangulation),3,2))
            for i in range(len(mesh.triangulation)):
                for j in range(3):
                    sub = np.asarray([ctrs[i], mid[i,j-2], mesh.nodes[mesh.triangulation[i]][j], mid[i,j-1]])
                    par_vols[i,j] = PolyArea(sub)
                    par_pts[i,j] = np.mean(sub, axis=0)
            par_pts = par_pts.reshape(-1,2)
            par_vols = par_vols.flatten()
        return par_pts, par_vols
    
    def kde_dist(self, f = 0, area= 200):
        """A method that combines the points through the mesh sequence to generate a Kernel Density Estimate (KDE)
        used as a background field to generate the numerical particle control mesh.

        f  : int
            Target element size function (f==0: 0.5*erfc(3.6*(Z_{bar}-0.5)), f==1: 100*10**(-4*Z_{bar})).
        """
        self.comb_mesh = Mesh(f_img = self.img_sequence[0], g_img = self.img_sequence[1], target_nodes=self.target_nodes, boundary = self.boundary, exclusions = self.exclusions) # Create mesh object with roi corresponding to the first image.
        self.comb_mesh.nodes = self.comb_mesh.boundary # Overwrite mesh points with roi.
        for i in range(len(self.meshes)):
            self.comb_mesh.nodes = np.append(self.comb_mesh.nodes, self.meshes[i].nodes[len(self.comb_mesh.boundary):], axis=0) # Append non-roi mesh points.
        self.comb_mesh.triangulation = np.reshape(gmsh.model.mesh.triangulate(self.comb_mesh.nodes.flatten())-1, (-1,3)) # Define triangles for combined points.
        kernel = spst.gaussian_kde(self.comb_mesh.nodes.T) # Create kde.
        Z = kernel(np.mean(self.comb_mesh.nodes[self.comb_mesh.triangulation], axis=1).T) # Sample kde at element centroids.
        if f == 0:
            self.comb_mesh.areas = 0.5*spsp.erfc(3.6*((Z-np.min(Z))/(np.max(Z)-np.min(Z))-0.5))*area # Set target element areas.
        elif f == 1:
            self.comb_mesh.areas = area*10**(-2*(Z-np.min(Z))/(np.max(Z)-np.min(Z)))
        elif f == 2:
            self.comb_mesh.areas = area*(1-((Z-np.min(Z))/(np.max(Z)-np.min(Z))))
        elif f == 3:
            self.comb_mesh.areas = area*(1-((Z-np.min(Z))/(np.max(Z)-np.min(Z))))**2
        self.comb_mesh._adaptive_remesh(scale=1.7, target=self.target_nodes, nodes = self.comb_mesh.nodes, triangulation = self.comb_mesh.triangulation, areas = self.comb_mesh.areas) # Generate background field. 
        self.comb_mesh._update_mesh() # Extract the numerical particle control mesh. 

#    def integrate(self, parms):
#        """Apply constitutive model via UMAT and integrate work along strain-path.
#        
#        parms : 
#            0 - M, slope of CSL.
#            1 - lamda, slope of NCL/ICL (ln(1+e):ln(p) plane).
#            2 - kappa, slope of URL (ln(1+e):ln(p) plane).
#            3 - N, ln(1+e) @ reference stress.
#            4 - nu, Poisson's ratio.
#            5 - s_f, Reference sensitivity.
#            6 - k, Rate of structure degradation.
#            7 - A, Deviatoric/volumetric strain balance.
#            8 - OCR, Initial overconsolidation ratio.
#            9 - k_w, Pore water bulk modulus.
#            10 - e, initial void ratio.
#            11 - s_ep, Initial value of sensitivity.
#            """
#
#        self.psp = np.zeros(self.pep.shape)
#
#        for p in range(len(particles)):
#            
#            
#        # Calculate initial stress
#
#
#
#            
#        #umat.umat()
#
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
