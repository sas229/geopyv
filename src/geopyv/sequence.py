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
    update_register : `numpy.ndarray` (N)s
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
        The meshes are stored in self.meshes and the mesh-image index references are stored in self.update_register. 

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
        self.update_register = np.zeros(len(self.img_sequence)-1, dtype=int) # Mesh-image reference.

        # Solve. 
        f_index = 0
        g_index = 1
        while g_index < len(self.img_sequence):
            print("Solving for image pair {}-{}".format(f_index, g_index))
            mesh = Mesh(f_img = self.img_sequence[f_index], g_img = self.img_sequence[g_index], target_nodes = self.target_nodes, boundary = self.boundary, exclusions = self.exclusions, size_lower_bound = self.size_lower_bound, size_upper_bound = self.size_upper_bound) # Initialise mesh object.
            mesh.solve(seed_coord=self.seed_coord, template=self.template, max_iterations=self.max_iterations, max_norm=self.max_norm, adaptive_iterations=self.adaptive_iterations, method=self.method, order=self.order, tolerance=self.tolerance, alpha=self.alpha, beta=self.beta) # Solve mesh.
            if mesh.update and self.update_register[g_index-1] == 0: # Correlation coefficient thresholds not met (consequently no mesh generated).  
                f_index = g_index - 1
                self.update_register[f_index] = 1 # Update recorded reference image for future meshes.
            else:
                self.meshes[g_index-1] = mesh # Store the generated mesh.
                g_index += 1 # Iterate the target image index. 
        
    def particle(self, key = "AUTO", coords = None, vols = None, opt = 0): #key = 1, coord_opt = 0, par_pts = None, par_vols = None):
        """A method to generate strain paths using interpolation from the meshes to a distribution of numerical particles.
        
        Parameters
        ----------
        key : str
            "AUTO" : Particle positions defined according to a kernel density estimation using all meshes in the sequence,
                    volumes defined using Voronoi method.
                    inp : None.
            "MANUAL" : Particle positions defined by the user via "inp", volumes defined using Voronoi method.
                    inp : np.ndarray, (N,2). 
            "MESH" : Particle positions defined using a user-selected mesh via "inp", volumes defined using Voronoi method.
                    inp : geopyv.Mesh object. 
        opt : int
            0 - Combined kde particle distribution with size function 0:
            1 - Combined kde particle distribution with size function 1:
            2 - Combined kde particle distribution with size function 2:
            3 - Combined kde particle distribution with size function 3:

        """

        if key == "AUTO":
            if type(opt) != int:
                raise TypeError("opt must be an integer.")
            elif opt > 3 or opt < 0:
                raise ValueError("opt must be 0, 1, 2, or 3.")
            init_coords, init_vols = _distribution(inp = opt)
        elif key == "MANUAL":
            if type(coords) != np.ndarray:
                raise TypeError("Expected np.ndarray, instead recieved {} for coords.".format(type(input)))
            elif type(vols) != np.ndarray:
                raise TypeError("Expected np.ndarray, instead recieved {} for vols.".format(type(input)))
            elif coords.shape[1] != 2:
                raise ValueError("Expected an np.ndarray of shape (N,2) for coords, instead recieved np.ndarray of shape {}.".format(input.shape))
            elif len(coords) != len(vols):
                raise ValueError("Coordinate and volume lengths mismatched.")
            init_coords = coords
            init_vols = vols
        else:
            raise ValueError("Distribution key not recognised.")
    
        self.particles = np.empty(len(init_vols), dtype=object)
        for i in range(len(self.particles)):
            self.particles[i] = Particle(coord = init_coords[i], meshes = self.meshes, update_register = self.update_register, vol = init_vols[i])
            self.particles[i].solve()

    def _distribution(self, opt = None):
        """Internal method for distributing particles across the region of interest (RoI). 
        
        Parameters
        ----------
        opt : int
            Specifies the size function for the kde produced mesh:
            0 - Combined kde particle distribution with size function 0:
            1 - Combined kde particle distribution with size function 1:
            2 - Combined kde particle distribution with size function 2:
            3 - Combined kde particle distribution with size function 3:"""

        mesh = _combination_mesh(opt)
        init_coords = np.mean(mesh.nodes[mesh.triangulation], axis = 1)
        M = np.ones((len(mesh.triangulation),3,3))
        M[:,1] = mesh.nodes[mesh.triangulation][:,:,0]
        M[:,2] = mesh.nodes[mesh.triangulation][:,:,1]
        init_vols = abs(0.5*np.linalg.det(M))

        return init_coords, init_vols
         
    def _combination_mesh(self, opt):
        mesh = Mesh(f_img = self.img_sequence[0], g_img = self.img_sequence[1], target_nodes=self.target_nodes, boundary = self.boundary, exclusions = self.exclusions) # Create mesh object with roi corresponding to the first image.
        mesh.nodes = self.mesh.boundary # Overwrite mesh points with roi.
        for i in range(len(self.meshes)):
            mesh.nodes = np.append(mesh.nodes, self.meshes[i].nodes[len(mesh.boundary):], axis=0) # Append non-roi mesh points.
        mesh.triangulation = np.reshape(gmsh.model.mesh.triangulate(mesh.nodes.flatten())-1, (-1,3)) # Define triangles for combined points.
        kernel = spst.gaussian_kde(mesh.nodes.T) # Create kde.
        Z = kernel(np.mean(mesh.nodes[mesh.triangulation], axis=1).T) # Sample kde at element centroids.
        if opt == 0:
            mesh.areas = 0.5*spsp.erfc(3.6*((Z-np.min(Z))/(np.max(Z)-np.min(Z))-0.5))*area # Set target element areas.
        elif opt == 1:
            mesh.areas = area*10**(-2*(Z-np.min(Z))/(np.max(Z)-np.min(Z)))
        elif opt == 2:
            mesh.areas = area*(1-((Z-np.min(Z))/(np.max(Z)-np.min(Z))))
        elif opt == 3:
            mesh.areas = area*(1-((Z-np.min(Z))/(np.max(Z)-np.min(Z))))**2
        mesh._adaptive_remesh(scale=1.7, target=self.target_nodes, nodes = mesh.nodes, triangulation = mesh.triangulation, areas = mesh.areas) # Generate background field. 
        mesh._update_mesh() # Extract the numerical particle control mesh.

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
