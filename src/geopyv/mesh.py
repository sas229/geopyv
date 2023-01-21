import logging
import numpy as np
import scipy as sp
from geopyv.templates import Circle
from geopyv.image import Image
from geopyv.subset import Subset
from geopyv.geometry.utilities import area_to_length
from geopyv.plots import inspect_subset, convergence_subset, contour_mesh
import gmsh
from copy import deepcopy
from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL
from alive_progress import alive_bar

import faulthandler
import traceback
faulthandler.enable()

log = logging.getLogger(__name__)

class MeshBase:
    """Mesh base class to be used as a mixin."""
    def inspect(self, subset=None, show=True, block=True, save=None):
        """Method to show the mesh and associated quality metrics."""
        if subset != None:
            if subset >= 0 and subset < len(self.data["results"]["subsets"]):
                inspect_subset(self.data["results"]["subsets"][subset], show=show, block=block, save=save)
            else:
                raise ValueError("Subset index provided is out of the range of the mesh object contents.")

    def convergence(self, subset=None, show=True, block=True, save=None):
        """Method to plot the rate of convergence for the mesh."""
        if subset != None:
            if subset >= 0 and subset < len(self.data["results"]["subsets"]):
                convergence_subset(self.data["results"]["subsets"][subset], show=show, block=block, save=save)
            else:
                raise ValueError("Subset index provided is out of the range of the mesh object contents.")
    
    def contour(self, quantity="C_CC", imshow=True, colorbar=True, ticks=None, mesh=False, alpha=0.75, levels=None, axis=None, xlim=None, ylim=None, show=True, block=True, save=None):
        """Method to plot the contours of a given measure."""
        if quantity != None:
            fig, ax = contour_mesh(data=self.data, imshow=imshow, quantity=quantity, colorbar=colorbar, ticks=ticks, mesh=mesh, alpha=alpha, levels=levels, axis=axis, xlim=xlim, ylim=ylim, show=show, block=block, save=save)
            return fig, ax
    
    def quiver(self):
        """Method to plot a quiver plot of the displacements."""
        print("Plot quiver.")

class Mesh(MeshBase):

    def __init__(self, f_img, g_img, target_nodes=1000, boundary=None, exclusions=[], size_lower_bound = 1, size_upper_bound = 1000):
        """Initialisation of geopyv mesh object."""
        self.initialised = False
        # Check types.
        if type(f_img) != Image:
            raise TypeError("Reference image not geopyv.image.Image type.")
        elif type(g_img) != Image:
            raise TypeError("Target image not geopyv.image.Image type.")
        elif type(target_nodes) != int:
            raise TypeError("Maximum number of elements not of integer type.")
        elif target_nodes <= 0:
            raise ValueError("Invalid maximum number of elements.")
        elif type(boundary) != np.ndarray:
            raise TypeError("Boundary coordinate array of invalid type. Cannot initialise mesh.")
        elif type(exclusions) != list:
            raise TypeError("Exclusion coordinate array of invalid type. Cannot initialise mesh.")
        for exclusion in exclusions:
            if np.shape(exclusion)[1] != 2:
                raise ValueError("Exclusion coordinate array of invalid shape. Must be numpy.ndarray of size (n, 2).")

        # Store variables.
        self.f_img = f_img
        self.g_img = g_img
        self.target_nodes = target_nodes
        self.boundary = boundary
        self.exclusions = exclusions
        self.size_lower_bound = size_lower_bound
        self.size_upper_bound = size_upper_bound
        self.solved = False
        self.unsolvable = False

        # Define region of interest.
        self._define_RoI()

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2) # 0: silent except for fatal errors, 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug
        
        # Create initial mesh.
        print("Generating mesh using gmsh with approximately {n} nodes.".format(n=self.target_nodes))
        self._initial_mesh()
        print("Mesh generated with {n} nodes and {e} elements.".format(n=len(self.nodes), e=len(self.elements)))
        gmsh.finalize()

        # Data.
        self.data = {
            "type": "Mesh",
            "solved": self.solved,
            "unsolvable": self.unsolvable,
            "images": {
                "f_img": self.f_img.filepath,
                "g_img": self.g_img.filepath,
            },
            "target_nodes": self.target_nodes,
            "boundary": self.boundary,
            "exclusions": self.exclusions,
            "size_lower_bound": self.size_lower_bound,
            "size_upper_bound": self.size_upper_bound,
            "nodes": self.nodes,
            "elements": self.elements,
        }

    def set_target_nodes(self, target_nodes):
        """Method to create a mesh with a target number of nodes."""
        self.target_nodes = target_nodes

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2) # 0: silent except for fatal errors, 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug
        
        # Create mesh.
        print("Generating mesh using gmsh with approximately {n} nodes.".format(n=self.target_nodes))
        self._initial_mesh()
        print("Mesh generated with {n} nodes and {e} elements.".format(n=len(self.nodes), e=len(self.elements)))
        gmsh.finalize()        

    def solve(self, seed_coord=None, template=Circle(50), max_iterations=15, max_norm=1e-3, adaptive_iterations=0, method="ICGN", order=1, tolerance=0.7, alpha=0.5, beta=2):

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
        self.subset_bgf_nodes = None
        self.subset_bgf_values = None
        self.update = False
        if self.order == 1 and self.method != "WFAGN":
            self.p_0 = np.zeros(6)
        elif self.order == 1 and self.method == "WFAGN":
            self.p_0 = np.zeros(7)
        elif self.order == 2 and self.method != "WFAGN":
            self.p_0 = np.zeros(12)

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2) # 0: silent except for fatal errors, 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug
        
        # Create initial mesh.
        self._initial_mesh()
         
        # Solve initial mesh.
        self.message = "Solving initial mesh"
        self._find_seed_node()
        try:
            self._reliability_guided()
            # Solve adaptive iterations.
            for iteration in range(1, adaptive_iterations+1):
                self.message = "Adaptive iteration {}".format(iteration)
                self._adaptive_mesh()
                self._update_mesh()
                self._adaptive_subset()
                self._find_seed_node()
                self._reliability_guided()

            # Finalise.
            if self.update == True:
                self.solved = False
                self.unsolvable = True
                print('Error! The minimum correlation coefficient is below tolerance {field:.3f} < {tolerance:.3f}'.format(field=np.min(self.C_CC), tolerance=self.tolerance))
            else:
                # Pack data.
                self.solved = True
                self.data["nodes"] = self.nodes
                self.data["elements"] = self.elements
                self.data["solved"] = self.solved
                self.data["unsolvable"] = self.unsolvable

                # Pack settings.
                self.settings = {
                    "max_iterations": self.max_iterations,
                    "max_norm": self.max_norm,
                    "adaptive_iterations": self.adaptive_iterations,
                    "method": self.method,
                    "order": self.order,
                    "tolerance": self.tolerance,
                }
                self.data.update({"settings": self.settings})

                # Extract data from subsets.
                subset_data = []
                for subset in self.subsets:
                    subset_data.append(subset.data)

                # Pack results.
                self.results = {
                    "subsets": subset_data,
                    "displacements": self.displacements,
                    "du": self.du,
                    "d2u": self.d2u,
                    "C_CC": self.C_CC,
                }
                self.data.update({"results": self.results})
                print("Solved mesh. Minimum correlation coefficient: {min_C:.3f}; maximum correlation coefficient: {max_C:.3f}.".format(min_C=np.amin(self.C_CC), max_C=np.amax(self.C_CC)))
        except ValueError:
            print(traceback.format_exc())
            print("Error! Could not solve for all subsets.")
            self.update = True
        gmsh.finalize()
        
    def _update_mesh(self):
        """Private method to update the mesh variables."""
        _, nc, _ = gmsh.model.mesh.getNodes() # Extracts: node coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2) # Extracts: element node tags.
        self.nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        self.elements = np.reshape((np.asarray(ent)-1).flatten(), (-1, 6)) # Element connectivity array. 
    
    def _find_seed_node(self):
        """Private method to find seed node given seed coordinate."""
        dist = np.sqrt((self.nodes[:,0]-self.seed_coord[0])**2 + (self.nodes[:,1]-self.seed_coord[1])**2)
        self.seed_node = np.argmin(dist)

    def _define_RoI(self):
        """
        Private method to define the RoI.
        """

        # Create binary mask RoI.
        binary_img = ImagePIL.new('L', (np.shape(self.f_img.image_gs)[1], np.shape(self.f_img.image_gs)[0]), 0)
        ImageDrawPIL.Draw(binary_img).polygon(self.boundary.flatten().tolist(), outline=1, fill=1)

        # Create objects for mesh generation.
        self.segments = np.empty((np.shape(self.boundary)[0],2), dtype=np.int32) # Initiate segment array.
        self.segments[:,0] = np.arange(np.shape(self.boundary)[0], dtype=np.int32) # Fill segment array.
        self.segments[:,1] = np.roll(self.segments[:,0],-1) # Fill segment array.
        self.curves = [list(self.segments[:,0])] # Create curve list. 

        # Add exclusions.
        for exclusion in self.exclusions:
            ImageDrawPIL.Draw(binary_img).polygon(exclusion.flatten().tolist(), outline=1, fill=0) # Add exclusion to binary mask.
            cur_max_idx = np.amax(self.segments) # Highest index used by current segments.
            exclusion_segment = np.empty(np.shape(exclusion)) # Initiate exclusion segment array.
            exclusion_segment[:,0] = np.arange(cur_max_idx+1, cur_max_idx+1+np.shape(exclusion)[0]) # Fill exclusion segment array.
            exclusion_segment[:,1] = np.roll(exclusion_segment[:,0],-1) # Fill exclusion segment array.
            self.boundary = np.append(self.boundary, exclusion, axis=0) # Append exclusion to boundary array.
            self.segments = np.append(self.segments, exclusion_segment, axis=0).astype('int32') # Append exclusion segments to segment array.
            self.curves.append(list(exclusion_segment[:,0].astype('int32'))) # Append exclusion curve to curve list.

        # Finalise mask.
        self.mask = np.array(binary_img)

    def _initial_mesh(self):
        """Private method to optimize the element size to generate approximately the desired number of elements."""
        f = lambda size: self._uniform_remesh(size, self.boundary, self.segments, self.curves, self.target_nodes, self.size_lower_bound)
        res = minimize_scalar(f, bounds=(self.size_lower_bound, self.size_upper_bound), method='bounded')
        _, nc, _ = gmsh.model.mesh.getNodes() # Extracts: node tags, node coordinates, parametric coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2) # Extracts: element types, element tags, element node tags.
        self.nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        self.elements = np.reshape((np.asarray(ent)-1).flatten(), (-1, 6)) # Element connectivity array. 

    def _adaptive_mesh(self):
        D = abs(self.du[:,0,1]+self.du[:,1,0])*self.areas # Elemental shear strain-area products.
        D_b = np.mean(D) # Mean elemental shear strain-area product.
        self.areas *= (np.clip(D/D_b, self.alpha, self.beta))**-2 # Target element areas calculated. 
        f = lambda scale: self._adaptive_remesh(scale, self.target_nodes, self.nodes, self.elements, self.areas)
        minimize_scalar(f)

    @staticmethod
    def _uniform_remesh(size, boundary, segments, curves, target_nodes, size_lower_bound):
        """
        Private method to prepare the initial mesh.
        """
        # Make mesh.
        gmsh.model.add("base") # Create model.

        # Add points.
        for i in range(np.shape(boundary)[0]):
            gmsh.model.occ.addPoint(boundary[i,0], boundary[i,1], 0, size, i)       
        
        # Add line segments.
        for i in range(np.shape(segments)[0]):
            gmsh.model.occ.addLine(segments[i,0], segments[i,1], i)
        
        # Add curves.
        for i in range(len(curves)):
            gmsh.model.occ.addCurveLoop(curves[i], i)
        curve_indices = list(np.arange(len(curves), dtype=np.int32))

        # Create surface.        
        gmsh.model.occ.addPlaneSurface(curve_indices, 0)
        
        # Generate mesh.
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_lower_bound)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize() 
        gmsh.model.mesh.setOrder(2)
        
        # Get mesh topology.
        _, nc, _ = gmsh.model.mesh.getNodes() # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        number_nodes = len(nodes)

        return abs(number_nodes - target_nodes)

    @staticmethod
    def _adaptive_remesh(scale, target, nodes, triangulation, areas):
        lengths = area_to_length(areas*scale) # Convert target areas to target characteristic lengths.
        bg = gmsh.view.add("bg", 1) # Create background view.
        data = np.pad(nodes[triangulation[:,:3]], ((0,0),(0,0),(0,2)), mode='constant') # Prepare data input (coordinates and buffer).
        data[:,:,3] = np.reshape(np.repeat(lengths, 3), (-1,3)) # Fill data input buffer with target weights.
        data = np.transpose(data, (0,2,1)).flatten() # Reshape for input.
        gmsh.view.addListData(bg, "ST", len(triangulation),  data) # Add data to view.
        bgf = gmsh.model.mesh.field.add("PostView") # Add view to field. 
        gmsh.model.mesh.field.setNumber(bgf, "ViewTag", bg) # Establish field reference (important for function reuse).
        gmsh.model.mesh.field.setAsBackgroundMesh(bgf) # Set field as background.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0) # Prevent boundary influence on mesh.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0) # Prevent point influence on mesh.
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0) # Prevent curve influence on mesh.
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.clear() # Tidy.
        gmsh.model.mesh.generate(2) # Generate mesh.
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(2)
        
        nt, nc, npar = gmsh.model.mesh.getNodes() # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        error = (np.shape(nodes)[0] - target)**2
        return error

    def _adaptive_subset(self):
        subset_bgf = sp.interpolate.RBFInterpolator(self.subset_bgf_nodes,self.subset_bgf_values,neighbors = 10, kernel = "cubic")
        subset_sizes = subset_bgf(self.nodes)

    def _update_subset_bgf(self):
        if self.subset_bgf_nodes is not None:
            self.subset_bgf_nodes = np.append(self.subset_bgf_nodes, np.mean(self.nodes[self.elements], axis = 1),axis=0)
            self.subset_bgf_values = np.append(self.subset_bgf_values, np.mean(self.d2u, axis=(1,2)), axis = 0)
        else:
            self.subset_bgf_nodes = np.mean(self.nodes[self.elements], axis = 1)
            self.subset_bgf_values = np.mean(self.d2u, axis=(1,2))

    def _element_area(self):
        """
        A private method to calculate the element areas.
        """

        M = np.ones((len(self.elements),3,3))
        M[:,1] = self.nodes[self.elements[:,:3]][:,:,0] # [:3] will provide corner nodes in both 1st and 2nd order element case.
        M[:,2] = self.nodes[self.elements[:,:3]][:,:,1]
        self.areas = 0.5*np.linalg.det(M)

    def _element_strains(self):
        """
        A private method to calculate the elemental strain the "B" matrix relating 
        element node displacements to elemental strain.
        """
        # Local coordinates
        A = np.ones((len(self.elements),3,3))
        A[:,:,1:] = self.nodes[self.elements[:,:3]]
        lc = np.ones((len(self.elements,),3))/3

        # Weighting function (and derivatives to 2nd order)
        N = np.zeros((len(self.elements),6))
        N = np.asarray([1/9,1/9,1/9,4/9,4/9,4/9])
        dN = np.asarray([[1/3,0,-1/3,4/3,-4/3,0],
                        [0,1/3,-1/3,4/3,0,-4/3]])
        d2N = np.asarray([[4,0,4,0,0,-8],
                            [0,0,4,4,-4,-4],
                            [0,4,4,0,-8,0]])

        # 1st Order Strains
        J_x_T = dN@self.nodes[self.elements]
        J_u_T = dN@self.displacements[self.elements]
        du = np.linalg.inv(J_x_T)@J_u_T

        # 2nd Order Strains
        d2udzeta2 = d2N@self.displacements[self.elements]    
        J_zeta = np.zeros((len(self.elements),2,2))
        J_zeta[:,0,0] = self.nodes[self.elements][:,1,1]-self.nodes[self.elements][:,2,1]
        J_zeta[:,0,1] = self.nodes[self.elements][:,2,0]-self.nodes[self.elements][:,1,0]
        J_zeta[:,1,0] = self.nodes[self.elements][:,2,1]-self.nodes[self.elements][:,0,1]
        J_zeta[:,1,1] = self.nodes[self.elements][:,0,0]-self.nodes[self.elements][:,2,0]
        J_zeta /= np.linalg.det(A)[:,None,None]
        d2u = np.zeros((len(self.elements),3,2))
        d2u[:,0,0] = d2udzeta2[:,0,0]*J_zeta[:,0,0]**2+2*d2udzeta2[:,1,0]*J_zeta[:,0,0]*J_zeta[:,1,0]+d2udzeta2[:,2,0]*J_zeta[:,1,0]**2
        d2u[:,0,1] = d2udzeta2[:,0,1]*J_zeta[:,0,0]**2+2*d2udzeta2[:,1,1]*J_zeta[:,0,0]*J_zeta[:,1,0]+d2udzeta2[:,2,1]*J_zeta[:,1,0]**2
        d2u[:,1,0] = d2udzeta2[:,0,0]*J_zeta[:,0,0]*J_zeta[:,0,1]+d2udzeta2[:,1,0]*(J_zeta[:,0,0]*J_zeta[:,1,1]+J_zeta[:,1,0]*J_zeta[:,0,1])+d2udzeta2[:,2,0]*J_zeta[:,1,0]*J_zeta[:,1,1]
        d2u[:,1,1] = d2udzeta2[:,0,1]*J_zeta[:,0,0]*J_zeta[:,0,1]+d2udzeta2[:,1,1]*(J_zeta[:,0,0]*J_zeta[:,1,1]+J_zeta[:,1,0]*J_zeta[:,0,1])+d2udzeta2[:,2,1]*J_zeta[:,1,0]*J_zeta[:,1,1]
        d2u[:,2,0] = d2udzeta2[:,0,0]*J_zeta[:,0,1]**2+2*d2udzeta2[:,1,0]*J_zeta[:,0,1]*J_zeta[:,1,1]+d2udzeta2[:,2,0]*J_zeta[:,1,1]**2
        d2u[:,2,1] = d2udzeta2[:,0,1]*J_zeta[:,0,1]**2+2*d2udzeta2[:,1,1]*J_zeta[:,0,1]*J_zeta[:,1,1]+d2udzeta2[:,2,1]*J_zeta[:,1,1]**2
        
        self.du = du
        self.d2u = d2u

    def _reliability_guided(self):
        """
        A private method to perform reliability-guided (RG) PIV analysis.
        """

        # Set up.
        m = np.shape(self.nodes)[0]
        n = np.shape(self.p_0)[0]
        self.subset_solved = np.zeros(m, dtype = int) # Solved/unsolved reference array (1 if unsolved, -1 if solved).
        self.C_CC = np.zeros(m, dtype=np.float64) # Correlation coefficient array. 
        self.subsets = np.empty(m, dtype = object) # Initiate subset array.
        self.p = np.zeros((m, n), dtype=np.float64) # Warp function array.
        self.displacements = np.zeros((m, 2), dtype=np.float64) # Displacement output array.
        
        # All nodes.
        entities = gmsh.model.getEntities()
        self.node_tags = []
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(e[0], e[1])
            self.node_tags = np.append(self.node_tags, tags.flatten()).astype(int)

        # Interior and boundary nodes.
        entities = gmsh.model.getEntities(2)
        self.interior_node_tags = []
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(e[0], e[1])
            self.interior_node_tags = np.append(self.interior_node_tags, tags.flatten()).astype(int)
        self.boundary_node_tags = np.setdiff1d(self.node_tags, self.interior_node_tags).astype(int)-1
        
        # Template masking using binary mask.
        for tag in range(len(self.node_tags)):
            if tag in self.boundary_node_tags:
                centre = self.nodes[tag]
                template = deepcopy(self.template)
                template.mask(centre, self.mask)
                self.subsets[tag] = Subset(self.nodes[tag], self.f_img, self.g_img, template) # Create masked boundary subset. 
            else:
                self.subsets[tag] = Subset(self.nodes[tag], self.f_img, self.g_img, self.template) # Create full subset. 
        # Solve subsets in mesh.
        number_nodes = np.shape(self.nodes)[0]
        with alive_bar(number_nodes, dual_line=True, bar='blocks', title=self.message) as self.bar:
            # Solve for seed.
            self.bar.text = "-> Solving seed subset..."
            self.subsets[self.seed_node].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0=self.p_0, method=self.method, tolerance=0.9) # Solve for seed subset.
            self.bar()
            if self.subsets[self.seed_node].solved:
                self._store_variables(self.seed_node, seed=True)
                
                # Solve for neighbours of the seed subset.
                p_0 = self.subsets[self.seed_node].p # Set seed subset warp function as the preconditioning. 
                self._neighbours(self.seed_node, p_0) # Solve for neighbouring subsets.

                # Solve through sorted queue.
                self.bar.text = "-> Solving remaining subsets using reliability guided approach..."
                count = 0
                while np.max(self.subset_solved)>-1:
                    # Identify next subset.
                    cur_idx = np.argmax(self.subset_solved*self.C_CC) # Subset with highest correlation coefficient selected.
                    p_0 = self.subsets[cur_idx].p # Precondition based on selected subset.
                    self.subset_solved[cur_idx] = -1 # Set as solved. 
                    self._neighbours(cur_idx, p_0) # Calculate for neighbouring subsets.
                    if count == number_nodes:
                        break
                    count += 1
                    self.bar()
                
                # Update
                if not any(self.subset_solved != -1): # If all solved...
                    if np.amin(self.C_CC) < self.tolerance: # ...but minimum correlation coefficient is less than tolerance...
                        self.update = True # ... raise update flag.
                else: # If any remain unsolved...
                    self.update = True #... raise update flag.
            else:
                # Set update attribute flag if seed correlation threshold not exceeded.
                self.update = True 
                print('Error! The seed subset correlation is below tolerance {seed:.3f} < {tolerance:.3f}'.format(seed=self.subsets[self.seed_node].C_CC, tolerance=self.subsets[self.seed_node].tolerance))
            
            # Compute element areas and strains.
            self._element_area()
            self._element_strains()
            self._update_subset_bgf()

    def _connectivity(self,idx):
        """
        A private method that returns the indices of nodes connected to the index node according to the input array.
        
        Parameters
        ----------
        idx : int
            Index of node. 
        arr : numpy.ndarray (N) 
            Mesh array. 

        """
        element_idxs = np.argwhere(self.elements==idx)
        pts_idxs = []
        for i in range(len(element_idxs)):
            if element_idxs[i,1] == 0: # If 1
                pts_idxs.append(self.elements[element_idxs[i,0],3::2]) # Add 4,6
            elif element_idxs[i,1] == 1: #If 2
                pts_idxs.append(self.elements[element_idxs[i,0],3:5]) # Add 4,5
            elif element_idxs[i,1] == 2: # If 3
                pts_idxs.append(self.elements[element_idxs[i,0],4:]) # Add 5,6
            elif element_idxs[i,1] == 3: # If 4
                pts_idxs.append(self.elements[element_idxs[i,0],:2]) # Add 1,2
            elif element_idxs[i,1] == 4: # If 5
                pts_idxs.append(self.elements[element_idxs[i,0],1:3]) # Add 2,3
            elif element_idxs[i,1] == 5: # If 6
                pts_idxs.append(self.elements[element_idxs[i,0],:3:2]) # Add 1,3
        pts_idxs = np.unique(pts_idxs)
        
        return pts_idxs

    def _neighbours(self, cur_idx, p_0):
        """
        Method to calculate the correlation coefficients and warp functions of the neighbouring nodes.

        Parameters
        p_0 : numpy.ndarray (N)
            Preconditioning warp function.
        """
        
        neighbours = self._connectivity(cur_idx)
        for idx in neighbours:
            if self.subset_solved[idx] == 0: # If not previously solved.
            # Use nearest-neighbout pre-conditioning.
                self.subsets[idx].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0=p_0, method=self.method, tolerance=self.tolerance)
                if self.subsets[idx].solved: # Check against tolerance.
                    self._store_variables(idx)
                else:
                    # Try more extrapolated pre-conditioning.
                    diff = self.nodes[idx] - self.nodes[cur_idx]
                    p = self.subsets[cur_idx].p
                    if np.shape(p)[0] == 6:
                        p_0[0] = p[0] + p[2]*diff[0] + p[3]*diff[1]
                        p_0[1] = p[1] + p[4]*diff[0] + p[5]*diff[1]
                    elif np.shape(p)[0] == 12:
                        p_0[0] = p[0] + p[2]*diff[0] + p[3]*diff[1] + 0.5*p[6]*diff[0]**2 + p[7]*diff[0]*diff[1] + 0.5*p[8]*diff[1]**2 # CHECK!!
                        p_0[1] = p[1] + p[4]*diff[0] + p[5]*diff[1] + 0.5*p[9]*diff[0]**2 + p[10]*diff[0]*diff[1] + 0.5*p[11]*diff[1]**2 # CHECK!!
                    self.subsets[idx].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0=p_0, method=self.method, tolerance=self.tolerance)
                    if self.subsets[idx].solved:
                        self._store_variables(idx)
                    else:
                        # Finally, try the NCC initial guess.
                        self.subsets[idx].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0 = np.zeros(np.shape(p_0)), method=self.method, tolerance=self.tolerance)
                        if self.subsets[idx].solved:
                                self._store_variables(idx)

    def _store_variables(self, idx, seed=False):
        """Store variables."""
        if seed == True:
            flag = -1
        else:
            flag = 1
        self.subset_solved[idx] = flag
        self.C_CC[idx] = np.max((self.subsets[idx].C_CC, 0)) # Clip correlation coefficient to positive values.
        self.p[idx] = self.subsets[idx].p.flatten()
        self.displacements[idx, 0] = self.subsets[idx].u
        self.displacements[idx, 1] = self.subsets[idx].v

class MeshResults(MeshBase):
    """MeshResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Mesh object.

    Attributes
    ----------
    data : dict
        geopyv data dict from Mesh object.
    """

    def __init__(self, data):
        """Initialisation of geopyv MeshResults class."""
        self.data = data