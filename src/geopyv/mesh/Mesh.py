import numpy as np
from geopyv.templates import Circle
from geopyv.image import Image
from geopyv.subset import Subset
from geopyv.geometry.utilities import area_to_length
import gmsh
from copy import deepcopy
from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL
from alive_progress import alive_bar

import faulthandler
faulthandler.enable()

class Mesh:

    def __init__(self, f_img, g_img, target_nodes=1000, boundary=None, exclusions=[], size_lower_bound = 25, size_upper_bound = 250):
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

        # Initialise gmsh.
        print("Generating mesh using gmsh with approximately {n} nodes.".format(n=self.target_nodes))
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2) # 0: silent except for fatal errors, 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug
        # Create mesh.
        self._define_RoI()
        self._mesh()

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
        if self.order == 1 and self.method != "WFAGN":
            self.p_0 = np.zeros(6)
        elif self.order == 1 and self.method == "WFAGN":
            self.p_0 = np.zeros(7)
        elif self.order == 2 and self.method != "WFAGN":
            self.p_0 = np.zeros(12)
         
        # Solve initial mesh.
        self.message = "Solving initial mesh"
        self._find_seed_node()
        self._reliability_guided()
      
        # Solve adaptive iterations.
        for iteration in range(1, adaptive_iterations+1):
            self.message = "Adaptive iteration {}".format(iteration)
            D = abs(self.strains[:,2])*self.areas # Elemental shear strain-area products.
            D_b = np.mean(D) # Mean elemental shear strain-area product.
            self.areas *= (np.clip(D/D_b, self.alpha, self.beta))**-2 # Target element areas calculated. 
            f = lambda scale: self._adaptive_remesh(scale, self.target_nodes, self.nodes, self.triangulation, self.areas)
            minimize_scalar(f)
            self._update_mesh()
            self._find_seed_node()
            self._reliability_guided()

        # Finalise.
        if self.update == True:
            print('Error! The minimum correlation coefficient is below tolerance {field:.3f} < {tolerance:.3f}'.format(field=np.min(self.C_CC), tolerance=self.tolerance))
        else:
            print("Solved mesh. Minimum correlation coefficient: {min_C:.3f}; maximum correlation coefficient: {max_C:.3f}.".format(min_C=np.amin(self.C_CC), max_C=np.amax(self.C_CC)))
        gmsh.finalize()


    def _update_mesh(self):
        """Private method to update the mesh variables."""
        _, nc, _ = gmsh.model.mesh.getNodes() # Extracts: node coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2) # Extracts: element node tags.
        self.nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        self.triangulation = np.reshape((np.asarray(ent)-1).flatten(), (-1, 3)) # Element connectivity array. 
    

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
        self.boundary = self.boundary
        self.segments = np.empty((np.shape(self.boundary)[0],2), dtype=np.int32) # Initiate segment array.
        self.segments[:,0] = np.arange(np.shape(self.boundary)[0], dtype=np.int32) # Fill segment array.
        self.segments[:,1] = np.roll(self.segments[:,0],-1) # Fill segment array.
        self.curves = [list(self.segments[:,0])] # Create curve list. 

        # Add exclusions.
        for exclusion in self.exclusions:
            ImageDrawPIL.Draw(binary_img).polygon(exclusion.flatten().tolist(), outline=1, fill=0) # Add exclusion to binary mask.
            current_max_index = np.amax(self.segments) # Highest index used by current segments.
            exclusion_segment = np.empty(np.shape(exclusion)) # Initiate exclusion segment array.
            exclusion_segment[:,0] = np.arange(current_max_index+1, current_max_index+1+np.shape(exclusion)[0]) # Fill exclusion segment array.
            exclusion_segment[:,1] = np.roll(exclusion_segment[:,0],-1) # Fill exclusion segment array.
            self.boundary = np.append(self.boundary, exclusion, axis=0) # Append exclusion to boundary array.
            self.segments = np.append(self.segments, exclusion_segment, axis=0).astype('int32') # Append exclusion segments to segment array.
            self.curves.append(list(exclusion_segment[:,0].astype('int32'))) # Append exclusion curve to curve list.

        # Finalise mask.
        self.mask = np.array(binary_img)


    def _mesh(self):
        """Private method to optimize the element size to generate approximately the desired number of elements."""
        f = lambda size: self._initial_mesh(size, self.boundary, self.segments, self.curves, self.target_nodes)
        res = minimize_scalar(f, bounds=(self.size_lower_bound, self.size_upper_bound), method='bounded')
        _, nc, _ = gmsh.model.mesh.getNodes() # Extracts: node tags, node coordinates, parametric coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2) # Extracts: element types, element tags, element node tags.
        self.nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        self.triangulation = np.reshape((np.asarray(ent)-1).flatten(), (-1, 3)) # Element connectivity array. 
        print("Mesh generated with {n} nodes and {e} elements.".format(n=len(self.nodes), e=len(self.triangulation)))


    @staticmethod
    def _initial_mesh(size, boundary, segments, curves, target_nodes):
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
        gmsh.option.setNumber("Mesh.MeshSizeMin", 25)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize()  
        
        # Get mesh topology.
        _, nc, _ = gmsh.model.mesh.getNodes() # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        number_nodes = len(nodes)

        return abs(number_nodes - target_nodes)

    @staticmethod
    def _adaptive_remesh(scale, target, nodes, triangulation, areas):
        lengths = area_to_length(areas*scale) # Convert target areas to target characteristic lengths.
        bg = gmsh.view.add("bg", 1) # Create background view.
        data = np.pad(nodes[triangulation], ((0,0),(0,0),(0,2)), mode='constant') # Prepare data input (coordinates and buffer).
        data[:,:,3] = np.reshape(np.repeat(lengths, 3), (-1,3)) # Fill data input buffer with target weights.
        data = np.transpose(data, (0,2,1)).flatten() # Reshape for input.
        gmsh.view.addListData(bg, "ST", len(triangulation),  data) # Add data to view.
        bgf = gmsh.model.mesh.field.add("PostView") # Add view to field. 
        gmsh.model.mesh.field.setNumber(bgf, "ViewTag", bg) # Establish field reference (important for function reuse).
        gmsh.model.mesh.field.setAsBackgroundMesh(bgf) # Set field as background.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0) # Prevent boundary influence on mesh.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0) # Prevent point influence on mesh.
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0) # Prevent curve influence on mesh.
        gmsh.model.mesh.clear() # Tidy.
        gmsh.model.mesh.generate(2) # Generate mesh.
        gmsh.model.mesh.optimize()
        nt, nc, npar = gmsh.model.mesh.getNodes() # Extracts: node tags, node coordinates, parametric coordinates.
        ety, et, ent = gmsh.model.mesh.getElements(dim=2) # Extracts: element types, element tags, element node tags.
        nodes = np.column_stack((nc[0::3], nc[1::3])) # Nodal coordinate array (x,y).
        error = (np.shape(nodes)[0] - target)**2
        return error


    def _element_area(self):
        """
        A private method to calculate the element areas.
        """

        M = np.ones((len(self.triangulation),3,3))
        M[:,1] = self.nodes[self.triangulation][:,:,0]
        M[:,2] = self.nodes[self.triangulation][:,:,1]
        self.areas = 0.5*np.linalg.det(M)


    def _element_strains(self):
        """
        A private method to calculate the elemental strain the "B" matrix relating 
        element node displacements to elemental strain..
        """

        self.Bs = np.zeros((len(self.triangulation), 3, 6)) # Initiate B-matrix array. 
        BM = np.asarray([[0,1,-1],[-1,0,1],[1,-1,0]]) # Matrix mutiplier. 
        diff = BM@self.nodes[self.triangulation]
        self.Bs[:,0,::2] = diff[:,:,1] # Input matrix values...
        self.Bs[:,1,1::2] = -diff[:,:,0]
        self.Bs[:,2,::2] = -diff[:,:,0]
        self.Bs[:,2,1::2] = diff[:,:,1]
        self.Bs = 1/(2*self.areas)[:,None,None]*self.Bs
        self.strains = np.einsum('ijk,ikl->ij', self.Bs, np.reshape(self.p[self.triangulation,:2], (-1,6,1)))


    def _reliability_guided(self):
        """
        A private method to perform reliability-guided (RG) PIV analysis.
        """

        # Set up.
        self.update = False
        m = np.shape(self.nodes)[0]
        n = np.shape(self.p_0)[0]
        self.solved = np.zeros(m, dtype = int) # Solved/unsolved reference array (1 if unsolved, -1 if solved).
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
                while np.max(self.solved)>-1:
                    # Identify next subset.
                    cur_idx = np.argmax(self.solved*self.C_CC) # Subset with highest correlation coefficient selected.
                    p_0 = self.subsets[cur_idx].p # Precondition based on selected subset.
                    self.solved[cur_idx] = -1 # Set as solved. 
                    self._neighbours(cur_idx, p_0) # Calculate for neighbouring subsets.
                    if count == number_nodes:
                        break
                    count += 1
                    self.bar()
                
                # If minimum correlation coefficient less than tolerance, raise update flag.
                if np.amin(self.C_CC) < self.tolerance:
                    self.update = True
            else:
                # Set update attribute flag if seed correlation threshold not exceeded.
                self.update = True 
                print('Error! The seed subset correlation is below tolerance {seed:.3f} < {tolerance:.3f}'.format(seed=self.subsets[self.seed_node].C_CC, tolerance=self.subsets[self.seed_node].tolerance))
            
            # Compute element areas and strains.
            self._element_area()
            self._element_strains()

    def _connectivity(self, idx, arr):
        """
        A private method that returns the indices of nodes connected to the index node according to the input array.
        
        Parameters
        ----------
        idx : int
            Index of node. 
        arr : numpy.ndarray (N) 
            Mesh array. 
            
        .. note::
            * If arr is self.triangulation, connectivity finds the nodes connected to the indexed node.
            * If arr is self.segments, connectivity finds the segments connected to the indexed node.
        """
        arr_idxs = np.argwhere(np.any(arr == idx, axis=1)==True).flatten()
        pts_idxs = np.unique(arr[arr_idxs])
        pts_idxs = np.delete(pts_idxs, np.argwhere(pts_idxs==idx))

        return pts_idxs.tolist()

    def _neighbours(self, cur_idx, p_0):
        """
        Method to calculate the correlation coefficients and warp functions of the neighbouring nodes.

        Parameters
        p_0 : numpy.ndarray (N)
            Preconditioning warp function.
        """
        
        neighbours = self._connectivity(cur_idx, self.triangulation)
        for index in neighbours:
            if self.solved[index] == 0: # If not previously solved.
            # Use nearest-neighbout pre-conditioning.
                self.subsets[index].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0=p_0, method=self.method, tolerance=self.tolerance)
                if self.subsets[index].solved: # Check against tolerance.
                    self._store_variables(index)
                else:
                    # Try more extrapolated pre-conditioning.
                    diff = self.nodes[index] - self.nodes[cur_idx]
                    p = self.subsets[cur_idx].p
                    if np.shape(p)[0] == 6:
                        p_0[0] = p[0] + p[2]*diff[0] + p[3]*diff[1]
                        p_0[1] = p[1] + p[4]*diff[0] + p[5]*diff[1]
                    elif np.shape(p)[0] == 12:
                        p_0[0] = p[0] + p[2]*diff[0] + p[3]*diff[1] + 0.5*p[6]*diff[0]**2 + p[7]*diff[0]*diff[1] + 0.5*p[8]*diff[1]**2 # CHECK!!
                        p_0[1] = p[1] + p[4]*diff[0] + p[5]*diff[1] + 0.5*p[9]*diff[0]**2 + p[10]*diff[0]*diff[1] + 0.5*p[11]*diff[1]**2 # CHECK!!
                    self.subsets[index].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0=p_0, method=self.method, tolerance=self.tolerance)
                    if self.subsets[index].solved:
                        self._store_variables(index)
                    else:
                        # Finally, try the NCC initial guess.
                        self.subsets[index].solve(max_norm=self.max_norm, max_iterations=self.max_iterations, p_0 = np.zeros(np.shape(p_0)), method=self.method, tolerance=self.tolerance)
                        if self.subsets[index].C_CC>self.tolerance: # Check against tolerance.
                            self._store_variables(index)
                    

    def _store_variables(self, index, seed=False):
        """Store variables."""
        if seed == True:
            flag = -1
        else:
            flag = 1
        self.solved[index] = flag
        self.C_CC[index] = np.max((self.subsets[index].C_CC, 0)) # Clip correlation coefficient to positive values.
        self.p[index] = self.subsets[index].p.flatten()
        self.displacements[index, 0] = self.subsets[index].u
        self.displacements[index, 1] = -self.subsets[index].v

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