"""

Sequence module for geopyv.

"""
import logging
import numpy as np
import scipy as sp
import geopyv as gp 
import re
import os

log = logging.getLogger(__name__)

class SequenceBase:
    """Sequence base class to be used as a mixin."""
    def inspect(self, mesh=None, show=True, block=True, save=None):
        """Method to show the sequence and associated mesh properties."""
        # If a mesh index is given, inspect the mesh.
        #if mesh != None:
        #    if mesh >= 0 and mesh < len(self.data[])


class Sequence(SequenceBase):

    def __init__(self, *, image_folder = '.', image_file_type = ".jpg", target_nodes=1000, boundary=None, exclusions=[], size_lower_bound = 1, size_upper_bound = 1000):
        """Initialisation of geopyv sequence object."""
        self.initialised = False
        # Check types.
        if type(image_folder) != str:
            log.error("image_folder type not recognised. Expected a string.")
            return False
        elif os.path.isdir(image_folder) == False:
            log.error("image_folder does not exist.")
            return False
        if type(image_file_type) != str:
            log.error("image_file_type type not recognised. Expected a string.")
            return False
        elif image_file_type not in [".jpg", ".png", ".bmp"]:
            log.error("image_file_type not recognised. Expected: '.jpg', '.png', or '.bmp'.")
            return False
        if type(target_nodes) != int:
            log.error("Target nodes not of integer type.")
            return False
        if type(boundary) != np.ndarray:
            log.error("Boundary coordinate array of invalid type. Cannot initialise mesh.")
        if np.shape(boundary)[1] != 2:
            log.error("Boundary coordinate array of invalid shape. Must be numpy.ndarray of size (n, 2).")
            return False
        if type(exclusions) != list:
            log.error("Exclusion coordinate array of invalid type. Cannot initialise mesh.")
            return False
        for exclusion in exclusions:
            if np.shape(exclusion)[1] != 2:
                log.error("Exclusion coordinate array of invalid shape. Must be numpy.ndarray of size (n, 2).")
                return False
        
        # Store variables.
        self._image_folder = image_folder
        self._common_file_name = os.path.commonprefix(os.listdir(image_folder)).rstrip('0123456789')
        self._image_indices = np.asarray(sorted([int(re.findall(r'\d+',x)[-1]) for x in os.listdir(image_folder)]))
        self._number_images = np.shape(self._image_indices)[0]
        self._image_file_type = image_file_type
        self._target_nodes = target_nodes
        self._boundary = boundary
        self._exclusions = exclusions
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        
    def solve(self, *, trace = False, seed_coord=None, template=gp.templates.Circle(50), max_iterations=15, max_norm=1e-3, adaptive_iterations=0, method="ICGN", order=1, tolerance=0.7, alpha=0.5, beta=2):

        # Check inputs.
        if type(seed_coord) != np.ndarray:
            try:
                seed_coord = np.asarray(seed_coord)
            except:
                log.error("Seed coordinate is not of numpy.ndarray type. Cannot initiate solver.")
                return False
        elif type(adaptive_iterations) != int:
            log.error("Number of adaptive iterations of invalid type. Must be an integer greater than or equal to zero.")
            return False
        if template == None:
            template = gp.templates.Circle(50)
        elif type(template) != gp.templates.Circle and type(template) != gp.templates.Square:
            log.error("Template is not a type defined in geopyv.templates.")
            return False
        
        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._method = method
        self._order = order
        self._tolerance = tolerance
        self._alpha = alpha
        self._beta = beta
        self._p_0 = np.zeros(6*self._order)

        # Prepare output. 
        self.meshes = np.empty(len(self._image_indices)-1, dtype=object) # Adapted meshes. 
        self.update_register = np.zeros(len(self._image_indices)-1, dtype=int) # Mesh-image reference.

        # Solve. 
        _f_index = 0
        _g_index = 1
        _f_img = gp.image.Image(self._image_folder+"/"+self._common_file_name+str(self._image_indices[_f_index])+self._image_file_type)
        _g_img = gp.image.Image(self._image_folder+"/"+self._common_file_name+str(self._image_indices[_g_index])+self._image_file_type)
        while _g_index < len(self._image_indices-1):
            log.info("Solving for image pair {}-{}.".format(self._image_indices[_f_index], self._image_indices[_g_index]))
            mesh = gp.mesh.Mesh(f_img = _f_img, g_img = _g_img, target_nodes = self._target_nodes, boundary = self._boundary, exclusions = self._exclusions, size_lower_bound = self._size_lower_bound, size_upper_bound = self._size_upper_bound) # Initialise mesh object.
            mesh.solve(seed_coord=self._seed_coord, template=self._template, max_iterations=self._max_iterations, max_norm=self._max_norm, adaptive_iterations=self._adaptive_iterations, method=self._method, order=self._order, tolerance=self._tolerance, alpha=self._alpha, beta=self._beta) # Solve mesh.
            if mesh._update and self.update_register[_g_index-1] == 0: # Correlation coefficient thresholds not met (consequently no mesh generated).  
                if trace:
                    self._trace(_f_index, _g_index)
                _f_index = _g_index - 1
                self.update_register[_f_index] = 1 # Update recorded reference image for future meshes.
                del(_f_img)
                _f_img = gp.image.Image(self._image_folder+"/"+self._common_file_name+str(self._image_indices[_f_index])+self._image_file_type)
            else:
                gp.io.save(object=mesh, filename="mesh_"+str(self._image_indices[_f_index])+"_"+str(self._image_indices[_g_index]))
                del(mesh)
                _g_index += 1 # Iterate the target image index. 
                del(_g_img)
                if _g_index != len(self._image_indices-1):
                    _g_img = gp.image.Image(self._image_folder+"/"+self._common_file_name+str(self._image_indices[_g_index])+self._image_file_type)
        del(_f_img)

    
    def _trace(self, _f_index, _g_index):
        log.message("Tracing exclusion displacement.")
        mesh = gp.io.load(filename="mesh_"+str(self._image_indices[_f_index])+"_"+str(self._image_indices[_g_index-1]))
        i = len(self._boundary)
        for exclusion in self._exclusions:
            j = len(exclusion)
            exclusion += mesh.data["results"]["displacements"][i+j]
        del(mesh)

    def particle(self, coords, vols):
        """A method to propogate "particles" across the domain upon which strain path interpolation is performed."""

        self.particles = np.empty(len(coords), dtype = object)
        for i in range(len(self.particles)):
            self.particles[i] = gp.particle.Particle(coord = coords[i], meshes = self.meshes, update_register = self.update_register, vol = vols[i])
            self.particles[i].solve()
