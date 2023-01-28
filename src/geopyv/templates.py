import logging
import numpy as np

log = logging.getLogger(__name__)

class Template:
    """
    
    Template base class.
    
    """
    def __init__(self, size):
        """
        
        Base class for geopyv subset template.

        Parameters
        ----------
        size : int
            Size of the subset.


        Attributes
        ----------
        shape : `str`
            String describing the shape of the subset.
        dimension : `str`
            String describing the meaning of the size attribute.
        size : `int`
            Size of the subset.
        n_px : `int`
            Number of pixels in the subset template.
        coords : `numpy.ndarray` (Nx, 2)
            2D array of subset template coordinates of type `float`.
        subset_mask : `numpy.ndarray` (Nx, 2)
            2D array of coordinates to mask of type `float`.

        """
        self.size = size
        self._check_size_and_type()
        self.n_px = []
        self.coords = []
        self.shape = "None"
        self.dimension = "None"
        self.subset_mask = np.zeros(((2*self.size)+1, (2*self.size)+1)).astype(np.intc)

    def _check_size_and_type(self):
        """
        
        Private method to check if size is a positive integer, and if not convert to
        a positive integer.
        
        """
        # Check size.
        if self.size < 0:
            self.size = int(np.abs(self.size))
            log.warning(
                "Negative subset size specified. Converted to an absolute value of {} pixels.".format(self.size)
            )
        # Check type of size.
        if isinstance(self.size, int) is False:
            self.size = int(self.size)
            log.warning(
                "Subset size not specified as an integer. Converted to an integer of {} pixels.".format(self.size)
            )
        return
    
    def mask(self, centre, mask):
        """
        
        Method to mask subset based on binary mask from mesh.
        
        Parameters
        ----------
        centre : `numpy.ndarray` (x,y)
            Centre of subset.
        mask : `numpy.ndarray` (Nx,Ny)
            Mask to be applied to the mesh. Value of 0 indicates pixels to mask in template.

        """
        x_coords = (self.coords[:,0] + centre[0]).astype(np.intc)
        y_coords = (self.coords[:,1] + centre[1]).astype(np.intc)
        masked_coords = np.zeros((1,2))
        count = 0
        for i in range(np.shape(self.coords)[0]):
            x = x_coords[i]
            y = y_coords[i]
            if mask[y, x] == 1:
                if count == 0:
                    masked_coords = self.coords[i,:]
                else:
                    masked_coords = np.row_stack((masked_coords, self.coords[i,:]))
                count += 1
            else:
                x_s = int(self.coords[i,0]+self.size)
                y_s = int(self.coords[i,1]+self.size)
                self.subset_mask[x_s, y_s] = 1
        self.coords = masked_coords
        self.n_px = np.shape(self.coords)[0]


class Circle(Template):
    """
    
    Circular subset template class.

    """
    def __init__(self, radius=25):
        """
        
        Class for circular subset template. Subclassed from Template.

        Parameters
        ----------
        radius : int, optional
            Radius of the subset. Defaults to a value of 25.


        Attributes
        ----------
        shape : `str`
            String describing the shape of the subset.
        dimension : `str`
            String describing the meaning of the size attribute.
        size : `int`
            Radius of the subset.
        n_px : `int`
            Number of pixels in the subset template.
        coords : `numpy.ndarray` (Nx, 2)
            2D array of subset template coordinates of type `float`.
        subset_mask : `numpy.ndarray` (Nx, 2)
            2D array of coordinates to mask of type `float`.
        
        """
        super().__init__(radius)
        self.shape = "circle"
        self.dimension = "radius"

        # Create template for extracting circular subset information by checking if
        # pixels are within the subset radius.
        x, y = np.meshgrid(
            np.arange(-self.size, self.size + 1, 1),
            np.arange(-self.size, self.size + 1, 1),
        )
        dist = np.sqrt(x ** 2 + y ** 2)
        x_s, y_s = np.where(dist <= self.size)

        # Create template coordinates array.
        self.n_px = x_s.shape[0]
        self.coords = np.empty((self.n_px, 2), order="F")
        self.coords[:, 0] = (x_s - self.size).astype(float)
        self.coords[:, 1] = (y_s - self.size).astype(float)

        # Modify subset mask.
        x, y = np.meshgrid(
            np.arange(-self.size, self.size + 1, 1),
            np.arange(-self.size, self.size + 1, 1),
        )
        dist = np.sqrt(x**2 + y**2)
        self.subset_mask[dist > self.size] = 255


class Square(Template):
    """
    
    Square subset template class.

    """
    def __init__(self, length=25):
        """
        
        Class for square subset template. Subclassed from Template.

        Parameters
        ----------
        length : int, optional
            Half length of the side of the subset. Defaults to a value of 25.


        Attributes
        ----------
        shape : `str`
            String describing the shape of the subset.
        dimension : `str`
            String describing the meaning of the size attribute.
        size : `int`
            Half length of side of the subset.
        n_px : `int`
            Number of pixels in the subset template.
        coords : `numpy.ndarray` (Nx, 2)
            2D array of subset template coordinates of type `float`.
        subset_mask : `numpy.ndarray` (Nx, 2)
            2D array of coordinates to mask of type `float`.
        
        """
        super().__init__(length)
        self.shape = "square"
        self.dimension = "length"

        # Create template for square subset.
        x_s, y_s = np.meshgrid(
            np.arange(-self.size, self.size+1, 1),
            np.arange(-self.size, self.size+1, 1),
        )

        # Create template coordinates array.
        self.n_px = (2*self.size+1)**2
        self.coords = np.empty((self.n_px, 2), order="F")
        self.coords[:, 0] = np.ravel(x_s)
        self.coords[:, 1] = np.ravel(y_s)
