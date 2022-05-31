import numpy as np

def area_to_length(area):
        """Function that returns a characteristic length given an element area, based on an equilateral triangle.
        
        Parameters
        ----------
        area : float
            Element area.
        """

        return np.sqrt(4*abs(area)/np.sqrt(3))