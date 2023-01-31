"""

Object module for geopyv.

"""


class Object:
    """

    Base class for any geopyv object.

    """

    def __init__(self, object_type):
        """

        Base class object initialiser.

        Parameters
        ----------
        object_type : str
            Object type.

        """
        self.object_type = object_type
