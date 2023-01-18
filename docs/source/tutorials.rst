Tutorials
=========

The following sections describe the building blocks of a PIV/DIC analysis performed using the `geopyv` package. 
The tutorials start with the `Subset` object, which allows the displacement of a point to be tracked between two images.
Next, the `Mesh` object is introduced, which is a finite element mesh of subsets. This object includes adaptivity
finctionality, such that small elements are created in regions of high deformation and large elements are created
in regions of low deformation. The `Sequence` object analyses a sequence of `Mesh` generated from a sequence of images, 
and finally the `Particle` object, uses the `Sequence` to develop a strain (and eventually stress) history.


Using the Subset object
-----------------------

This is the most basic object provided by the `geopyv` package. Use it to track the movement of a point between one
image (the reference image) and another (the target image). To begin, import the `Subset` class:

.. code-block:: python

   from geopyv.subset import Subset

The `Subset` object has 

Loading the images
~~~~~~~~~~~~~~~~~~

Using the Mesh object
---------------------

Using the Sequence object
-------------------------

Using the Particle object
-------------------------