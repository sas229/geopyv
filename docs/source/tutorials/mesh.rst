.. _Mesh Tutorial:

Mesh
====

The following tutorial showcases the functionality of the Mesh class, which allows the deformation of a 2D region to be computed between a `reference` and `target` image pair.

Creating a mesh
---------------

Meshes can be created most simply as follows:

.. code-block:: python

    # Imports.
    import geopyv as gp

    # Create mesh.
    mesh = gp.mesh.Mesh()
Solving the mesh
----------------

Adaptivity
^^^^^^^^^^

.. _mesh_data_structure:

Accessing the data
------------------
