import logging 
import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle, Square
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv import log
import tracemalloc

level = logging.WARN
log.initialise(level)

tracemalloc.start()

# Subset test.
ref = Image("./tests/ref.jpg")
tar = Image("./tests/tar.jpg")
coord = np.asarray([300.,300.])
# subset = Subset(f_img=ref, g_img=tar, f_coord=np.asarray([300,300]), template=Square(25))
subset = Subset(f_img=ref, g_img=tar)
subset.inspect()
subset.solve()
subset.convergence()

subset_memory = tracemalloc.get_traced_memory()

# Mesh test.
template = Circle(25)
boundary = np.asarray([[200.0, 200.0],[200.,800.],[800.,800.],[800.,200.]])
mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary)
mesh.solve(seed_coord=coord, template=template, adaptive_iterations=0, method = "ICGN")

mesh_memory = tracemalloc.get_traced_memory()
difference = mesh_memory[1]-subset_memory[1]

print("\n")
print("geopyv memory usage:\n")
print("One subset: {:.2f} Mb".format(subset_memory[0]/1000000))
print("Mesh (1000 subsets): {:.2f} Mb".format(mesh_memory[0]/1000000))
print("\n")

tracemalloc.stop()