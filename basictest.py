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
tar2 = Image("./tests/tar.jpg")
coord = np.asarray([300.,300.])
coord1 = np.asarray([301.,310.])
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
mesh1 = Mesh(f_img=tar, g_img=ref, target_nodes=1000, boundary=boundary)
mesh1.set_target_nodes(500)
mesh.solve(seed_coord=coord, template=template, adaptive_iterations=0, method = "ICGN")
mesh1.solve(seed_coord=coord1, template=template, adaptive_iterations=0, method = "ICGN")
print(id(mesh))
print(id(mesh1))

mesh_memory = tracemalloc.get_traced_memory()
difference = mesh_memory[1]-subset_memory[1]
# del(mesh.subsets)
# mesh_no_subsets_memory = tracemalloc.get_traced_memory()

print("\ngeopyv memory usage:\n")
print("One subset: {:.2f} Mb".format(subset_memory[0]/1000000))
# print("Mesh (inc. 1000 subsets): {:.2f} Mb".format(mesh_memory[0]/1000000))
print("Mesh (no subsets): {:.2f} Mb".format(mesh_memory[0]/1000000))
print("\n")
# print("Difference: {} Mb".format(difference/1000000))

tracemalloc.stop()