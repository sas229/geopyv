import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle, Square
from geopyv.subset import Subset
from geopyv.mesh import Mesh


# Subset test.
print("Performing subset test...")
ref = Image("./tests/ref.jpg")
tar = Image("./tests/tar.jpg")
coord = np.asarray([300.,300.])
subset = Subset(f_img=ref, g_img=tar, f_coord=np.asarray([300,300]), template=Square(25))
subset.inspect()
subset.results()
success = subset.solve()
subset.results()
print("Horizontal displacement: {} px; Vertical displacement: {} px".format(subset.u, subset.v))
subset.convergence()

# # Mesh test.
# print("Performing mesh test...")
# boundary = np.asarray([[200.0, 200.0],[200.,800.],[800.,800.],[800.,200.]])
# mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary)
# mesh.solve(seed_coord=seed, template=Circle(25), adaptive_iterations=3, method = "ICGN")