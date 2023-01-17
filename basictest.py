import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle
from geopyv.subset import Subset
from geopyv.mesh import Mesh


# Subset test.
print("Performing subset test...")
ref = Image("./tests/ref.jpg")
tar = Image("./tests/tar.jpg")
seed = np.asarray([300.,300.])
subset = Subset(seed, ref, tar, template=Circle(25))
subset.inspect()
success = subset.solve()
print("Horizontal displacement: {} px; Vertical displacement: {} px\n".format(subset.u, subset.v))

# Mesh test.
print("Performing mesh test...")
boundary = np.asarray([[200.0, 200.0],[200.,800.],[800.,800.],[800.,200.]])
mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary)
mesh.solve(seed_coord=seed, template=Circle(25), adaptive_iterations=3, method = "ICGN")