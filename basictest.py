import logging 
import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle, Square
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv import log
from geopyv import io

level = logging.WARN
log.initialise(level)

# Subset test.
ref = Image("./tests/ref.jpg")
tar = Image("./tests/tar.jpg")
coord = np.asarray([300.,300.])
subset = Subset(f_img=ref, g_img=tar, f_coord=np.asarray([300,300]), template=Circle(25))
# subset = Subset(f_img=ref, g_img=tar)
subset.inspect()
subset.solve()
subset.convergence()
subset.save("test")
print("subset type:", type(subset))
del(subset)

# Load subset test.
subset = io.load("test")
subset.inspect()
subset.convergence()
print("subset type:", type(subset))

# Mesh test.
template = Circle(25)
boundary = np.asarray([[200.0, 200.0],[200.,800.],[800.,800.],[800.,200.]])
mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary)
mesh.solve(seed_coord=coord, template=template, adaptive_iterations=0, method = "ICGN")

