import logging 
import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle, Square
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv import log
from geopyv import io
from geopyv.geometry.exclusions import circular_exclusion
import matplotlib.pyplot as plt

level = logging.WARN
log.initialise(level)

# # Subset test.
# ref = Image("./images/T-Bar/IMG_1062.jpg")
# tar = Image("./images/T-Bar/IMG_1064.jpg")
# subset = Subset(f_img=ref, g_img=tar, f_coord=np.asarray([400,400]), template=Circle(50))
# # subset = Subset(f_img=ref, g_img=tar)
# subset.inspect()
# subset.solve()
# subset.convergence()
# # subset.save("test")
# # print("subset type:", type(subset))
# # del(subset)

# io.save(subset, "test")
# del(subset)

# Load subset test.
subset = io.load("test")
subset.inspect(show=False)
subset.convergence(show=False)
print("subset type:", type(subset))

# # Mesh test.
# template = Circle(25)
# boundary = np.asarray(
#     [[200.0, 200.0],
#     [200.0, 2700.0],
#     [3900.0, 2700.0],
#     [3900.0, 200.0]]
# )
# exclusions = []
# exclusions.append(circular_exclusion(np.asarray([1925, 1470]), radius=430, size=100))
# seed = np.asarray([400, 400.0])
# mesh = Mesh(f_img=ref, g_img=tar, target_nodes=2000, boundary=boundary, exclusions=exclusions)
# io.save(mesh, "mesh")
# beta = 5.0
# alpha = 1/beta
# mesh.solve(seed_coord=seed, template=template, adaptive_iterations=0, method = "ICGN", alpha=alpha, beta=beta)
# io.save(mesh, "mesh")
# del(mesh)

mesh = io.load("mesh")

# The commands are basically standard matplotlib...
mesh.contour("u", colorbar=False, alpha=0.75, levels=np.arange(-5, 6, 1), axis="off", xlim=((900,2900)), ylim=((500,2500)))

# You can also return the fig and ax objects and add customised items to the plot.
fig, ax = mesh.contour("v", alpha=1.0, levels=np.arange(-5, 6, 1))
ax.plot([0, 2000], [1000, 1000], color='k', linewidth=3.0, zorder=10)
# plt.show()

# mesh.contour("C_CC", mesh=True)
# mesh.contour("u_x")
# mesh.contour("u_y", mesh=True)
# mesh.contour("v_x", imshow=False)
# mesh.contour("v_y")
# print(type(mesh.data["results"]["subsets"]))
# mesh.inspect(subset=0)
# mesh.convergence(subset=0)