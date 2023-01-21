import logging 
import numpy as np
import geopyv
import matplotlib.pyplot as plt

level = logging.WARN
geopyv.log.initialise(level)

# # Subset test.
# ref = geopyv.image.Image("./images/T-Bar/IMG_1062.jpg")
# tar = geopyv.image.Image("./images/T-Bar/IMG_1064.jpg")
# template = geopyv.templates.Circle(50)
# subset = geopyv.subset.Subset(f_img=ref, g_img=tar, f_coord=np.asarray([400,400]), template=template)
# # subset = Subset(f_img=ref, g_img=tar)
# subset.inspect()
# subset.solve()
# subset.convergence()

# geopyv.io.save(subset, "test")
# del(subset)

# # Load subset test.
# subset = geopyv.io.load("test")
# subset.inspect(show=False)
# subset.convergence(show=False)
# print("subset type:", type(subset))

# # Mesh test.
# template = geopyv.templates.Circle(25)
# boundary = np.asarray(
#     [[200.0, 200.0],
#     [200.0, 2700.0],
#     [3900.0, 2700.0],
#     [3900.0, 200.0]]
# )
# exclusions = []
# exclusions.append(geopyv.geometry.exclusions.circular_exclusion(np.asarray([1925, 1470]), radius=430, size=100))
# seed = np.asarray([400, 400.0])
# mesh = geopyv.mesh.Mesh(f_img=ref, g_img=tar, target_nodes=2000, boundary=boundary, exclusions=exclusions)
# geopyv.io.save(mesh, "mesh")
# beta = 5.0
# alpha = 1/beta
# mesh.solve(seed_coord=seed, template=template, adaptive_iterations=0, method="ICGN", alpha=alpha, beta=beta)
# geopyv.io.save(mesh, "mesh")
# del(mesh)

mesh = geopyv.io.load("mesh")

# The commands are basically standard matplotlib...
mesh.contour(quantity="u", colorbar=False, alpha=0.75, levels=np.arange(-5, 6, 1), axis="off", xlim=((900,2900)), ylim=((500,2500)))
mesh.contour(quantity="C_CC")
mesh.contour(quantity="iterations")

# You can also return the fig and ax objects and add customised items to the plot.
fig, ax = mesh.contour("v", alpha=1.0, levels=np.arange(-5, 6, 1))
ax.plot([0, 2000], [1000, 1000], color='k', linewidth=3.0, zorder=10)
ax.set_xlim((0,2000))
plt.show()

# Let's inspect some clipped subsets.
mesh.inspect(subset=0)
mesh.inspect(subset=1)
mesh.inspect(subset=2)
mesh.inspect(subset=3)
mesh.inspect(subset=4)

# You can inspect subsets and their convergence via the mesh object by passing a subset number.
mesh.inspect(subset=0)
mesh.convergence(subset=0)

# If you supply a subset index that is out of range you get a ValueError.
mesh.convergence(subset=4000)