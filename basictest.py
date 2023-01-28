import numpy as np
import geopyv as gp
import matplotlib.pyplot as plt

# Subset test.
ref = gp.image.Image("./images/T-Bar/IMG_1062.jpg")
tar = gp.image.Image("./images/T-Bar/IMG_1064.jpg")
template = gp.templates.Circle(50)
subset = gp.subset.Subset(f_img=ref, g_img=tar, f_coord=np.asarray([1000,1000]), template=template)
# subset = gp.subset.Subset(g_img=tar, template=template)
subset.inspect()
subset.solve()
subset.convergence()
gp.io.save(subset, "test")
gp.io.save(subset)
del(subset)

# Load subset test.
subset = gp.io.load("test")
subset.inspect(show=False)
subset.convergence(show=False)

# Mesh test.
template = gp.templates.Circle(50)
boundary = np.asarray(
    [[200.0, 200.0],
    [200.0, 2700.0],
    [3900.0, 2700.0],
    [3900.0, 200.0]]
)
exclusions = []
exclusions.append(gp.geometry.exclusions.circular_exclusion(np.asarray([1925, 1470]), radius=430, size=100))
seed = np.asarray([400, 400.0])
mesh = gp.mesh.Mesh(f_img=ref, g_img=tar, target_nodes=2000, boundary=boundary, exclusions=exclusions)
mesh.inspect()
gp.io.save(mesh, "mesh")
beta = 5.0
alpha = 1/beta
mesh.solve(seed_coord=seed, template=template, adaptive_iterations=0, method="ICGN", alpha=alpha, beta=beta, tolerance=0.7)
gp.io.save(mesh, "mesh")
del(mesh)

mesh = gp.io.load("mesh")

# The commands are basically standard matplotlib...
mesh.convergence()
mesh.convergence(quantity="norm")
mesh.convergence(quantity="iterations")
mesh.quiver()
mesh.contour(quantity="u", colorbar=False, alpha=0.75, levels=np.arange(-5, 6, 1), axis="off", xlim=((900,2900)), ylim=((500,2500)))
mesh.contour(quantity="C_ZNCC", mesh=True)
mesh.contour(quantity="iterations")
mesh.contour(quantity="R")

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

# Sequence test.
template = gp.templates.Circle(50)
boundary = np.asarray(
    [[200.0, 200.0],
    [200.0, 2700.0],
    [3900.0, 2700.0],
    [3900.0, 200.0]]
)
exclusions = []
exclusions.append(gp.geometry.exclusions.circular_exclusion(np.asarray([1925, 1470]), radius=430, size=100))
seed = np.asarray([400, 400.0])
sequence = gp.sequence.Sequence(image_folder = "./images/T-Bar", target_nodes=1000, boundary=boundary, exclusions=exclusions)

beta = 5.0
alpha = 1/beta
sequence.solve(trace = False, seed_coord=seed, template=template, adaptive_iterations=0, method="ICGN", alpha=alpha, beta=beta, tolerance=0.7)
