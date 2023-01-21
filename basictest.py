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
import matplotlib.tri as tri
# from matplotlib.colors import ListedColormap

level = logging.WARN
log.initialise(level)

# Subset test.
ref = Image("./images/T-Bar/IMG_1062.jpg")
tar = Image("./images/T-Bar/IMG_1064.jpg")
subset = Subset(f_img=ref, g_img=tar, f_coord=np.asarray([400,400]), template=Circle(50))
# subset = Subset(f_img=ref, g_img=tar)
subset.inspect()
subset.solve()
subset.convergence()
# subset.save("test")
# print("subset type:", type(subset))
# del(subset)

io.save(subset, "test")
del(subset)

# Load subset test.
subset = io.load("test")
subset.inspect()
subset.convergence()
print("subset type:", type(subset))

# Mesh test.
template = Circle(25)
boundary = np.asarray(
    [[200.0, 200.0],
    [200.0, 2700.0],
    [3900.0, 2700.0],
    [3900.0, 200.0]]
)
exclusions = []
exclusions.append(circular_exclusion(np.asarray([1925, 1470]), radius=430, size=100))
seed = np.asarray([400, 400.0])
mesh = Mesh(f_img=ref, g_img=tar, target_nodes=2000, boundary=boundary, exclusions=exclusions)
io.save(mesh, "mesh")
beta = 5.0
alpha = 1/beta
mesh.solve(seed_coord=seed, template=template, adaptive_iterations=0, method = "ICGN", alpha=alpha, beta=beta)
io.save(mesh, "mesh")
del(mesh)

mesh = io.load("mesh")
print(type(mesh.data["results"]["displacements"]))

# # Plot quantity and levels.
# # quantity = node_strains[:,2]*100 # Percentage strains.
# quantity = mesh.displacements[:,1]
# # quantity = mesh.du[:,0]
# # max = np.round(np.ceil(np.max(np.abs(quantity))),0)
# # levels = np.arange(-max, max+0.5, 0.5)
# levels = np.arange(-10, 10+0.5, 0.5)

# # Analyse and plot data (to be moved into Mesh as methods...).
# unsolved = np.flatnonzero(np.where(mesh.solved > -1, mesh.solved,0))
# poor = np.flatnonzero(np.where(mesh.C_CC < 0.7, mesh.solved,0))

# fig, ax = plt.subplots()
# node_order = [0 ,3, 1, 4, 2, 5]
# elements = mesh.elements
# plot_element_structure = [[0, 3, 5], [1, 3, 4], [2, 4, 5], [3, 4, 5]]
# plot_elements = []
# for element in elements:
#     plot_elements.append([element[0], element[3], element[5]])
#     plot_elements.append([element[1], element[3], element[4]])
#     plot_elements.append([element[2], element[4], element[5]])
#     plot_elements.append([element[3], element[4], element[5]])
#     x = [mesh.subsets[element[0]].x, mesh.subsets[element[3]].x, mesh.subsets[element[1]].x, mesh.subsets[element[4]].x, mesh.subsets[element[2]].x, mesh.subsets[element[5]].x, mesh.subsets[element[0]].x]
#     y = [mesh.subsets[element[0]].y, mesh.subsets[element[3]].y, mesh.subsets[element[1]].y, mesh.subsets[element[4]].y, mesh.subsets[element[2]].y, mesh.subsets[element[5]].y, mesh.subsets[element[0]].y]
#     ax.plot(x, y, color="cyan", alpha=0.25, linewidth = '0.5')
# mesh_triangulation = np.asarray(plot_elements)
# triangulation = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh_triangulation)
# contours = ax.tripcolor(triangulation, quantity, vmin=-10, vmax=10, shading="gouraud")
# # contours = ax.tricontourf(triangulation, quantity, alpha=0.75, vmin=-10, vmax=10, levels=50)
# # ax.triplot(triangulation, linewidth=0.5, color='k', linestyle='-', alpha=0.25)
# fig.colorbar(contours, label = 'v [px]')
# plt.imshow(mesh.f_img.image_gs, cmap='gray')

# for i in range(np.shape(unsolved)[0]):
#     ax.scatter(mesh.nodes[unsolved[i],0], mesh.nodes[unsolved[i],1], marker='o', color='red')
# for i in range(np.shape(poor)[0]):
#     ax.scatter(mesh.nodes[poor[i],0], mesh.nodes[poor[i],1], marker='o', color='orange')
# # for i in range(np.shape(mesh.boundary_node_tags)[0]):
# #     ax.scatter(mesh.nodes[mesh.boundary_node_tags[i],0], mesh.nodes[mesh.boundary_node_tags[i],1], marker='o', color='blue')

# plt.show()