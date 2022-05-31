import numpy as np
from geopyv.image import Image
from geopyv.templates import Square, Circle
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv.sequence import Sequence
from geopyv.geometry.exclusions import circular_exclusion, circular_exclusion_list
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cv2

# # Subset test.
# coord = np.asarray((100.0, 100.0))
# ref = Image("./images/Uplift/IMG_30.jpg")
# tar = Image("./images/Uplift/IMG_38.jpg")
# template = Circle(25)
# subset = Subset(coord, ref, tar, template)
# success = subset.solve(p_0=np.zeros(12), method="ICGN")
# subset.inspect()
# subset.convergence()

# Uplift mesh test.
#ref = Image("./images/Uplift/IMG_30.jpg")
#tar = Image("./images/Uplift/IMG_38.jpg")
#template = Circle(25)
#boundary = np.asarray([[200.0, 1250.0],
#                        [200.0, 2900.0],
#                        [3850.0, 2900.0],
#                        [3850.0, 1250.0]])
#exclusions = []
#exclusions.append(circular_exclusion(np.asarray([1990, 2220]), radius=195, size=50))
#seed = np.asarray([300.0, 1500.0])

# Sequence test.
img_sequence = np.empty(3, dtype=object)
for i in range(3):
    img_sequence[i] = Image("./images/Uplift/IMG_"+str(30+i)+".jpg")
template = Circle(25)
boundary = np.asarray([[200.0, 1250.0],
                        [200.0, 2900.0],
                        [3850.0, 2900.0],
                        [3850.0, 1250.0]])
exclusions = []
exclusions.append(circular_exclusion(np.asarray([1990, 2220]), radius=195, size=50))
seed = np.asarray([300.0, 1500.0])
sequence = Sequence(img_sequence = img_sequence, target_nodes=1000, boundary=boundary, exclusions=exclusions)
sequence.solve(seed_coord=seed, max_iterations=50, max_norm=1e-5, tolerance=0.7, template=template, order=1, adaptive_iterations=3, alpha=0.1, beta=10)
# # Test 48.
# ref = Image("./images/Test48/IMG_30.jpg")
# tar = Image("./images/Test48/IMG_35.jpg")
# template = Circle(25)
# boundary = np.asarray([[100.0, 1350.0],
#                         [100.0, 2900.0],
#                         [3950.0, 2900.0],
#                         [3950.0, 1350.0]])
# exclusions = []
# exclusions.append(circular_exclusion(np.asarray([2880, 2035]), radius=130, size=50))
# seed = np.asarray([300.0, 1500.0])


# # Test 102.
# ref = Image("./images/Test102/IMG_400.jpg")
# tar = Image("./images/Test102/IMG_405.jpg")
# template = Circle(25)
# boundary = np.asarray([[100.0, 900.0],
#                         [100.0, 2900.0],
#                         [3950.0, 2900.0],
#                         [3950.0, 900.0]])
# exclusions = []
# exclusions.append(circular_exclusion(np.asarray([2568, 2019]), radius=130, size=50))
# seed = np.asarray([300.0, 1500.0])

## Instantiate ad solve mesh.
#mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary, exclusions=exclusions)
#mesh.solve(seed_coord=seed, max_iterations=50, max_norm=1e-5, tolerance=0.7, template=template, order=1, adaptive_iterations=3, alpha=0.1, beta=10)
#
## Analyse and plot data (to be moved into Mesh as methods...).
#unsolved = np.flatnonzero(np.where(mesh.solved > -1, mesh.solved,0))
#poor = np.flatnonzero(np.where(mesh.C_CC < 0.7, mesh.solved,0))
#
## Get average nodal strains (should be a better way to do this...).
#node_strains = np.zeros((np.shape(mesh.nodes)[0], 3))
#for i in range(np.shape(mesh.nodes)[0]):
#    elements = np.argwhere(np.any(mesh.triangulation == i, axis=1)==True).flatten()
#    sum_strain = 0
#    count = 0
#    for j in range(np.shape(elements)[0]):
#        sum_strain += mesh.strains[elements[j]]
#        count += 1
#    node_strains[i] = sum_strain/count
#
## Plot quantity and levels.
#quantity = node_strains[:,2]*100 # Percentage strains.
##quantity = mesh.displacements[:,1]
## max = np.round(np.ceil(np.max(np.abs(quantity))),0)
## levels = np.arange(-max, max+0.5, 0.5)
#levels = np.arange(-10, 10+0.5, 0.5)
#
## Plot.
#fig, ax = plt.subplots()
#triangulation = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.triangulation)
#ax.imshow(ref.image_gs, cmap="gist_gray")
#contours = ax.tricontourf(triangulation, quantity, alpha=0.75, levels=levels, extend='both')
#ax.triplot(triangulation, linewidth=0.5, color='k', linestyle='-', alpha=0.25)
#ax.axis('equal')
#ax.set_axis_off()
#plt.colorbar(contours, ax=ax)
#plt.show()
#plt.tight_layout()

# # Plot seed, unsolved and poor subsets as coloured circles. 
# ax.invert_yaxis()
# ax.scatter(mesh.nodes[mesh.seed_node,0], mesh.nodes[mesh.seed_node,1], marker='o', color='green')
# for i in range(np.shape(unsolved)[0]):
#     ax.scatter(mesh.nodes[unsolved[i],0], mesh.nodes[unsolved[i],1], marker='o', color='red')
# for i in range(np.shape(poor)[0]):
#     ax.scatter(mesh.nodes[poor[i],0], mesh.nodes[poor[i],1], marker='o', color='orange')
# for i in range(np.shape(mesh.boundary_node_tags)[0]):
#     ax.scatter(mesh.nodes[mesh.boundary_node_tags[i],0], mesh.nodes[mesh.boundary_node_tags[i],1], marker='o', color='blue')




