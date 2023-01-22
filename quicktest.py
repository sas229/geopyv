import numpy as np
from geopyv.image import Image
from geopyv.templates import Square, Circle
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv.sequence import Sequence
from geopyv.particle import Particle
from geopyv.geometry.exclusions import circular_exclusion, circular_exclusion_list
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cv2

# Subset test.
#coord = np.asarray((100.0, 100.0))
#ref = Image("./images/Speckle/ref_sinus2.jpg")
#tar = Image("./images/Speckle/tar_sinus2.jpg")
#template = Circle(25)
#subset = Subset(coord, ref, tar, template)
#success = subset.pysolve(p_0=np.zeros(12), method="WFAGN")
#print(subset.D_f)
#print(subset.D_f_v2)
#subset.inspect()
#subset.convergence()


#ref = Image("./images/Speckle/ref_sinus2.jpg")
#tar = Image("./images/Speckle/tar_sinus2.jpg")
ref = Image("./images/Speckle/_u_0.jpg")
tar = Image("./images/Speckle/_u_1.jpg")
boundary = np.asarray([[100.0, 100.0],[100.,300.],[300.,300.],[300.,100.]])
seed = np.asarray([200.,200.])
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
p_0 = np.zeros(7)
p_0[-1] = 10000

subset = Subset(seed, ref, tar, template=Circle(20))
success = subset.solve()
print(success)
mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary)
mesh.solve(seed_coord=seed, template=Circle(20), adaptive_iterations=1, method = "ICGN")
particle = Particle(np.asarray([mesh]), coord = np.asarray([200,200]), p_init = np.zeros(12), vol=1)
particle.solve("MC", np.zeros(6), np.asarray([100, 10, 0.3, 0.5, 0.4, 0.]))
print(particle.coords)
print(particle.ps)
print(particle.stress_path)


#coords = np.empty((len(mesh.subsets),2))
#D_0s = np.empty(len(mesh.subsets))
#D_0_it_dist = np.empty((len(mesh.subsets), mesh.subsets[0].max_iterations))
#for i in range(len(mesh.subsets)):
#    D_0_it_dist[i] = mesh.subsets[i].D_0_log.flatten()
#    coords[i] = mesh.subsets[i].f_coord
#    D_0s[i] = mesh.subsets[i].p[-1]    
#
#print(D_0_it_dist)
#fig, ax = plt.subplots()
#ax.hist(D_0_it_dist[:,0], label = "1")
#ax.hist(D_0_it_dist[:,1], label = "2")
#ax.hist(D_0_it_dist[:,2], label = "3")
#plt.show()
#
#fig, ax = plt.subplots()
#ax.imshow(ref.image_gs, cmap="gist_gray")
#triangulation = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.triangulation)
#contours = ax.tripcolor(triangulation, D_0s, alpha=0.75, vmin = 0, shading='gouraud')
#ax.triplot(triangulation, linewidth=0.5, color='k', linestyle='-', alpha=0.25)
#ax.axis('equal')
#ax.set_axis_off()
#plt.colorbar(contours, ax=ax, fraction=0.02, pad=0.04)
#plt.show()
#plt.tight_layout()



#quantity = np.zeros(len(mesh.subsets))
#for i in range(len(quantity)):
#    quantity = abs(mesh.subsets[i].p[-1])
#
#fig, ax = plt.subplots()
#
#triangulation = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.triangulation)
#ax.imshow(ref.image_gs, cmap="gist_gray")
## contours = ax.tricontourf(triangulation, quantity, alpha=0.75, levels=levels, extend='both')
#contours = ax.tripcolor(triangulation, quantity, alpha=0.75, vmin=-10, vmax=10, shading='gouraud')
#ax.triplot(triangulation, linewidth=0.5, color='k', linestyle='-', alpha=0.25)
#ax.axis('equal')
#ax.set_axis_off()
#plt.colorbar(contours, ax=ax, fraction=0.02, pad=0.04)
#plt.show()
#plt.tight_layout()


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
#img_sequence = np.empty(3, dtype=object)
#for i in range(3):
#    img_sequence[i] = Image("./images/Uplift/IMG_"+str(30+i)+".jpg")
#template = Circle(25)
#boundary = np.asarray([[200.0, 1250.0],
#                        [200.0, 2900.0],
#                        [3850.0, 2900.0],
#                        [3850.0, 1250.0]])
#exclusions = []
#exclusions.append(circular_exclusion(np.asarray([1990, 2220]), radius=195, size=50))
#seed = np.asarray([300.0, 1500.0])
#sequence = Sequence(img_sequence = img_sequence, target_nodes=1000, boundary=boundary, exclusions=exclusions)
#sequence.solve(seed_coord=seed, max_iterations=50, max_norm=1e-5, tolerance=0.7, template=template, order=1, adaptive_iterations=3, alpha=0.1, beta=10)
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
#poor = np.flatnonzero(np.where(mesh.C_ZNCC < 0.7, mesh.solved,0))
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
# Plot.
fig, ax = plt.subplots()
triangulation = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.triangulation)
ax.imshow(ref.image_gs, cmap="gist_gray")
# contours = ax.tricontourf(triangulation, quantity, alpha=0.75, levels=levels, extend='both')
contours = ax.tripcolor(triangulation, quantity, alpha=0.75, vmin=-10, vmax=10, shading='gouraud')
ax.triplot(triangulation, linewidth=0.5, color='k', linestyle='-', alpha=0.25)
ax.axis('equal')
ax.set_axis_off()
plt.colorbar(contours, ax=ax, fraction=0.02, pad=0.04)
plt.show()
plt.tight_layout()
# 
# # Plot seed, unsolved and poor subsets as coloured circles. 
# ax.invert_yaxis()
# ax.scatter(mesh.nodes[mesh.seed_node,0], mesh.nodes[mesh.seed_node,1], marker='o', color='green')
# for i in range(np.shape(unsolved)[0]):
#     ax.scatter(mesh.nodes[unsolved[i],0], mesh.nodes[unsolved[i],1], marker='o', color='red')
# for i in range(np.shape(poor)[0]):
#     ax.scatter(mesh.nodes[poor[i],0], mesh.nodes[poor[i],1], marker='o', color='orange')
# for i in range(np.shape(mesh.boundary_node_tags)[0]):
#     ax.scatter(mesh.nodes[mesh.boundary_node_tags[i],0], mesh.nodes[mesh.boundary_node_tags[i],1], marker='o', color='blue')




