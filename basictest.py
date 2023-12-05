import numpy as np
import geopyv as gp
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

# Subset test.
# Subset setup.
ref = gp.image.Image("./images/T-Bar/IMG_1062.jpg")
tar = gp.image.Image("./images/T-Bar/IMG_1071.jpg")
template = gp.templates.Circle(50)

# Subset instantiation.
subset = gp.subset.Subset(
    f_img=ref, g_img=tar, f_coord=np.asarray([1000, 1000]), template=template
)

# Subset inspection.
subset.inspect()

# Subset solving.
subset.solve()

# Other subset methods.
subset.convergence()

# Save subset to pyv file.
gp.io.save(object=subset, filename="test")
del subset

# Load subset test.
subset = gp.io.load(filename="test")

# Mesh test.
# Mesh setup.
ref = gp.image.Image("./images/ref.jpg")
tar = gp.image.Image("./images/tar.jpg")
template = gp.templates.Circle(50)
boundary_obj = gp.geometry.region.Path(
    nodes=np.asarray([[200.0, 200.0], [200.0, 800.0], [800.0, 800.0], [800.0, 200.0]]),
    hard=False,
)
exclusions_objs = []
exclusions_objs.append(
    gp.geometry.region.Circle(
        centre=np.asarray([700.0, 700.0]),
        radius=50.0,
        size=20.0,
        option="F",
        hard=True,
    )
)
seed = np.asarray([501, 501.0])
alpha = 0.2

# Mesh instantiation.
mesh = gp.mesh.Mesh(
    f_img=ref,
    g_img=tar,
    target_nodes=100,
    boundary_obj=boundary_obj,
    exclusion_objs=exclusions_objs,
    mesh_order=1,
)

# Mesh inspection.
mesh.inspect()
# mesh.inspect(subset_index=5000)
# Mesh saving : note, prior to solving, a geopyv object cannot be saved.
gp.io.save(object=mesh, filename="mesh")

# Mesh solving.
mesh.solve(
    seed_coord=seed,
    template=template,
    adaptive_iterations=3,
    method="ICGN",
    alpha=alpha,
    tolerance=0.7,
)

# Mesh saving.
gp.io.save(object=mesh, filename="mesh")
del mesh

# Mesh loading.
mesh = gp.io.load(filename="mesh")

# Other mesh methods (plot functionaility).
# The commands are basically standard matplotlib...
mesh.convergence()
mesh.convergence(quantity="norm")
mesh.convergence(quantity="iterations")
mesh.quiver()
mesh.contour(
    quantity="u",
    colorbar=False,
    alpha=0.75,
    levels=np.arange(-5, 6, 1),
    axis=False,
    xlim=((900, 2900)),
    ylim=((500, 2500)),
)
mesh.contour(quantity="C_ZNCC", mesh=True)
mesh.contour(quantity="iterations")
mesh.contour(quantity="R")

# You can also return the fig and ax objects and add customised items to the plot.
fig, ax = mesh.contour(quantity="v", alpha=1.0, levels=np.arange(-5, 6, 1))
ax.plot([0, 2000], [1000, 1000], color="k", linewidth=3.0, zorder=10)
ax.set_xlim((0, 2000))
plt.show()

# Let's inspect some clipped subsets.
mesh.inspect(subset_index=0)
mesh.inspect(subset_index=1)
mesh.inspect(subset_index=2)
mesh.inspect(subset_index=3)
mesh.inspect(subset_index=4)

# You can inspect subsets and their convergence
# via the mesh object by passing a subset number.
mesh.inspect(subset_index=0)
mesh.convergence(subset_index=0)

# If you supply a subset index that is out of range you get a ValueError.
mesh.convergence(subset_index=4000)

# Sequence test.
# Sequence setup.
template = gp.templates.Circle(50)
boundary_obj = gp.geometry.region.Path(
    nodes=np.asarray([[200.0, 200.0], [200.0, 800.0], [800.0, 800.0], [800.0, 200.0]]),
    hard=False,
)
exclusion_objs = []
exclusion_objs.append(
    gp.geometry.region.Circle(
        centre=np.asarray([700.0, 700.0]),
        radius=50.0,
        size=20.0,
        option="F",
        hard=True,
    )
)
seed = np.asarray([501, 501.0])
alpha = 0.2

# Sequence instantiation.
sequence = gp.sequence.Sequence(
    image_dir="images/sequence/",
    target_nodes=1000,
    boundary_obj=boundary_obj,
    exclusion_objs=exclusion_objs,
    save_by_reference=True,
    mesh_dir="images/meshes/",
)

# Sequence solving.
sequence.solve(
    seed_coord=seed,
    template=template,
    adaptive_iterations=3,
    method="ICGN",
    alpha=alpha,
    tolerance=0.7,
    sync=True,
)

# Sequence saving.
gp.io.save(object=sequence, filename="sequence")
del sequence

# Sequence loading.
sequence = gp.io.load(filename="sequence")

# Other sequence methods (plot functionality).
sequence.inspect(mesh_index=0)
sequence.inspect(mesh_index=3, subset_index=20)
sequence.convergence(mesh_index=1)
sequence.convergence(mesh_index=2, subset_index=4)
sequence.contour(mesh_index=1, quantity="R", mesh=True)
sequence.quiver(mesh_index=3)

# Calibration.
# So far, subset, mesh and sequence have been operating on the raw image data.
# To track in object space or undistorted image space, we need to calibrate.

# Calibration setup.

dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

# Calibration instantiation.
calibration = gp.calibration.Calibration(
    calibration_dir="images/T-Bar/Calibration/",
    dictionary=dictionary,
    board_parameters=[29, 18, 10, 8],
)

# Calibration solving.
calibration.solve(index=11, binary=110)

# Calibration saving.
gp.io.save(object=calibration, filename="calibration")
del calibration

# Calibration loading.
calibration = gp.io.load(filename="calibration")

# Calibration attributes.
print("Intrinsic Matrix: \n{}\n".format(calibration.data["intrinsic_matrix"]))
print("Extrinsic Matrix: \n{}\n".format(calibration.data["extrinsic_matrix"]))
print("Distortion parameters: \n{}\n".format(calibration.data["distortion"]))

# Calibration methods.
# Map from image space to object space.
imgpnts = np.asarray([[200.0, 100.0], [50.0, 75.0], [430.0, 1000.0]])
objpnts = calibration.i2o(imgpnts=imgpnts)
print("Object points: \n{}\n".format(objpnts))
# Map from object space to image space.
imgpnts = calibration.o2i(objpnts=objpnts)
print("Image points: \n{}\n".format(imgpnts))
# Calibrate objects.
# calibration.calibrate(object = subset)
# calibration.calibrate(object = mesh)
calibration.calibrate(object=sequence)
# Particle test.
# Particle setup.
coordinate_0 = np.asarray([1600.0, 1900.0])
objpnt = calibration.i2o(imgpnts=np.asarray([coordinate_0]))
# Particle instantiation.
particle = gp.particle.Particle(series=sequence, coordinate_0=objpnt[0, :2])

# Particle solving.
particle.solve()

# Particle saving.
gp.io.save(object=particle, filename="particle")
del particle

# Particle loading.
particle = gp.io.load(filename="particle")

# Other particle methods.
particle.trace(quantity="warps", component=1)

# Field test.
# Field setup.
target_particles = 500

# Field instantiation.
field = gp.field.Field(series=sequence, target_particles=target_particles, space="I")

# Field solving.
field.solve()

# Field saving.
gp.io.save(object=field, filename="field")
del field

# Field loading.
field = gp.io.load(filename="field")

# Other field methods.
field.trace(quantity="warps", component=2)
field.inspect()

# Extracting data.
component = 0
particle_index = 4

# Coordinates.
coordinates = field.data["particles"][particle_index]["results"]["coordinates"]
print(coordinates)

# Warps.
# Full.
warps = field.data["particles"][particle_index]["results"]["warps"]  # warp components :
# [u, v, dudx, dudy, dvdx, dvdy, d2udx2, d2udxdy, d2udy2, d2vdx2, d2vdxdy, d2vdy2]
print(warps)
warps = field.data["particles"][particle_index]["results"]["warps"][
    2:5, :2
]  # Time steps 2-5 for components 0 and 1 i.e. displacements.
print(warps)

# e.g. to extract volume progression. Note, area is in pixels.
volume_array = field.data["particles"][particle_index]["results"]["volumes"]
print(volume_array)
volume = field.data["particles"][particle_index]["results"]["volumes"][
    3
]  # At time step 3.
print(volume)
