import logging 
import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle, Square
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv import log
from geopyv import io
import tracemalloc
import pickle

def name_of_global_obj(xx):
    return [objname for objname, oid in globals().items() if id(oid)==id(xx)][0]

def list_global_objects():
    print(len([objname for objname, oid in globals().items()]))

level = logging.WARN
log.initialise(level)

# tracemalloc.start()

# Subset test.
ref = Image("./tests/ref.jpg")
tar = Image("./tests/tar.jpg")
# coord = np.asarray([300.,300.])
subset = Subset(f_img=ref, g_img=tar, f_coord=np.asarray([300,300]), template=Circle(25))
# subset = Subset(f_img=ref, g_img=tar)
subset.inspect()
subset.solve()
subset.convergence()

# subset_memory = tracemalloc.get_traced_memory()

# # Mesh test.
# template = Circle(25)
# boundary = np.asarray([[200.0, 200.0],[200.,800.],[800.,800.],[800.,200.]])
# mesh = Mesh(f_img=ref, g_img=tar, target_nodes=1000, boundary=boundary)
# mesh.solve(seed_coord=coord, template=template, adaptive_iterations=0, method = "ICGN")

# mesh_memory = tracemalloc.get_traced_memory()
# difference = mesh_memory[1]-subset_memory[1]

# print("\n")
# print("geopyv memory usage:\n")
# print("One subset: {:.2f} Mb".format(subset_memory[0]/1000000))
# print("Mesh (1000 subsets): {:.2f} Mb".format(mesh_memory[0]/1000000))
# print("\n")

# tracemalloc.stop()

# print(globals().items())
output = {
    "Subsets": [],
    }
for i in globals().copy().items():
    if isinstance(i[1], Subset) == True:
        name = i[0]
        s = i[1]
        data = {
            "f_img": s.f_img.filepath,
            "g_img": s.g_img.filepath,
            "settings": s.settings,
            "quality": s.quality,
            "results": s.results,
        }
        d = {name: data}
        output["Subsets"].append(d)


subset_data = subset.export()
print(subset_data["results"])

subset.save("test")
with open("test.pyv", "rb") as infile:
    pickled_data = pickle.load(infile)
print(pickled_data["quality"])

data = io.load("test")
print(data["settings"])

data = io.load("test")

io.save(subset, "test2")