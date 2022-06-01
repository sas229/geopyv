import h5py
import numpy as np

# def descend_obj(obj,sep='\t'):
#     """
#     Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes.
#     """
#     if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
#         for key in obj.keys():
#             print(sep,'-',key,':',obj[key])
#             descend_obj(obj[key],sep=sep+'\t')
#     elif type(obj)==h5py._hl.dataset.Dataset:
#         for key in obj.attrs.keys():
#             print(sep+'\t','-',key,':',obj.attrs[key])

# def h5dump(path,group='/'):
#     """
#     print HDF5 file metadata

#     group: you can give a specific group, defaults to the root group.
#     """
#     with h5py.File(path,'r') as f:
#          descend_obj(f[group])

# h5dump('test.h5')

def descend(obj, data):
    """
    Iterate through groups in a HDF5 file and save the groups and datasets names and datasets attributes into the data dict.
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.keys():
            if key == "subsets":
                print("Processing subsets...")
                n = len(obj[key])
                subset_array = np.empty(n, dtype=dict)
                # Fill numpy array with dict of subset data.
                for i in range(n):
                    subset_object = obj[key][str(i+1)]
                    subset_array[i] = {}
                    for subset_key in subset_object.keys():
                        subset_array[i][subset_key] = subset_object[subset_key][()]
                # Assign array to key.
                data[key] = subset_array
            else:
                data[key] = descend(obj[key], {})
            
    elif type(obj)==h5py._hl.dataset.Dataset:
        data = obj[()]
            
    return data

def h5load(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group.
    """
    data = {}
    with h5py.File(path,'r') as f:
        data = descend(f[group], data)

    return data

data = h5load('test.h5')

print(list(data.keys()))
print(list(data["IMG_30_IMG_38"].keys()))
# print(data["IMG_30_IMG_38"]["subsets"])
print(data["IMG_30_IMG_38"]["subsets"][0]["f_coord"][0])
# print(data["IMG_30_IMG_38"]["subsets"]["999"]["C_CC"])

print(data["IMG_30_IMG_38"]["triangulation"])

with h5py.File('test.h5','r') as f:
    C_CC_999 = f["IMG_30_IMG_38"]["subsets"]["999"]["C_CC"][()]
    f_coords_999 = f["IMG_30_IMG_38"]["subsets"]["999"]["f_coords"][()]
print(C_CC_999)
print(np.shape(f_coords_999))

# hf = h5py.File('test.h5', 'r')
# nodes = hf["IMG_30_IMG_38"]["nodes"][:]
# print(nodes)
# print(type(nodes))
# print(np.shape(nodes))

# def descend(obj):
#     if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
#         print("Group")
#         for key in obj.keys():
#             descend(obj[key])
#     elif type(obj)==h5py._hl.dataset.Dataset:
#         print("Dataset")

# with h5py.File('test.h5','r') as f:
#     descend(f['/'])
#     # for inner_key in hf[key].keys():
#     #     print(inner_key)