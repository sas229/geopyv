import h5py
import numpy as np
import os
from PIL import Image
import io

def recursively_load(obj, data):
    """
    Iterate through groups in a HDF5 file and save the groups and datasets names and attributes into the data dict.
    """
    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.keys():
            data[key] = recursively_load(obj[key], {})
    elif type(obj)==h5py._hl.dataset.Dataset:
        data = obj[()]
            
    return data


def recursively_save(data, f):
    """
    Iterate through groups in a HDF5 file and save the groups and datasets names and attributes into the data dict.
    """
    for key in data.keys():
        if type(data[key]) == dict:
            g = f.create_group(key)
            g = recursively_save(data[key], g)
        else:
            f.create_dataset(key, data=data[key])
            
    return f

def recursively_find_images(data, images):
    """
    Iterate through data dict and make a list of images and path for images.
    """
    for key in data.keys():
        if type(data[key]) == dict:
            images, path = recursively_find_images(data[key], images)
        elif key == "f_img":
            images.append(data[key])
        elif key == "g_img":
            images.append(data[key])
        elif key == "path":
            path = data[key]

    return images, path


def load(path, images=False):
    """
    Load data from HDF5 file.
    """
    data = {}
    with h5py.File(path,'r') as f:
        data = recursively_load(f['/'], data)
        if images == True:
            for filename in f["images"]:
                img = Image.open(io.BytesIO(f["images"][filename][0]))
                img.save(filename)

    return data


def save_to_HDF5(path, object, images=True):
    """
    Save data to HDF5 file.
    """
    if type(object) == dict:
        with h5py.File(path,'w') as f:
            f = recursively_save(object, f)
            if images == True:
                i, path = recursively_find_images(object, [])
                image_list = list(set(i))
                g = f.create_group("images")
                g.create_dataset("path", data=path)
                for filename in image_list:
                    fullpath = str(path) + "/" + str(filename)
                    f = open(fullpath, 'rb')
                    binary_data = f.read()
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    dset = g.create_dataset(filename, (100, ), dtype=dt)
                    dset[0] = np.fromstring(binary_data, dtype='uint8')

    else:
        raise TypeError("Invalid type passed to function. Object must be a dict.")

    return f