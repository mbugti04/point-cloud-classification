# This file is for processing the .3dm file type so that it may be usable.
# Mainly, it extracts the vertices from the meshes so that they may be used as 'point clouds'.
# These are saved into .json files.
# There are also helper functions to return the vertrices from the .json files.

import json
import matplotlib.pyplot as plt
import numpy as np
import rhino3dm
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import KNNGraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import gc
import memory_profiler



# Reads the .3dm file
# @memory_profiler.profile
def load_model(file_name):
     return rhino3dm.File3dm.Read(file_name)

# Extracts vertices from .3dm model
# @memory_profiler.profile
def process_3dm(model):
    combined_vertices = []
    if model is not None:
        for obj in model.Objects:
            geometry = obj.Geometry
            # objects in the NYC dataset are either breps or polylinecurves
            if isinstance(geometry, rhino3dm.Brep):
                brep = geometry
                vertices = [[vertex.Location.X, vertex.Location.Y, vertex.Location.Z] for vertex in brep.Vertices]
                for vertex in vertices:
                    combined_vertices.append(vertex)
    return combined_vertices

def save_to_json(vertices, file_name):
    vertices = vertices.tolist()
    with open(file_name, 'w') as f:
        json.dump(vertices, f)

def load_from_json(file_name):
    try:
        with open(file_name, 'r') as f:
            vertices = json.load(f)
        return vertices
    except FileNotFoundError:
        return False

# Labels for the different boroughs
classes = ["MN", "BX", "BK", "QN", "SI"]

# Converts a list of vertices into tensors
# which can be used for processing in pytorch
def vertex_to_data(vertices, label, k=6):
    data = Data(pos=torch.tensor(vertices, dtype=torch.float), y=torch.tensor([classes.index(label)]))
    return data

def get_all_models_from_json(samples=-1):
    file_directory = "data\\NYC\\"

    models = []
    for root, dirs, files in os.walk(file_directory):
        for file_name in files:
            if file_name.endswith(".json"):
                print("Loading vertices from file:", file_name)
                with open(os.path.join(root, file_name), 'r') as f:
                    vertices = json.load(f)
                    vertices = np.array(vertices)

                if samples != -1:
                    # if vertices has less than x vertices, pad with zeros
                    if len(vertices) < samples:
                        vertices += [[0, 0, 0]] * (samples - len(vertices))
                    # select x random vertices
                    vertices = vertices[np.random.choice(vertices.shape[0], samples, replace=False)]

                label = file_name[12:14]

                data = vertex_to_data(vertices, label, k=6)
                models.append(data)
    
    return models

# @memory_profiler.profile
# This method converts a .3dm file to .json
def process_models():
    # models = []
    file_directory = "data\\NYC\\"

    for root, dirs, files in os.walk(file_directory):
        for file_name in files:
            if file_name.endswith(".3dm"):
                json_file_name = os.path.join(root, file_name[:-4] + ".json")

                print("Obtaining vertices from 3dm file:", file_name)

                model = load_model(os.path.join(root, file_name))
                vertices = process_3dm(model)
                vertices = np.array(vertices)

                del model
                gc.collect()

                # Boolean to determine if to sample points when saving to file or not.
                # This saves space but loses information.
                # It is also possible to save all points but sample them at run-time
                #   so that you can vary the sample number without redownloading the entire dataset
                do_sample = False
                if do_sample:
                    samples = 8192
                    # if vertices has less than x vertices, pad with zeros
                    if len(vertices) < samples:
                        vertices += [[0, 0, 0]] * (samples - len(vertices))
                    # select x random vertices
                    vertices = vertices[np.random.choice(vertices.shape[0], samples, replace=False)]
                    # vertices = vertices[::100]


                save_to_json(vertices, json_file_name)

                # Delete the 3dm file after processing
                os.remove(os.path.join(root, file_name))