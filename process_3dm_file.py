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


classes = ["MN", "BX", "BK", "QN", "SI"]


# @memory_profiler.profile
def load_model(file_name):
     return rhino3dm.File3dm.Read(file_name)

# Pass in a .3dm model
# Returns a list of vertices
# @memory_profiler.profile
def process_3dm(model):
    combined_vertices = []

    if model is not None:
        for obj in model.Objects:
            geometry = obj.Geometry
            # objects in the NYC dataset are either breps or polylines(?)
            if isinstance(geometry, rhino3dm.Brep):
                brep = geometry
                faces = brep.Faces
                
                for face in faces:
                    mesh = face.GetMesh(rhino3dm.MeshType.Any)
                    if mesh is not None:
                        vertices = [[vertex.X, vertex.Y, vertex.Z] for vertex in mesh.Vertices]
                        for vertex in vertices:
                            combined_vertices.append(vertex)
        # if isinstance(geometry, rhino3dm.PolylineCurve):
        #     polylinecurve = geometry
        #     point_count = polylinecurve.PointCount
        #     for i in range(point_count):
        #         point = polylinecurve.Point(i)
        #         # print(point.X, point.Y, point.Z)
        #         combined_vertices.append([point.X, point.Y, point.Z]

    del model
    del geometry
    del brep
    del faces
    del mesh
    gc.collect()

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
    
def vertex_to_data(vertices, label, k=6):
    data = Data(pos=torch.tensor(vertices, dtype=torch.float), y=torch.tensor([classes.index(label)]))

    return data

def return_all_models_from_json():
    file_directory = "data\\NYC\\"

    models = []
    for root, dirs, files in os.walk(file_directory):
        for file_name in files:
            if file_name.endswith(".json"):
                print("Loading vertices from file:", file_name)
                with open(os.path.join(root, file_name), 'r') as f:
                    vertices = json.load(f)
                    vertices = np.array(vertices)
                    # vertices = vertices[np.random.choice(vertices.shape[0], 256, replace=False)]

                label = file_name[12:14]

                data = vertex_to_data(vertices, label, k=6)
                models.append(data)
    
    return models

# @memory_profiler.profile
def process_all_models():
    # models = []
    file_directory = "data\\NYC\\"

    for root, dirs, files in os.walk(file_directory):
        for file_name in files:
            if file_name.endswith(".3dm"):
                json_file_name = os.path.join(root, file_name[:-4] + ".json")
                if not load_from_json(json_file_name):
                    print("Obtaining vertices from 3dm file:", file_name)
                    model = load_model(os.path.join(root, file_name))
                    vertices = process_3dm(model)

                    del model
                    gc.collect()

                    samples = 8192
                    # if vertices has less than 256 vertices, pad with zeros
                    if len(vertices) < samples:
                        vertices += [[0, 0, 0]] * (samples - len(vertices))
                    vertices = np.array(vertices)
                    # select 256 random vertices
                    vertices = vertices[np.random.choice(vertices.shape[0], samples, replace=False)]
                    # vertices = vertices[::100]
                    save_to_json(vertices, json_file_name)
                    # Delete the 3dm file after processing
                    os.remove(os.path.join(root, file_name))
                    break

if __name__ == "__main__":
    file_directory = r"data\NYC"
    file_name = r"\NYC_3DModel_MN05"

    if not load_from_json(file_directory + file_name + ".json"):
        print("Loading vertices from 3dm file")

        model = load_model(file_directory + file_name + ".3dm")

        vertices = process_3dm(model)
        vertices = np.array(vertices)

        print("Vertex count: ", len(vertices))

        # Plot every x vertices
        x = 100
        vertices = vertices[::x]
        print("Vertex count: ", len(vertices))
        save_to_json(vertices, file_directory + file_name + ".json")
    else:
        print("Loading vertices from file")
        vertices = np.array(load_from_json(file_directory + file_name + ".json"))

    # graph = vertices_to_KNNGraph(vertices, k=6)
    # print(graph)

    data = vertex_to_data(vertices, k=6)

    if False:
        # plot vertices in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], s=1)
        plt.show()