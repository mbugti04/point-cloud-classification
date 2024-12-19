import trimesh
import matplotlib.pyplot as plt
import numpy as np
import rhino3dm

def load():
    combined_vertices = []
    # get file NYC_3DModel_MN01.3dm from local folder
    model = rhino3dm.File3dm.Read("NYC_3DModel_MN01.3dm")

    for obj in model.Objects:
            geometry = obj.Geometry
            if isinstance(geometry, rhino3dm.Brep):
                brep = geometry
                # face = brep.Faces[0]
                mesh = face.GetMesh(rhino3dm.MeshType.Any)

                # print (len(mesh.Faces))
                # print(mesh)
                vertices = [[vertex.X, vertex.Y, vertex.Z] for vertex in mesh.Vertices]
                # print(vertices)
                # faces = [face for face in mesh.Faces]
                # trimesh_mesh = trimesh.Trimesh(vertices=vertices)

                # for vertex in vertices:
                #     combined_vertices.append(vertex)

                print(brep)

                # break
                # return trimesh_mesh

                break

    return combined_vertices

load()