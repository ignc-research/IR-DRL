import numpy as np

def generate_obstacle():
    # define the dimensions
    width = 0.05
    height = 0.05
    depth = 0.05
    # Define the position
    position = np.array([0.1, 0.55, 0.3])

    # Generate the vertices of the cube
    vertices = np.array([
     position + np.array([0, 0, 0]),
        position + np.array([width, 0, 0]),
        position + np.array([width, height, 0]),
        position + np.array([0, height, 0]),
        position + np.array([0, 0, depth]),
        position + np.array([width, 0, depth]),
        position + np.array([width, height, depth]),
        position + np.array([0, height, depth])
    ])

    # Define the faces of the cube
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (0,1,4),
        (1,4,5), 
        (0,4,3),
        (4,3,7),
        (7,3,2),
        (2,6,7),
        (1,2,6),
        (1,5,6),
        (4,6,7),
        (4,5,6)
  
    ]

    return vertices, faces
