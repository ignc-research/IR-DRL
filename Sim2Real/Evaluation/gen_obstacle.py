import numpy as np

def generate_obstacle():
    # Define the obstacle dimensions and position
    length = 0.1
    width = 0.1
    height = 0.1
    position = np.array([0.2, 0.5, 0.3])

    # Generate the vertices of the rectangular prism obstacle
    vertices = np.array([
        position + np.array([0, 0, 0]),
        position + np.array([length, 0, 0]),
        position + np.array([length, width, 0]),
        position + np.array([0, width, 0]),
        position + np.array([0, 0, height]),
        position + np.array([length, 0, height]),
        position + np.array([length, width, height]),
        position + np.array([0, width, height])
    ])

    # Define the faces of the rectangular prism obstacle
    faces = [
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (0, 1, 2, 3),
        (4, 5, 6, 7)
    ]

    return vertices, faces
