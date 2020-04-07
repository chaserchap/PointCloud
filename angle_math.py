import numpy as np
from scipy.spatial import distance

def get_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)

def vector_magnitude(a, b):
    """Magnitude of the vector (aka the distance) between two points."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_residual(a1, a2, p):
    """Calculate the distance of b from the line created between a1 and a2."""
    a, b, c = find_line_equation(a1, a2)
    return abs(a * p[0] + b * p[1] + c) / np.sqrt((a**2) + (b**2))

def find_line_equation(pt_a, pt_b):
    a = pt_b[1] - pt_a[1]
    b = pt_a[0] - pt_b[0]
    c = -(a*(pt_a[0]) + b*(pt_a[1]))

    return (a, b, c)

def get_extremus_points(arr):
    """Find the points that are farthest apart."""
    dists = distance.squareform(distance.pdist(arr))
    return np.unravel_index(np.argmax(dists), dists.shape)

def get_residual_np(arr, line=None):

    if line is None:
        extremus = get_extremus_points(arr)
        line = find_line_equation(arr[extremus[0]], arr[extremus[1]])

    return (np.sum(arr*line[0:2], axis=1) + line[2]) / np.sqrt((line[0]**2)+(line[1]**2))
