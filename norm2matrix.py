import sys 
import numpy as np

def check_arg(args):
    if len(args) == 4:
        try:
            nx, ny, nz = float(args[1]), float(args[2]), float(args[3])
            return nx, ny, nz
        except ValueError:
            print("Arguments are set to float value.")
            print("Run command : python3 ./main.py nx ny nz")
            sys.exit(1)
    
    elif len(args) < 4:
        print("Arguments are too short.")
        print("Run command : python3 ./main.py nx ny nz")
        sys.exit(1)

    elif len(args) > 4:
        print("Arguments are too long.")
        print("Run command : python3 ./main.py nx ny nz")
        sys.exit(1)

def calc_matrix(normal_vector):
    # normalize
    normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    cos_theta = np.dot(normalized_normal_vector, np.array([0, 0, 1]))  
    theta = np.arccos(cos_theta)  

    rotation_axis = np.cross(normalized_normal_vector, np.array([0, 0, 1]))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    ux, uy, uz = rotation_axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    output_matrix = np.array([
    [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta, 0],
    [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta, 0],
    [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta), 0],
    [0, 0, 0, 1]])

    return output_matrix

def matrix_visualizer(raw_matrix):
    print("Transformation matrix\n---- copy below matrix ----")
    visualized_matrix = '\n'.join([' '.join(map(str, row)) for row in raw_matrix])
    print(visualized_matrix)

if __name__ == "__main__":
    args = sys.argv
    nx, ny, nz = check_arg(args)

    normal_vector = np.array([nx, ny, nz])
    matrix = calc_matrix(normal_vector)

    matrix_visualizer(matrix)
