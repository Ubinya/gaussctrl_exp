from noise import pnoise3
import numpy as np
import os

def create_sphere(radius, stacks, slices):
    vertices = []
    indices = []

    for stack in range(stacks + 1):
        phi = stack / stacks * np.pi
        for slice in range(slices + 1):
            theta = slice / slices * 2 * np.pi
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            vertices.extend([x, y, z])

    for stack in range(stacks):
        for slice in range(slices):
            first = (stack * (slices + 1)) + slice
            second = first + slices + 1
            indices.extend([first, second, first + 1, second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_sphere0(radius, subdivisions=16):
    vertices = []
    indices = []

    for i in range(subdivisions + 1):
        lat = np.pi * (-0.5 + float(i) / subdivisions)
        for j in range(subdivisions + 1):
            lon = 2 * np.pi * float(j) / subdivisions
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            vertices.append([x, y, z])

    for i in range(subdivisions):
        for j in range(subdivisions):
            first = i * (subdivisions + 1) + j
            second = first + subdivisions + 1
            indices.append(first)
            indices.append(second)
            indices.append(first + 1)
            indices.append(second)
            indices.append(second + 1)
            indices.append(first + 1)

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)




def generate_uniform_points_in_cube(cube_size=2.0,resolution=50):
    """
    Args:
        cube_size: boundary of noise points
        resolution: resolution per axis
    Return:
        points npy with shape [resolution^3, 3]
    """
    x = np.linspace(-cube_size / 2, cube_size / 2, resolution)
    y = np.linspace(-cube_size / 2, cube_size / 2, resolution)
    z = np.linspace(-cube_size / 2, cube_size / 2, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return points
    
def gen_perlin_noise(
                    noise_shape, 
                    scale=0.05, 
                    octaves=1,
                    persistence=1.0,
                    lacunarity=2.0,
                    seed=42,
                    normalize=True):
    """
    Args:
        scale: Scale of the noise
        octaves: Number of layers to composite the noise
        persistence: to favor lower or higher level of octaves
        lacunarity: frequncy times for adjacent octaves
        seed: random seed
    Return:
    """
    noise_data = np.zeros(noise_shape)

    for x in range(noise_shape[0]):
        for y in range(noise_shape[1]):
            for z in range(noise_shape[2]):
                noise_data[x][y][z] = pnoise3(x * scale, 
                                                y * scale, 
                                                z * scale, 
                                                octaves=octaves, 
                                                persistence=persistence, 
                                                lacunarity=lacunarity, 
                                                repeatx=1024, 
                                                repeaty=1024, 
                                                repeatz=1024, 
                                                base=seed)  # Set the seed here
    if normalize:
        min_val = np.min(noise_data)
        max_val = np.max(noise_data)
        normalized_noise = (noise_data - min_val) / (max_val - min_val)
        return normalized_noise
    else:
        return noise_data

## TODO
def load_noise_from_npy(noise_npy_path, 
                        noise_shape,
                        scale=0.05, 
                        octaves=1,
                        persistence=1.0,
                        lacunarity=2.0,
                        seed=42,
                        normalize=True):
    # Check if arg is not None and is a string
    if noise_npy_path is not None and isinstance(noise_npy_path, str):
        if os.path.exists(noise_npy_path):
            print(f"noise file '{noise_npy_path}' exists. reading")
            noise_np = np.load(noise_npy_path).astype(np.float32)
            assert noise_np.shape == noise_shape, "noise_np shape not match"
            return noise_np
        else:
            print(f"noise '{noise_npy_path}' not exists. generating new noise and save to '{noise_npy_path}'")
            return gen_perlin_noise(noise_shape,scale,octaves,persistence,lacunarity,seed,normalize)
    else:
        print("noise_npy_path is None or not a string.")
        return False


## seems useless?
def apply_noise_mask(input_pts, noise_mask, resolution, cube_size):
    """Apply the noise mask to the point cloud."""
    # Map points to the noise grid
    grid_indices = ((input_pts + cube_size / 2) / cube_size * (resolution - 1)).astype(int)
    grid_indices = np.clip(grid_indices, 0, resolution - 1)
    mask = noise_mask[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
    return input_pts[mask]

def gen_gauss_pts(noise=False,num_pts=100000,mean=0.0,std_dev=0.4):
    points=np.random.normal(loc=mean, scale=std_dev, size=(num_pts, 3)).astype(np.float32)
    if noise:
        noise_scale = 0.05 # Scale of the noise
        octaves = 1 # Number of layers to composite the noise
        persistence = 1.0 # to favor lower or higher level of octaves
        lacunarity = 2.0 # frequncy times for adjacent octaves
        threshold = 0.8
        noise_seed = 100

        noise_values=np.array([pnoise3(p[0] * noise_scale,
                                       p[1] * noise_scale,
                                       p[2] * noise_scale,
                                       octaves=octaves, 
                                       persistence=persistence, 
                                       lacunarity=lacunarity, 
                                       repeatx=1024, 
                                       repeaty=1024, 
                                       repeatz=1024, 
                                       base=noise_seed
                                      ) for p in points])
        min_val = np.min(noise_values)
        max_val = np.max(noise_values)
        normalized_values = (noise_values - min_val) / (max_val - min_val)
        filtered_pts = points[normalized_values > threshold]
        return filtered_pts
    else:
        return points