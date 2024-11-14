import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from pygame.locals import *
import OpenGL.GLU as glu
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image
from BerlinNoise import generate_uniform_point_cloud,generate_perlin_noise,apply_noise_mask,normalize_noise
from utils import lookAt,perspective,compute_view_matrix,ortho
import pygame
from noise import pnoise3
import math
import OpenEXR
import Imath

def get_mat_view(mat_c2w):
    tmp_mat = np.zeros([4,4],dtype=mat_c2w.dtype)
    tmp_mat[:3,:4] = mat_c2w 
    if np.linalg.det(tmp_mat)== 0.0:
        mat_view = None
        print("blender proj mat not invertible.")
    else:
        mat_view = np.linalg.inv(tmp_mat)
    return mat_view.astype(np.float32)

def get_mat_proj(fx, fy, cx, cy, width, height, near, far):
    # 计算近平面上的坐标
    l = -cx * near / fx
    r = (width - cx) * near / fx
    b = -cy * near / fy
    t = (height - cy) * near / fy

    # 构建投影矩阵
    mat_proj = np.array([
        [2 * near / (r - l), 0, (r + l) / (r - l), 0],
        [0, 2 * near / (t - b), (t + b) / (t - b), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ])
    return mat_proj.astype(np.float32)
