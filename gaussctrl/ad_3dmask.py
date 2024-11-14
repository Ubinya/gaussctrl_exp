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

vertex_shader_source ="""
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 instancePos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 fragPos;

void main()
{
    gl_Position = projection * view * model * vec4(aPos + instancePos, 1.0);
    fragPos = projection * view * model * vec4(aPos + instancePos, 1.0);
    //gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# Fragment shader source code
fragment_shader_source = """
#version 330 core
in vec4 fragPos;
out vec4 FragColor;
uniform sampler2D depthTexture;
uniform mat4 frag_unprojection;
uniform mat4 tex_unprojection;

// depth texture back to true world coordinates
vec3 GetViewPosTex(vec2 screen_uv, mat4 unprojection)
{
    vec2 texCoords = screen_uv * 0.5 + 0.5; // Convert from [-1, 1] to [0, 1]
    float depthValue = texture(depthTexture, texCoords).r;
    // needs test
	vec4 viewPos = unprojection * vec4(screen_uv.x, screen_uv.y, depthValue, 1);
	viewPos /= viewPos.w;
	viewPos.z = -viewPos.z; // default cam direction is negtive
	return viewPos.xyz;
}

void main()
{
    // Calculate depth in view space
    //float depth = fragPos.z;
    //float depth = gl_FragCoord.z;
    float depth_threshold = 0.05 + 0.0001;

    //Normalize depth to [0, 1] range for visualization
    //float near = 0.1; 
    //float far = 3.0; 
    //float normalizedDepth = (depth + near) / (far - near);


    vec4 fragPos_ndc = fragPos / fragPos.w; // ndc range [-1,1] 
    vec4 clipSpacePos = vec4(fragPos_ndc.xyz, 1.0);
    vec4 viewSpacePos = frag_unprojection * clipSpacePos;
    viewSpacePos /= viewSpacePos.w;
    float view_frag_depth = length(viewSpacePos.xyz);

    // no need for blender, depth tex has been true world coords
    //vec3 view_tex_pos = GetViewPosTex(gl_FragCoord.xy, tex_unprojection);  

    vec2 texCoords = fragPos_ndc.xy * 0.5 + 0.5; // Convert from [-1, 1] to [0, 1]
    float view_tex_depth = texture(depthTexture, texCoords).r; // true depth in view
    //float view_frag_depth = frag_projection[3].z/(depth * -2.0 + 1.0 - frag_projection[2].z);
    //float view_frag_depth_neg = (2*(gl_FragCoord.z/gl_FragCoord.w)-frag_projection[2].w)/(frag_projection[2].z-1.0);
    //float view_frag_depth = -view_frag_depth_neg;
    if (abs(view_tex_depth - view_frag_depth) > depth_threshold) {
        discard;
    }
    //if (view_frag_depth < view_tex_depth){
    //    discard;
    //}
    vec3 rgb=fragPos_ndc.xyz* 0.5 + 0.5;

    FragColor = vec4(vec3(rgb.rgb), 1.0);
    //FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
}
"""


def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(shader))
    return shader





def gen_gauss_pts(noise=False, num_pts=100000, mean=0.0, std_dev=0.4):
    points = np.random.normal(loc=mean, scale=std_dev, size=(num_pts, 3)).astype(np.float32)
    if noise:
        noise_scale = 0.05
        threshold = 0.8
        noise_values = np.array([pnoise3(p[0] * noise_scale, p[1] * noise_scale, p[2] * noise_scale) for p in points])
        min_val = np.min(noise_values)
        max_val = np.max(noise_values)
        normalized_values = (noise_values - min_val) / (max_val - min_val)
        filtered_pts = points[normalized_values > threshold]
        return filtered_pts
    else:
        return points


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


def get_camera_projection_matrix(x, y,
                                 near=0.1, far=100.0,
                                 focal_len=50.0, sensor_width=36.0, ):
    # 获取相机的宽高比
    aspect_ratio = x / y

    # 计算视场角
    fov = 2 * np.arctan((sensor_width / 2) / focal_len)

    # 获取裁剪平面
    near_clip = near
    far_clip = far

    # 计算投影矩阵
    f = 1.0 / np.tan(fov / 2)
    a = aspect_ratio

    projection_matrix = np.array([
        [f / a, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_clip + near_clip) / (near_clip - far_clip), (2 * far_clip * near_clip) / (near_clip - far_clip)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

    return projection_matrix


def load_depth_texture_png(image_path):
    image = Image.open(image_path).convert('L')
    image_data = np.array(image, dtype=np.float32) / 255.0
    print(f"shape of depth_np:{image_data.shape}")
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, image.width, image.height, 0, gl.GL_RED, gl.GL_FLOAT, image_data)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id


def load_depth_texture_exr(image_path):
    exr_file = OpenEXR.InputFile(image_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # 读取深度通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('V', FLOAT)
    # 将通道数据转换为numpy数组
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth_np = np.reshape(depth, (size[1], size[0]))
    depth_np = np.clip(depth_np, 0.0, 1000.0)  # clamp exr depth from 0.0 to 1000.0
    depth_np = np.flipud(depth_np)
    # depth_np[depth_np==1000.0]=3.0
    # print(f"max:{np.max(depth_np)},min:{np.min(depth_np)}")

    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, depth_np.shape[1], depth_np.shape[0], 0, gl.GL_RED, gl.GL_FLOAT,
                    depth_np)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id


def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    return program


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def compute_fov(focal_length, sensor_size=36):
    # default: width of the sensor, 36 mm (for horizontal FOV)
    # Compute the FOV in radians
    fov_rad = 2 * math.atan(sensor_size / (2 * focal_length))
    # Convert the FOV to degrees
    fov_deg = math.degrees(fov_rad)
    return fov_deg


def generate_uniform_points_in_cube(noise=False, noise_scale=0.1, threshold=0.8, seed=42,
                                    cube_size=2.0, resolution=50):

    if noise:

    else:
        return points.astype(np.float32)
class MultiViewConsistMask(object):
    def __init__(self):
        self.gen_3d_noise()
        self.set_cam()
        self.set_near_far()

    def set_near_far(self,near=0.1,far=100.0):
        self.near = near
        self.far = far

    def set_width_and_height(self, width, height, sensor_width=36.0, focal_length=50.0):
        self.width = width
        self.height = height
        self.sensor_width = sensor_width
        self.focal_length = focal_length

    def gen_mask_pts(self, cube_size=2.0, resolution=50, use_noise=False):
        self.resolution = resolution
        self.cube_size = cube_size
        x = np.linspace(-cube_size / 2, cube_size / 2, resolution)
        y = np.linspace(-cube_size / 2, cube_size / 2, resolution)
        z = np.linspace(-cube_size / 2, cube_size / 2, resolution)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        if use_noise:
            noised_pts = self.apply_noise_3d_mask(points,noise_scale=0.1,threshold=0.8)
            self.pts = noised_pts
        else:
            self.pts = points

    def apply_noise_3d_mask(self, input_pts, noise_scale=0.1, threshold=0.8, seed=42):
        ### size not tested !!!
        x_size, y_size, z_size = input_pts.shape[1], input_pts.shape[2], input_pts.shape[3]
        noise_values = np.zeros((x_size, y_size, z_size))

        for i in range(x_size):
            for j in range(y_size):
                for k in range(z_size):
                    x, y, z = grid[0][i][j][k], grid[1][i][j][k], grid[2][i][j][k]
                    noise_values[i][j][k] = snoise3(x * noise_scale, y * noise_scale, z * noise_scale, octaves=6)
        # normalize
        min_val = np.min(noise_values)
        max_val = np.max(noise_values)
        noise_values = (noise_values - min_val) / (max_val - min_val)
        # binarize the noise
        noise_mask = noise_values > threshold
        ### apply 3d mask not tested !!!
        grid_indices = ((input_pts + self.cube_size / 2) / self.cube_size * (self.resolution - 1)).astype(int)
        grid_indices = np.clip(grid_indices, 0, self.resolution - 1)
        mask = noise_mask[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
        return input_pts[mask].astype(np.float32)

    def create_sphere(self, radius, stacks, slices):
        sphere_vertices = []
        sphere_indices = []
        for stack in range(stacks + 1):
            phi = stack / stacks * np.pi
            for slice in range(slices + 1):
                theta = slice / slices * 2 * np.pi
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)
                sphere_vertices.extend([x, y, z])
        for stack in range(stacks):
            for slice in range(slices):
                first = (stack * (slices + 1)) + slice
                second = first + slices + 1
                sphere_indices.extend([first, second, first + 1, second, second + 1, first + 1])
        return np.array(sphere_vertices, dtype=np.float32), np.array(sphere_indices, dtype=np.uint32)

    def get_cam_proj_mat(self, projection_type):
        if projection_type == 'persp':
            aspect_ratio = self.width / self.height
            fov = 2 * np.arctan((self.sensor_width / 2) / self.focal_length)
            near_clip = self.near
            far_clip = self.far
            f = 1.0 / np.tan(fov / 2)
            a = aspect_ratio
            projection_matrix = np.array([
                [f / a, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far_clip + near_clip) / (near_clip - far_clip),
                 (2 * far_clip * near_clip) / (near_clip - far_clip)],
                [0, 0, -1, 0]
            ], dtype=np.float32)
            return projection_matrix
        elif projection_type == 'ortho':
            left = -half_size
            right = half_size
            bottom = -half_size
            top = half_size
            near_clip = z_near
            far_clip = z_far
            ### not implemented
            return ortho(left, right, bottom, top, near_clip, far_clip)

    def set_cam(self, pos=(1.5,1.5,1.5),distance=-1.0):
        self.camera_pos = np.array(pos)  # Camera position
        if distance > 0.0:
            # prepare point to render
            norms = np.linalg.norm(self.camera_pos, axis=0, keepdims=True)
            self.camera_pos = (camera_pos / norms) * distance

    def set_projection_type(self, projection_type='persp'):
        self.projection_type = projection_type

    def get_view_mat(self):
        """
            Compute the view matrix.

            Parameters:
            camera_position (np.ndarray): The position of the camera (3D vector).
            target (np.ndarray): The point the camera is looking at (3D vector).
            up_vector (np.ndarray): The up direction vector (3D vector).

            Returns:
            np.ndarray: The 4x4 view matrix.
            """
        # Compute the forward vector (z-axis)
        forward = normalize(self.camera_pos - target)
        # Compute the right vector (x-axis)
        right = normalize(np.cross(up_vector, forward))
        # Compute the true up vector (y-axis)
        up = np.cross(forward, right)
        # Create the rotation matrix
        rotation_matrix = np.array([
            [right[0], right[1], right[2], 0],
            [up[0], up[1], up[2], 0],
            [forward[0], forward[1], forward[2], 0],
            [0, 0, 0, 1]
        ])
        # Create the translation matrix
        translation_matrix = np.array([
            [1, 0, 0, -self.camera_pos[0]],
            [0, 1, 0, -self.camera_pos[1]],
            [0, 0, 1, -self.camera_pos[2]],
            [0, 0, 0, 1]
        ])
        # The view matrix is the product of the rotation and translation matrices
        view_matrix = rotation_matrix @ translation_matrix
        # Transpose the matrix to match OpenGL's column-major order
        view_matrix = view_matrix.T
        return view_matrix
    def render_mask_for_current_cam(self, mat_3dgs_view, mat_3dgs_proj, depth_npy):
        # --- prepare matrix
        mat_model = np.identity(4, dtype=np.float32)
        #look_at = np.array([0.0, 0.0, 0.0])  # Look-at point
        #up_vector = np.array([0.0, -1.0, 0.0])  # Up vector
        # mat_view = self.get_view_mat(look_at, up_vector)  # T ed
        mat_view = mat_3dgs_view
        # mat_proj = self.get_cam_proj_mat(self.projection_type).T
        mat_projection = mat_3dgs_proj

        # default test projection mat
        #mat_proj_blender = get_camera_projection_matrix(x=512, y=512, near=0.1, far=3)

        #
        if np.linalg.det(mat_3dgs_proj) == 0.0:
            print("proj mat from 3dgs not invertible.")
        else:
            mat_unprojection = np.linalg.inv(mat_3dgs_proj)

        # --- prepare shader
        shader_program = compileProgram(
            compileShader(vertex_shader_source, gl.GL_VERTEX_SHADER),
            compileShader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)
        )

        # --- prepare geometry
        # create mask sphere
        vertices, indices = create_sphere(sphere_size, 8, 8)

        # create buffer for pts
        pts_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pts_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pts.nbytes, pts, gl.GL_STATIC_DRAW)

        # Create a buffer for the sphere vertices
        vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # Create a buffer for the sphere indices
        index_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        # Bind vertex buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # Bind instance buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pts_buffer)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribDivisor(1, 1)  # Tell OpenGL this is an instanced vertex attribute

        # bind the index buffer
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index_buffer)

        # --- prepare ogl settings
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(shader_program)

        model_loc = gl.glGetUniformLocation(shader_program, "model")
        view_loc = gl.glGetUniformLocation(shader_program, "view")
        projection_loc = gl.glGetUniformLocation(shader_program, "projection")
        gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, mat_model)
        gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, mat_view)
        gl.glUniformMatrix4fv(projection_loc, 1, gl.GL_FALSE, mat_projection)

        # --- draw
        gl.glBindVertexArray(vao)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glUniform1i(gl.glGetUniformLocation(shader_program, "depthTexture"), 0)

        tex_unproj_loc = gl.glGetUniformLocation(shader_program, "tex_unprojection")
        frag_unproj_loc = gl.glGetUniformLocation(shader_program, "frag_unprojection")

        gl.glUniformMatrix4fv(tex_unproj_loc, 1, gl.GL_FALSE, mat_unproj_blender)
        gl.glUniformMatrix4fv(frag_unproj_loc, 1, gl.GL_FALSE, mat_unprojection)

        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(indices), gl.GL_UNSIGNED_INT, None, len(self.pts))



        # --- save result back to npy
        # Create a NumPy array to hold the pixel data
        # For RGB, we need 3 channels, and for RGBA, we need 4 channels
        pixel_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # For RGB
        # pixel_data = np.zeros((height, width, 4), dtype=np.uint8)  # For RGBA

        # Read the pixel data from the framebuffer
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixel_data)

        # The pixel data is now in the pixel_data NumPy array
        # Note: OpenGL reads pixels from the bottom-left corner, so you may need to flip the array
        pixel_data = np.flipud(pixel_data)  # Flip vertically if needed
        return pixel_data










