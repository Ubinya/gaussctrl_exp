import numpy as np
from PIL import Image
from PIL import ImageOps
import math
#import OpenEXR
import os
from gaussctrl.ad_noise import generate_uniform_points_in_cube,create_sphere,load_noise_from_npy

import pygame
from pygame.locals import *
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from OpenGL.GL.shaders import compileProgram, compileShader



def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    # Check for compilation errors
    compile_status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if compile_status != gl.GL_TRUE:
        info_log = gl.glGetShaderInfoLog(shader)
        raise RuntimeError(f"Shader compilation failed: {info_log}")

    return shader

def compile_program(vertex_shader, fragment_shader):
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL Error: {error}")

    # Check for linking errors
    link_status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if link_status != gl.GL_TRUE:
        info_log = gl.glGetProgramInfoLog(program)
        raise RuntimeError(f"Program linking failed: {info_log}")

    return program

def projection_matrix_splatfacto(fovx, fovy,znear=0.001, zfar=1000.0):
    """
    OpenGL-style perspective projection matrix
    but input range [n,f] ndc range [0,1]
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return np.array(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32
    )

def projection_matrix_ogl(fovx, fovy,znear=0.001, zfar=1000.0):
    """
    -n to -1, -f to 1
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return np.array(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, -(f + n) / (f - n), -2.0 * f * n / (f - n)],
            [0.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32
    )


def projection_matrix_dx(x,y,
                        near=0.1,far=100.0,
                        focal_len=50.0,sensor_width=36.0,):
    aspect_ratio = x / y
    fov = 2 * np.arctan((sensor_width / 2) / focal_len)
    near_clip = near
    far_clip = far
    f = 1.0 / np.tan(fov / 2)
    a = aspect_ratio
    projection_matrix = np.array([
        [f / a, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_clip + near_clip) / (near_clip - far_clip), (2 * far_clip * near_clip) / (near_clip - far_clip)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    return projection_matrix

def depth_np2texture(depth_np, print_range=False):
    depth_np = depth_np.astype(np.float32)
    if print_range:
        print(f'range of input depth npy: {np.min(depth_np)}, {np.max(depth_np)}')
    depth_np=np.clip(depth_np, 0.0, 1000.0) # clamp exr depth from 0.0 to 1000.0
    depth_np=np.flipud(depth_np)
    depth_np=np.fliplr(depth_np)
    #depth_np[depth_np==1000.0]=3.0
    #print(f"max:{np.max(depth_np)},min:{np.min(depth_np)}")

    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, depth_np.shape[1], depth_np.shape[0], 0, gl.GL_RED, gl.GL_FLOAT, depth_np)
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

def inv_mat(mat):
    if np.linalg.det(mat)== 0.0:
        print("inv_mat: mat not invertible.")
    else:
        return np.linalg.inv(mat)

def extract_camera_parameters(c2w_matrix):
    # Ensure the input is a numpy array
    c2w_matrix = np.array(c2w_matrix)
    # Extract the translation vector (camera position)
    cam_pos = c2w_matrix[:3, 3]
    # Extract the rotation matrix (R)
    R = c2w_matrix[:3, :3]
    # Extract the forward vector (negative of the third column of R)
    forward = -R[:, 2]
    # Extract the up vector (the second column of R)
    up = R[:, 1]
    return cam_pos, up, forward

def compute_view_matrix_ogl(cam_pos, up, forward):
    # Normalize the forward and up vectors
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)

    # Calculate the right vector
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    # Recalculate the up vector
    up = np.cross(forward, right)

    # Create the view matrix
    view_matrix = np.zeros((4, 4))
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up
    view_matrix[2, :3] = -forward
    view_matrix[3, 3] = 1

    # Apply translation to the view matrix
    view_matrix[0, 3] = -np.dot(right, cam_pos)
    view_matrix[1, 3] = -np.dot(up, cam_pos)
    view_matrix[2, 3] = np.dot(forward, cam_pos)

    return view_matrix

class MultiVeiwNoiseRenderer(object):
    def __init__(self, width, height,z_near=0.001,z_far=1000.0):

        self.width=width
        self.height=height
        self.z_near=z_near
        self.z_far=z_far

        self.cube_size = 2.0
        self.resolution = 100

        self.noise_threshold = 0.8
        self.noise_seed = 99
        self.noise_scale = 0.2
        #self.noise_unit_size = 0.02
        self.noise_unit_size = 0.015
        self.frag_depth_threshold = 0.016

        self.pts = None
        self.noise_np = None

        self.vis_idx = 0
        self.make_shader_source()

        self.setup_pyogl()

    def setup_pyogl(self):
        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
        glut.glutInitWindowSize(self.width, self.height)
        glut.glutCreateWindow(b"OpenGL Offscreen")
        glut.glutHideWindow()


    def setup_noise(self, noise_path):
        tmp_pts = generate_uniform_points_in_cube(self.cube_size,self.resolution)
        self.noise_np = load_noise_from_npy(noise_path,(self.resolution,self.resolution,self.resolution))
        filtered_indices=np.where(self.noise_np.flatten() > self.noise_threshold)[0]
        self.pts = tmp_pts[filtered_indices].astype(np.float32)

        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        #pygame.display.set_caption('Instanced Spheres')
        
        self.shader_vert = compile_shader(self.vertex_shader_source, gl.GL_VERTEX_SHADER)
        self.shader_frag = compile_shader(self.fragment_shader_source, gl.GL_FRAGMENT_SHADER)
        self.shader_program = compile_program(self.shader_vert, self.shader_frag)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # 创建球体模型
        self.vertices, self.indices = create_sphere(self.noise_unit_size, 8, 8)
        
        # Create a buffer for the sphere vertices
        vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)

        # Create a buffer for the sphere indices
        index_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)

        # create buffer for pts
        pts_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pts_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pts.nbytes, self.pts, gl.GL_STATIC_DRAW)

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
        
    def render_noise_SS(self, depth_np, mat_view, mat_proj,mat_c2w=None):
        """
        Args:
            depth_np: ndarray rendered by 3dgs, true distance from the cam in world coords.

        Returns:
        """
        mat_model = np.identity(4, dtype=np.float32)
        mat_unproj_tex = inv_mat(mat_proj)
        mat_view = self.c2w_to_viewmat(mat_c2w)

        # Load depth texture
        texture_id = depth_np2texture(depth_np,print_range=True)
        
        # Set up OpenGL context
        #gl.glDisable(gl.GL_DEPTH_TEST)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(self.shader_program)

        # Set uniform matrices
        model_loc = gl.glGetUniformLocation(self.shader_program, "model")
        view_loc = gl.glGetUniformLocation(self.shader_program, "view")
        projection_loc = gl.glGetUniformLocation(self.shader_program, "projection")
        gl.glUniformMatrix4fv(model_loc, 1, gl.GL_TRUE, mat_model)
        gl.glUniformMatrix4fv(view_loc, 1, gl.GL_TRUE, mat_view)
        gl.glUniformMatrix4fv(projection_loc, 1, gl.GL_TRUE, mat_proj)
        # Set up the perspective projection matrix

        # Draw instanced spheres
        gl.glBindVertexArray(self.vao)

        u_time_location = gl.glGetUniformLocation(shader_program, "u_depth_threshold")
        gl.glUniform1f(u_time_location, self.frag_depth_threshold)  # Pass the uniform arg

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "depthTexture"), 0)

        frag_unproj_loc = gl.glGetUniformLocation(self.shader_program, "frag_unprojection")
        gl.glUniformMatrix4fv(frag_unproj_loc, 1, gl.GL_TRUE, mat_unproj_tex)

        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None, len(self.pts))
        
        #gl.glReadBuffer(gl.GL_FRONT)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        
        # Convert the pixel data to a NumPy array
        rendered_img = Image.frombytes("RGBA", (self.width, self.height), pixels)
        #print(f"shape: {rendered_np.shape},max:{np.max(rendered_np)},min:{np.min(rendered_np)}")
        #mask_img = Image.fromarray(rendered_np)
        rendered_np = np.array(rendered_img)
        rendered_np=np.fliplr(rendered_np)
        rendered_np=np.flipud(rendered_np)
        
        ### maybe need to np.flipud()
        return rendered_np
        

    def save_result_mask(self, mask_np, save_path):
        save_img_name = f"{self.vis_idx}.png"
        self.vis_idx += 1
        os.makedirs(save_path, exist_ok=True)
        save_img_path = os.path.join(save_path,save_img_name)
        mask_img = Image.fromarray(mask_np)
        mask_img.save(save_img_path)
    
    def c2w_to_viewmat(self,mat_c2w):
        pos, up, fw = extract_camera_parameters(mat_c2w)
        mat_view = compute_view_matrix_ogl(pos, up, fw)
        print(f"pos:{pos},up:{up},fw:{fw}")
        return mat_view

    def binarize_mask(self, rgb_mask): # [h,w,3]
        # Step 1: Create a boolean mask where the RGB values are not [0, 0, 0]
        not_black_mask = np.any(mask_rgb != [0, 0, 0], axis=-1)
        # Step 2: Convert the boolean mask to integers (0s and 1s)
        binary_mask = not_black_mask.astype(np.uint8)
        # Step 3: Reshape to (h, w, 1) if needed
        binary_mask = binary_mask[:, :, np.newaxis]  # Add a new axis to make it (h, w, 1)
        return binary_mask

    def depth_np_ploter(self, depth_np,save_path):
        depth_img = depth_np.squeeze()  # Now shape is [h, w]

        # Step 2: Normalize the depth values to the range [0, 255]
        # Assuming depth values are in the range [0, max_depth]
        max_depth = np.max(depth_img)
        min_depth = np.min(depth_img)

        # Normalize to [0, 1]
        normalized_depth = (depth_img - min_depth) / (max_depth - min_depth)

        # Scale to [0, 255]
        scaled_depth = (normalized_depth * 255).astype(np.uint8)

        # Step 3: Convert to a PIL Image
        grayscale_image = Image.fromarray(scaled_depth)

        # Step 4: Save the grayscale image
        grayscale_image.save(save_path)

        print(f"Grayscale depth image saved as {save_path}")

    def make_shader_source(self):
        # Vertex shader source code
        self.vertex_shader_source ="""
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 instancePos;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec4 fragPos;
        out vec4 colorPos;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos + instancePos, 1.0);
            vec4 viewSpacePos = view * model * vec4(aPos + instancePos, 1.0);
            fragPos = projection * viewSpacePos;
            colorPos = vec4(((aPos + instancePos)-vec3(-1.0,-1.0,-1.0))/2.0,1.0);
            //gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
        """
        # Fragment shader source code
        self.fragment_shader_source = """
        #version 330 core
        in vec4 fragPos;
        in vec4 colorPos;
        out vec4 FragColor;
        uniform sampler2D depthTexture;
        uniform mat4 frag_unprojection;
        uniform mat4 tex_unprojection;
        uniform float u_depth_threshold; // Float uniform

        // depth texture back to
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
            //float depth_threshold = 0.05 + 0.0001;
            float depth_threshold = 0.016 + 0.0001;
            
            //Normalize depth to [0, 1] range for visualization
            //float near = 0.1; 
            //float far = 3.0; 
            //float normalizedDepth = (depth + near) / (far - near);

            // frag depth
            vec4 fragPos_ndc = fragPos / fragPos.w; // ndc range [-1,1] 
            vec4 clipSpacePos = vec4(fragPos_ndc.xyz, 1.0);
            vec4 viewSpacePos = frag_unprojection * clipSpacePos;
            viewSpacePos /= viewSpacePos.w;
            float view_frag_depth = length(viewSpacePos.xyz);

            //vec3 view_tex_pos = GetViewPosTex(gl_FragCoord.xy, tex_unprojection);
            
            // tex depth
            vec2 texCoords = fragPos_ndc.xy * 0.5 + 0.5; // Convert xy from [-1, 1] to [0, 1]
            float view_tex_depth = texture(depthTexture, texCoords).r; // true depth in view
            
            if ((view_tex_depth - view_frag_depth) < 0.0) {
                discard;
            }
            if ((view_tex_depth - view_frag_depth) > depth_threshold) {
                discard;
            }

            // --- for simple test
            //if (view_tex_depth > view_frag_depth) {
            //    discard;
            //}
            
            vec3 rgb=fragPos_ndc.xyz* 0.5 + 0.5;
            
            //vec3 rgb=vec3(gl_FragCoord.z/gl_FragCoord.w);
            
            FragColor = vec4(vec3(rgb), 1.0);
            FragColor = colorPos;
            // --- old test begin
            //float view_frag_depth = frag_projection[3].z/(depth * -2.0 + 1.0 - frag_projection[2].z);
            //float view_frag_depth_neg = (2*(gl_FragCoord.z/gl_FragCoord.w)-frag_projection[2].w)/(frag_projection[2].z-1.0);
            //float view_frag_depth = -view_frag_depth_neg;
            // --- old test end
            //FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
        }
        """