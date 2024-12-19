
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np 

'''
def get_correspondence(depth, pose, K, x_2d):
    b, h, w = depth.shape
    x3d = back_projection(depth, pose, K, x_2d)
    x3d = rearrange(x3d, 'b h w c -> b c (h w)')
    x3d = K[:, :3, :3]@x3d
    x3d = rearrange(x3d, 'b c (h w) -> b h w c', h=h, w=w)
    x2d = x3d[..., :2]/(x3d[..., 2:3]+1e-6)

    mask = depth == 0
    x2d[mask] = -1000000
    x3d[mask] = -1000000

    return x2d, x3d
'''


def get_key_value(key_value, xy_l, depth_query, depths, ori_h, ori_w, ori_h_r, ori_w_r, query_h, query_w):

    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h # num_pixels per query unit
    key_scale = ori_h_r//h # num_pixxels per key unit

    # [b,h,w,...] downsample to 
    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5 # content: pixel coords of ori depth mapped to ref

    key_values = []

    xy_proj = []
    depth_proj = []
    mask_proj = []
    kernal_size = 3
    depth_query = depth_query[:, query_scale//2::query_scale,query_scale//2::query_scale]
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone() # cur img
            # displacement
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i # pixel offset on ref img
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j # pixel offset on ref img
            xy_l_rescale = (xy_l_norm+0.5)*key_scale # pixel idx to feature idx
            xy_l_round = xy_l_rescale.round().long() # round feature idx to int
            mask = (xy_l_round[..., 0] >= 0)*(xy_l_round[..., 0] < ori_w) * (
                xy_l_round[..., 1] >= 0)*(xy_l_round[..., 1] < ori_h) # mask feature img
            xy_l_round[..., 0] = torch.clamp(xy_l_round[..., 0], 0, ori_w-1) # clamp feature img
            xy_l_round[..., 1] = torch.clamp(xy_l_round[..., 1], 0, ori_h-1)

            depth_i = torch.stack([depths[b_i, xy_l_round[b_i, ..., 1], xy_l_round[b_i, ..., 0]]
                                  for b_i in range(b)]) # extract ref depth
            mask = mask*(depth_i > 0)
            depth_i[~mask] = 10000
            depth_proj.append(depth_i) # ref depth
            
            mask_proj.append(mask*(depth_query>0))

            xy_proj.append(xy_l_rescale.clone())

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1 # map pixel idx to [-1,1]
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            _key_value = F.grid_sample( # use pixel idx to retrieve key (like rasterization)
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    depth_proj = torch.stack(depth_proj, dim=1)
    mask_proj = torch.stack(mask_proj, dim=1)

    xy_proj = rearrange(xy_proj, 'b n h w c -> (b n) h w c')
    depth_proj = rearrange(depth_proj, 'b n h w -> (b n) h w')

    
    xy_rel = (depth_query-depth_proj).abs()[...,None] # depth check

    xy_rel = rearrange(xy_rel, '(b n) h w c -> b n h w c', b=b)

    key_values = torch.stack(key_values, dim=1)
   
    return key_values, xy_rel, mask_proj


def get_query_value(query, key_value, xy_l, depth_query, depths, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]

    key_values = []
    masks = []
    xys = []

    for i in range(m):
        _, _, q_h, q_w = query.shape
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], depth_query, depths[:, i],
                                               img_h_l, img_w_l, img_h_r, img_w_r, q_h, q_w)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask

def get_inv_norm_depth(depths): 
    depth_valid_mask = depths > 0
    depth_inv = 1. / (depths + 1e-6)
    depth_max = [depth_inv[i][depth_valid_mask[i]].max()
                    for i in range(depth_inv.shape[0])]
    depth_min = [depth_inv[i][depth_valid_mask[i]].min()
                    for i in range(depth_inv.shape[0])]
    depth_max = torch.stack(depth_max, dim=0)[:, None, None]
    depth_min = torch.stack(depth_min, dim=0)[:, None, None]  # [num_views, 1, 1]
    #print(f"{depth_inv.shape},{depth_min.shape}")
    depth_inv_norm_full = (depth_inv - depth_min) / \
        (depth_max - depth_min + 1e-6) * 2.0 - 1.0  # [-1, 1]
    depth_inv_norm_full[~depth_valid_mask] = -2.0
    return depth_inv_norm_full # [b,h,w]

def extract_camera_position_and_forward(mats_view): # [b,m,4,4]
    # Ensure the view matrix is a 4x4 matrix
    assert mats_view.shape[2:] == (4, 4), "View matrix must be 4x4"
    # Extract the rotation part (3x3) and translation part (3x1)
    rotation = mats_view[:,:,:3, :3]  # Top-left 3x3
    translation = mats_view[:,:,:3, 3:]  # Top-right 3x1
    # Calculate the camera position in world space
    camera_position = -torch.linalg.inv(rotation) @ translation
    # Extract the forward direction (negative z-axis of the rotation matrix)
    forward_direction = -rotation[:,:,:, 2]  # Third column of the rotation matrix

    return camera_position, forward_direction


def depth_map_screen_to_world(depth_map, mats_proj, mats_view):
    '''
    Args:
    depth_map [b,m,h,w] tensor
    mats_proj [b,m,4,4] tensor
    mats_view [b,m,4,4] tensor
    Returns:
    depth_world [b,m,h,w,3] tensor
    '''
    #print(extract_camera_position_and_forward(view_mat.cpu().numpy()))
    b, m, h, w = depth_map.shape
    
    depth_map = rearrange(depth_map, 'b m h w -> (b m) h w') # [bm,h,w]
    mats_view = rearrange(mats_view, 'b m h w -> (b m) h w') # [bm,4,4]
    # rays_d [h,w,3]
    rays_d = torch.from_numpy(
                compute_ray_directions(resolution=(h, w)) 
            ).to(depth_map.device) 
    rays_d = rays_d[None,:,:,:].repeat(depth_map.shape[0],1,1,1) # [bm,h,w,3]
    
    pts_view = rays_d * depth_map.unsqueeze(-1) # [bm,h,w,3]
    pts_view = torch.cat((pts_view, 
                        torch.ones((pts_view.shape[0], h, w, 1), dtype=rays_d.dtype, device=rays_d.device))
                        ,dim=-1) # [bm,h,w,4]
    pts_view = rearrange(pts_view, 'b h w c -> b (h w) c').unsqueeze(-1) # [bm,hw,4,1]
    
    # Normalize by w to get 3D coordinates in camera space
    inv_view_mat = torch.linalg.inv(mats_view)[:,None,:,:] # [bm,1,4,4]
    ###### ?????? validation not tested yet
    pts_world = inv_view_mat.to(torch.float32) @ pts_view.to(torch.float32)  # [bm,hw, 4,1]
    pts_world /= pts_world[:,:,3:4,:] # divide by w
    pts_world = rearrange(pts_world.squeeze(-1), 'b (h w) c -> b h w c',h=h)
    pts_world = rearrange(pts_world.squeeze(-1), '(b m) h w c -> b m h w c',m=m)
    return pts_world[:,:,:,:,:3] # [b,m,h,w,3]

def pts_world_to_ref(pts_world, mats_proj, mats_view):
    '''
    Args:
    pts_world [b,1,h,w,3] tensor
    mats_proj [b,m,4,4] tensor
    mats_view [b,m,4,4] tensor
    Returns:
    pts_proj [b,m,h,w,3] tensor
    depth_on_ref [b,m,h,w,1] tensor
    '''
    b, _, h, w, c = pts_world.shape
    m = mats_proj.shape[1]
    pts_world = pts_world.repeat(1,m,1,1,1) # [b,m,h,w,3] 
    
    pts_world = torch.cat((pts_world, 
                torch.ones((b,m, h, w, 1), dtype=pts_world.dtype, device=pts_world.device))
                ,dim=-1) # [b,m,h,w,4]
    
    pts_world = rearrange(pts_world, 'b m h w c -> b m (h w) c').unsqueeze(-1) # [b,m,hw,4,1]
    pts_view = mats_view[:,:,None,:,:] @ pts_world # [b,m,hw,4,1]
    pts_view /= pts_view[:,:,:,3:4,:] # divide by w
    pts_proj = mats_proj[:,:,None,:,:] @ pts_view
    pts_proj /= pts_proj[:,:,:,3:4,:] # divide by w
    pts_proj = pts_proj[:,:,:,:3,:] # [b,m,hw,3,1]
    ###### is first 2 dim xy????
    xy_2d_on_ref = rearrange(pts_proj.squeeze(-1), 'b m (h w) c -> b m h w c',h=h)[:,:,:,:,:2] 
    xy_2d_on_ref = (xy_2d_on_ref + 1.0) / 2.0 * h # only 2d to resolution size
    # compute depth on ref
    cam_pos, _ = extract_camera_position_and_forward(mats_view) # [b,m,3,1]
    # [b, m, hw, 3] 
    diff = pts_view.squeeze(-1)[:,:,:,:3] - cam_pos[:,:,None].squeeze(-1)
    depth_on_ref = (diff ** 2).sum(dim=-1)  # Shape: (b, m, hw)
    # Calculate the distances
    depth_on_ref = torch.sqrt(depth_on_ref)  # Shape: (b, m, hw)
    depth_on_ref = rearrange(depth_on_ref, 'b m (h w) -> b m h w',h=h)
    return xy_2d_on_ref, depth_on_ref # [b,m,h,w,2] [b,m,h,w]

def compute_ray_directions(resolution, fov_y_rad=0.8880228256678113, near=0.001, far=1000.0,):
    # Calculate aspect ratio
    width, height = resolution
    aspect_ratio = width / height
    
    # Calculate the height and width of the viewport
    viewport_height = 2 * np.tan(fov_y_rad / 2) * near
    viewport_width = viewport_height * aspect_ratio
    
    # Calculate the center of the viewport
    center_x = viewport_width / 2
    center_y = viewport_height / 2
    
    # Create an array to hold ray directions
    ray_directions = np.zeros((height, width, 3), dtype=np.float32)
    
    # Compute ray directions for each pixel
    for y in range(height):
        for x in range(width):
            # Normalized device coordinates (NDC)
            ndc_x = (x + 0.5) / width
            ndc_y = (y + 0.5) / height
            
            # Calculate the position in the viewport
            ray_x = (ndc_x * viewport_width) - center_x
            ray_y = (ndc_y * viewport_height) - center_y
            
            # Ray direction (assuming camera is looking down -Z axis)
            ray_directions[y, x] = np.array([ray_x, ray_y, -near])
    
    # Normalize the ray directions
    ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    
    return ray_directions # [h,w,3]


