
import torch
import torch.nn.functional as F
from einops import rearrange
from .mv_modules.utils import back_projection, get_x_2d



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

def get_key_value(key_value, xy_l, xy_r, depth_query, depths, pose_rel, K, ori_h, ori_w, ori_h_r, ori_w_r, query_h, query_w):

    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

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
            depth_i[~mask] = 1000000
            depth_proj.append(depth_i) # ref depth
            
            mask_proj.append(mask*(depth_query>0))

            xy_proj.append(xy_l_rescale.clone())

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1 # map pixel idx to [-1,1]
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            _key_value = F.grid_sample( # use [-1,1] idx to extract the key
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    depth_proj = torch.stack(depth_proj, dim=1)
    mask_proj = torch.stack(mask_proj, dim=1)

    xy_proj = rearrange(xy_proj, 'b n h w c -> (b n) h w c')
    depth_proj = rearrange(depth_proj, 'b n h w -> (b n) h w')

    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    
    xy = torch.tensor(xy, device=key_value.device).float()[
        None].repeat(xy_proj.shape[0], 1, 1, 1)   
    
    xy_rel = (depth_query-depth_proj).abs()[...,None] # depth check

    xy_rel = rearrange(xy_rel, '(b n) h w c -> b n h w c', b=b)

    key_values = torch.stack(key_values, dim=1)
   
    return key_values, xy_rel, mask_proj


def get_query_value(query, key_value, xy_l, xy_r, depth_query, depths, pose_rel, K, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
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
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], xy_r[:, i], depth_query, depths[:, i], pose_rel[:, i], K,
                                               img_h_l, img_w_l, img_h_r, img_w_r, q_h, q_w)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask

def extract_camera_position_and_forward(view_matrix):
    # Ensure the view matrix is a 4x4 matrix
    assert view_matrix.shape == (4, 4), "View matrix must be 4x4"
    # Extract the rotation part (3x3) and translation part (3x1)
    rotation = view_matrix[:3, :3]  # Top-left 3x3
    translation = view_matrix[:3, 3]  # Top-right 3x1
    # Calculate the camera position in world space
    camera_position = -np.linalg.inv(rotation) @ translation
    # Extract the forward direction (negative z-axis of the rotation matrix)
    forward_direction = -rotation[:, 2]  # Third column of the rotation matrix

    return camera_position, forward_direction

def compute_ray_directions_from_projection_matrix(proj_matrix, h, w):
    # Create a grid of pixel coordinates
    x_indices, y_indices = np.indices((h, w))
    # Convert pixel coordinates to normalized device coordinates (NDC)
    # NDC coordinates range from -1 to 1
    x_ndc = (x_indices / (w - 1)) * 2 - 1  # Normalize to [-1, 1]
    y_ndc = (y_indices / (h - 1)) * 2 - 1  # Normalize to [-1, 1]
    # Create homogeneous coordinates for the pixel positions
    # The last row is set to 1 for homogeneous coordinates
    pixel_homogeneous = np.stack((x_ndc, y_ndc, np.ones_like(x_ndc)), axis=-1)  # Shape (h, w, 3)
    # Reshape to (h * w, 3) for matrix multiplication
    pixel_homogeneous = pixel_homogeneous.reshape(-1, 3).T  # Shape (3, h * w)
    # Invert the projection matrix
    proj_matrix_inv = np.linalg.inv(proj_matrix.cpu().numpy())
    # Compute ray directions in camera space
    ray_directions_homogeneous = proj_matrix_inv @ np.vstack((pixel_homogeneous, np.ones((1, pixel_homogeneous.shape[1])))) # Shape (4, h * w)
    # Convert from homogeneous to 3D coordinates
    ray_directions = ray_directions_homogeneous[:3] / ray_directions_homogeneous[3]  # Shape (3, h * w)
    # Reshape back to (h, w, 3)
    ray_directions = ray_directions.T.reshape(h, w, 3)
    # Normalize the ray directions
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)

    return ray_directions

def depth_map_world_to_ref(pts_world, proj_mat, view_mat):
    '''
    Args:
    pts_world [b,h,w,3] tensor
    proj_mat [4,4] tensor
    view_mat [b,m,4,4] tensor
    Returns:
    pts_proj [b,m,h,w,3] tensor
    depth_on_ref [b,m,h,w,1] tensor
    '''
    if pts_world.ndim == 3: # [h,w,3]
        pts_world = pts_world.unsqueeze(0) # [b,h,w,3] b=1
        proj_mat = proj_mat.unsqueeze(0) # [b,4,4] b=1
        view_mat = view_mat.unsqueeze(0) # [b,4,4] b=1
    b, h, w, _ = pts_world.shape
    
    pts_world = torch.cat((pts_world, 
                torch.ones((b, h, w, 1), dtype=pts_world.dtype, device=pts_world.device))
                ,dim=-1) # [b,h,w,4]
    pts_world = rearrange(pts_world, 'b h w c -> b (h w) c').unsqueeze(-1) # [b,hw,4,1]
    pts_view = view_mat[:,None,:,:] @ pts_world # [b,hw,4,1]
    pts_view /= pts_view[:,:,3:4,:] # divide by w
    pts_proj = proj_mat[:,None,:,:] @ pts_view
    pts_proj /= pts_proj[:,:,3:4,:] # divide by w
    cam_pos, _ = extract_camera_position_and_forward(view_mat.cpu().numpy())
    depth_on_ref = torch.mean((pts_world-cam_pos), dim=-1) # ?
    return pts_proj, depth_on_ref

def get_correspondance(depth_map, proj_mat, view_mat):
    '''
    Args:
    pts_world [b,h,w,3] tensor
    proj_mat [4,4] tensor
    view_mat [b,m,4,4] tensor
    Returns:
    correspondence tensor
    overlap_mask [b,m,h,w,1] tensor
    '''
    b,m,h,w = depth_map.shape
    overlap_ratios=torch.zeros(b, m, m, device=depths.device)
    correspondence = torch.zeros(b, m, m, h, w, 2, device=depth_map.device)
    
    pts_world = depth_map_screen_to_world(depth_map, proj_mat, view_mat)
    depth_list_on_ref = []
    
    for i in range(m):
        ori_pts_world = pts_world[:,i:i+1] # [b,1,h,w,3]
        pts_ij, depth_on_ref = depth_map_world_to_ref(ori_pts_world, proj_mat, view_mat)
        depth_list_on_ref.append(depth_on_ref)
        pts_ij = rearrange(pts_ij, '(b m) h w c -> b m h w c', b=b)
        correspondence[:, i] = point_ij
        mask=(point_ij[:,:,:,:,0]>=0)&(point_ij[:,:,:,:,0]<w)&(point_ij[:,:,:,:,1]>=0)&(point_ij[:,:,:,:,1]<h)
        mask=rearrange(mask, 'b m h w -> b m (h w)')
        overlap_ratios[:,i]=mask.float().mean(dim=-1)
    for b_i in range(b):
        for i in range(m):
            for j in range(i+1,m):
                overlap_ratios[b_i, i, j] = overlap_ratios[b_i, j, i]=min(overlap_ratios[b_i, i, j], overlap_ratios[b_i, j, i])
    overlap_mask=overlap_ratios>self.overlap_filter # filter image pairs that have too small overlaps
    cross_depths = torch.stack(depth_list_on_ref, dim=1) # [b,m,m,h,w,1]

def depth_map_screen_to_world(depth_map, proj_mat, view_mat):
    print(extract_camera_position_and_forward(view_mat.cpu().numpy()))
    if depth_map.ndim == 3: # [h,w,1]
        depth_map = depth_map.unsqueeze(0) # [b,h,w,1] b=1
        proj_mat = proj_mat.unsqueeze(0) # [b,4,4] b=1
        view_mat = view_mat.unsqueeze(0) # [b,4,4] b=1
    b, h, w, _ = depth_map.shape
    
    # (h, w, 3)
    for b_i in range(b):
        rays_d = torch.from_numpy(
                compute_ray_directions_from_projection_matrix(proj_mat[b_i],h, w)
            ).to(depth_map.device) 
    rays_d = rays_d[None,:,:,:].repeat(b,1,1,1) # [b,h,w,3]
    
    depth_map_view = rays_d * depth_map # [b,h,w,3]
    depth_map_view = torch.cat((depth_map_view, 
                                torch.ones((b, h, w, 1), dtype=rays_d.dtype, device=rays_d.device))
                               ,dim=-1) # [b,h,w,4]
    depth_map_view = rearrange(depth_map_view, 'b h w c -> b (h w) c').unsqueeze(-1) # [b,hw,4,1]
    
    # Normalize by w to get 3D coordinates in camera space
    inv_view_mat = torch.linalg.inv(view_mat)[:,None,:,:] # [b,1,4,4]

    # Convert camera space to world space
    depth_map_world = inv_view_mat.to(torch.float32) @ depth_map_view.to(torch.float32)  # Shape (b,hw, 4,1)
    depth_map_world /= depth_map_world[:,:,3:4,:] # divide by w
    depth_map_world = rearrange(depth_map_world.squeeze(-1), 'b (h w) c -> b h w c',h=h)
    return depth_map_world[:,:,:,:4] # [b,h,w,3]
