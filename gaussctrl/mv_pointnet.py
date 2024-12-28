import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch_scatter import scatter_mean, scatter_max
from einops import rearrange
import numpy as np
import math

class TriPlaneAttnProcessor:
    def __init__(self, self_attn_coeff, scatter_type='max', 
                unet=True, unet_kwargs={"depth": 4, "merge_mode": "concat", "start_filts": 32}, 
                plane_resolution=64, plane_type=['xz', 'xy', 'yz'], bbox_length=1.0, n_blocks=5,
                ):
        super().__init__()

        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.bbox_length = bbox_length

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
            
        self.self_attn_coeff = self_attn_coeff

    def __call__(self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask= None,
        temb= None,
        mv_scale= 1.0,
        #*args,
        **kwargs,
    ) -> torch.FloatTensor:
        # [b,h,w,3]
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
            
        input_ndim = hidden_states.ndim
        #input_ndim_3_to_4 = False
        
        if input_ndim == 4:
            hidden_states_mv = hidden_states
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        elif input_ndim == 3:
            batch_size, sequence_length, channel = hidden_states.shape
            height = math.sqrt(sequence_length)
            assert height.is_integer(), "t not sqrtable"
            height = int(height)
            width = height
            hidden_states_mv = rearrange(hidden_states,'b (h w) c -> b c h w',h=height)
            #input_ndim_3_to_4 = True
            
        if 'pts_world' in kwargs:
            print(f"shape of hidden_states_mv: {hidden_states_mv.shape}")
            hidden_states_mv_out = self.get_new_hidden_states_by_point_cloud(
                hidden_states=hidden_states_mv,
                pts=kwargs['pts_world']) # [b,c,h,w]
            hidden_states_mv_out = rearrange(hidden_states_mv_out,'b c h w -> b (h w) c')
            print(f"shape of hidden_states_mv_out: {hidden_states_mv_out.shape}")

        '''
        if input_ndim_3_to_4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2) # [b,t,c]
        '''
        
        if encoder_hidden_states is not None:
            batch_size, sequence_length, _ = encoder_hidden_states.shape
            
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        
        is_cross_attn = encoder_hidden_states is not None
        if encoder_hidden_states is None: # self
            encoder_hidden_states = hidden_states_mv_out
        elif attn.norm_cross: # cross and norm_cross
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        if is_cross_attn:
            query = attn.to_q(hidden_states)
            query = attn.head_to_batch_dim(query)
            key_cross = attn.to_k(encoder_hidden_states)
            key_cross = attn.head_to_batch_dim(key_cross)
            value_cross = attn.to_v(encoder_hidden_states)
            value_cross = attn.head_to_batch_dim(value_cross)
            
            attention_probs_cross = attn.get_attention_scores(query, key_cross, attention_mask)
            hidden_states = torch.bmm(attention_probs_cross, value_cross)
            print(f"shape of cross: {hidden_states.shape}")
        else:
            query = attn.to_q(hidden_states)
            query = attn.head_to_batch_dim(query)
            key_mv = attn.to_k(encoder_hidden_states)
            key_mv = attn.head_to_batch_dim(key_mv)
            value_mv = attn.to_v(encoder_hidden_states)
            value_mv = attn.head_to_batch_dim(value_mv)
            
            key_self = attn.to_k(hidden_states)
            key_self = attn.head_to_batch_dim(key_self)
            value_self = attn.to_v(hidden_states)
            value_self = attn.head_to_batch_dim(value_self)
            
            attention_probs_self = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs_self, value_self)
            
            attention_probs_mv = attn.get_attention_scores(query, key_mv, attention_mask)
            hidden_states_mv = torch.bmm(attention_probs_mv, value_mv)
            
            hidden_states = self.self_attn_coeff * hidden_states_self + (1-self.self_attn_coeff)* hidden_states_mv
            print(f"shape of self: {hidden_states.shape}")
        
        hidden_states = attn.batch_to_head_dim(hidden_states)
        print(f"shape of batch to head: {hidden_states.shape}")
        
        # linear proj
        #hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4: # [b,t=hw,c] to [b,c,h,w]
            hidden_states = rearrange(hidden_states,'b (h w) c -> b c h w',h=height) # hard coded

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
            
        
    def generate_plane_features(self, p, c, plane='xz'):
        # p [b,h,w,3] c [b,c,h,w]
        b, c_dim, h, w=c.shape
        c = rearrange(c,'b c h w -> b (h w) c') # [b,t,c]
        p = rearrange(p,'b h w c -> b (h w) c') # [b,t,3]
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane, bbox_length=self.bbox_length) # normalize to the range of (0, 1)
        # xy [b,t,2]
        index = self.coordinate2index(xy, self.reso_plane) # 1D index
        # index [b,1,t]
        #print(f"shape of index: {index.shape}")
        #print(f"range of index: {torch.min(index)}, {torch.max(index)}")

        # scatter plane features from points
        fea_plane = c.new_zeros(2*p.size(0), c.shape[2], self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x c x T
        fea_plane = scatter_mean(c, index.repeat(2,1,1), out=fea_plane) # B x c x reso^2
        fea_plane = fea_plane.reshape(2*p.size(0), c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x c x reso x reso)

        return fea_plane

    def forward_with_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        idx = int(plane_features.shape[1] / 3)
        fea = {}
        fea['xz'], fea['xy'], fea['yz'] = plane_features[:,0:idx,...], plane_features[:,idx:idx*2,...], plane_features[:,idx*2:,...]
        plane_feat_sum = 0

        plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
        plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)

    def forward(self, p, query):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        
        fea = {}
        plane_feat_sum = 0
        #denoise_loss = 0
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz') # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
            plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz') 
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
            plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)
    
    
        

    def get_new_hidden_states_by_point_cloud(self, 
                                hidden_states, # [b,c,h_l,w_l]
                                pts # [b,h,w,3]
                                ):
        #batch_size, T, D = pts.size()
        b, h, w, _ = pts.shape
        #print(f'shape of hidden states: {hidden_states.shape}')
        #exit()
        _, _, h_l, w_l = hidden_states.shape
        feature_pts = F.interpolate(hidden_states, size=(h, w), mode='bilinear', align_corners=False)
        #print_pointclout_centor_range(pts)
        #exit()
        
        fea = {}
        plane_feat_sum = 0
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(pts, feature_pts, plane='xz') # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
            plane_feat_sum += self.sample_plane_feature(pts, fea['xz'], 'xz') # [b,c,(hw)]
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(pts, feature_pts, plane='xy')
            plane_feat_sum += self.sample_plane_feature(pts, fea['xy'], 'xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(pts, feature_pts, plane='yz')
            plane_feat_sum += self.sample_plane_feature(pts, fea['yz'], 'yz')
        plane_feat_sum = rearrange(plane_feat_sum, 'b c (h w) -> b c h w', h=h)
        new_hidden_states = F.interpolate(plane_feat_sum, size=(h_l, w_l), mode='bilinear', align_corners=False)
        return new_hidden_states
        


    def sample_plane_feature(self, query, plane_feature, plane):
        query = rearrange(query,'b h w c -> b (h w) c') # [b,t=hw,3]
        xy = self.normalize_coordinate(query.clone(), bbox_length = self.bbox_length, plane=plane)
        #xy = xy[:, :, None].float() #[b,t=hw,1,2]
        xy = xy[:, :, None]#[b,t=hw,1,2]
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid.repeat(2,1,1,1), padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat # [b,c,t=hw]

    def normalize_coordinate(self, p, bbox_length=1.0, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'xz': # [B,T,3]
            xy = p[:, :, [0, 2]]
        elif plane =='xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        #### not tested for our mv framework!
        xy_new = xy / (bbox_length + 10e-6) # (-0.5, 0.5) 
        xy_new = xy_new + 0.5 # range (0, 1)

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new

    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            x (tensor): coordinate
            reso (int): defined resolution
            coord_type (str): coordinate type
        '''
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        index = index[:, None, :] # B,1,T
        return index

    # xy is the normalized coordinates of the point cloud of each plane 
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input 
    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

def depth_map_to_world(depth_map, proj_mat, view_mat):
    h, w, _ = depth_map.shape
    
    # Create a grid of pixel coordinates
    y_indices, x_indices = np.indices((h, w))
    
    # Normalize pixel coordinates to the range [0, 1]
    x_normalized = x_indices / (w - 1)  # Normalize x to [0, 1]
    y_normalized = y_indices / (h - 1)  # Normalize y to [0, 1]
    
    # Get the depth values and squeeze to remove the last dimension
    z_values = depth_map.squeeze()  # Shape (h, w)
    
    # Stack the normalized coordinates and depth values to create the local coordinates
    depth_map_ndc = np.stack((x_normalized, y_normalized, z_values), axis=-1)  # Shape (h, w, 3)
    one_channel = np.ones((h, w, 1), dtype=depth_map_ndc.dtype)
    depth_map_ndc = np.concatenate((depth_map_ndc, one_channel), axis=-1)

    inv_proj_mat = np.linalg.inv(proj_mat)

    # Convert homogeneous coordinates to camera space
    depth_map_view = depth_map_ndc @ inv_proj_mat.T  # Shape (h,w, 4)

    # Normalize by w to get 3D coordinates in camera space
    depth_map_view /= depth_map_view[:,:, 3:4]  # Shape (h,w, 4), broadcasting to divide by w
    inv_view_mat = np.linalg.inv(view_mat)

    # Convert camera space to world space
    depth_map_world = depth_map_view @ inv_view_mat.T  # Shape (h,w, 4)
    depth_map_world /= depth_map_world[:,:,3:4]


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

def print_pointclout_centor_range(pts):
    # pts [m]
    pts = pts.view(-1,3)
    center = pts.mean(dim=0)
    ranges = pts.max(dim=0).values - pts.min(dim=0).values
    print(f"center of pts: {center}, ranges: {ranges}")