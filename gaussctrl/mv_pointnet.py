import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch_scatter import scatter_mean, scatter_max

class Pointnet(nn.Module):
    def __init__(self, c_dim=512, dim=3, hidden_dim=128, scatter_type='max', 
                unet=True, unet_kwargs={"depth": 4, "merge_mode": "concat", "start_filts": 32}, 
                plane_resolution=64, plane_type=['xz', 'xy', 'yz'], padding=0.1, n_blocks=5,
                inject_noise=False):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = self.coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

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

    def get_point_cloud_features(self, p):
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

        return c

    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat

    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane =='xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
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
        index = index[:, None, :]
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

    
