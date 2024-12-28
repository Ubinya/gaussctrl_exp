# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GaussCtrl Pipeline and trainer"""

import os
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type, List
from rich.progress import Console
from copy import deepcopy
import numpy as np 
from PIL import Image
import mediapy as media
from lang_sam import LangSAM

import torch, random
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from gaussctrl.gc_datamanager import (
    GaussCtrlDataManagerConfig,
)
from diffusers.models.attention_processor import AttnProcessor
from gaussctrl import utils
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
import yaml

from gaussctrl.ad_render import MultiVeiwNoiseRenderer
from gaussctrl.mv_model import get_inv_norm_depth
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

from gaussctrl.mv_generator import DepthGenerator
from gaussctrl.mv_pointnet import TriPlaneAttnProcessor
import cv2

CONSOLE = Console(width=120)

@dataclass
class GaussCtrlPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussCtrlPipeline)
    """target class to instantiate"""
    datamanager: GaussCtrlDataManagerConfig = GaussCtrlDataManagerConfig()
    """specifies the datamanager config"""
    render_rate: int = 500
    """how many gauss steps for gauss training"""
    prompt: str = ""
    """Positive Prompt"""
    langsam_obj: str = ""
    """The object to be edited"""
    guidance_scale: float = 5
    """Classifier Free Guidance"""
    num_inference_steps: int = 20
    """Inference steps"""
    chunk_size: int = 5
    """Batch size for image editing, feel free to reduce to fit your GPU"""
    ref_view_num: int = 4
    """Number of reference frames"""
    diffusion_ckpt: str = 'CompVis/stable-diffusion-v1-4'
    """Diffusion checkpoints"""
    

class GaussCtrlPipeline(VanillaPipeline):
    """GaussCtrl pipeline"""

    config: GaussCtrlPipelineConfig

    def __init__(
        self,
        config: GaussCtrlPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.test_mode = test_mode
        self.langsam = LangSAM()
        
        self.prompt = self.config.prompt
        self.pipe_device = 'cuda:0'
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config.diffusion_ckpt, controlnet=controlnet).to(self.device).to(torch.float16)
        self.pipe.to(self.pipe_device)

        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.prompt + ', ' + added_prompt
        self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        
        view_num = len(self.datamanager.cameras) 
        anchors = [(view_num * i) // self.config.ref_view_num for i in range(self.config.ref_view_num)] + [view_num]
        
        random.seed(13789)
        self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])] 
        self.num_ref_views = len(self.ref_indices)

        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.config.chunk_size

        self.noise_renderer = MultiVeiwNoiseRenderer(width=512,height=512)

    def render_reverse1(self):
        depth_maps_lst = []
        mats_view_lst = []
        mats_proj_lst = []
        num_frames = 4 # m
        do_resize = True
        if do_resize:
            test_size = (128,128)
        else:
            test_size = (512,512)
        test_prompt = "a stone bear statue in the forest"
        ### render depths map, get view/proj mats
        for cam_idx in range(len(self.datamanager.cameras)):
            CONSOLE.print(f"Rendering view {cam_idx}", style="bold yellow")
            current_cam = self.datamanager.cameras[cam_idx].to(self.device)
            if current_cam.metadata is None:
                current_cam.metadata = {}
            current_cam.metadata["cam_idx"] = cam_idx
            rendered_image = self._model.get_outputs_for_camera(current_cam)

            rendered_rgb = rendered_image['rgb'].to(torch.float16) # [512 512 3] 0-1
            rendered_depth = rendered_image['depth'].to(torch.float16) # [512 512 1]
            
            mats_view_lst.append(rendered_image["mat_view"])
            mats_proj_lst.append(rendered_image['mat_proj'])
            if do_resize: # need to rerange dim order for F
                rendered_depth = rearrange(rendered_depth, 'h w c -> c h w').unsqueeze(0)
                rendered_depth=F.interpolate(rendered_depth, size=test_size, mode='nearest')
                rendered_depth = rearrange(rendered_depth, 'b c h w -> b h w c').squeeze(0)
            depth_maps_lst.append(rendered_depth)
            if len(depth_maps_lst) == num_frames:
                break
        ### edit
        ## make batch
        depths_maps_batch_lst = []
        mats_view_batch_lst = []
        # during test b==1
        mats_view = torch.stack(mats_view_lst,dim=0) # [m,4,4]
        mats_proj = torch.stack(mats_proj_lst,dim=0) # [m,4,4]
        depth_maps = torch.stack(depth_maps_lst,dim=0) # [m,h,w,1]
        save_tensors = {
            'mats_view':mats_view,
            'mats_proj':mats_proj,
            'depth_maps':depth_maps,
        }
        torch.save(save_tensors, "./gaussctrl/depth_tensors.pt")
        print("test tensor saved.")
        exit()
        #
        main_cfg_path = "./gaussctrl/depth_generation_fix_frames.yaml"
        ckpt_path = "./gaussctrl/depth_gen.ckpt"
        gen_config = yaml.load(open(main_cfg_path, 'rb'), Loader=yaml.SafeLoader)
        gen_model = DepthGenerator(gen_config)
        
        checkpoint = torch.load(ckpt_path)
        state_dict = checkpoint['state_dict']
        
        gen_model.load_state_dict(state_dict, strict=False)
        gen_model.to(torch.device("cuda"))
        gen_model.eval()
        ## make test batch
        batch = {}
        
        depth_inv_norm_full = get_inv_norm_depth(depth_maps.squeeze(-1)) # [m,h,w]
        if do_resize:
            d_inv_norm_size = test_size
        else:
            d_inv_norm_size = (512,512)
        depth_inv_norm_small= F.interpolate(depth_inv_norm_full.unsqueeze(1), # m,1,h,w
                                        (test_size[0]//8, test_size[1]//8), mode='nearest').squeeze(1)
            

        batch['depth_inv_norm_small'] = depth_inv_norm_small.unsqueeze(0) # [b=1,m,h,w]
        batch['depths']=depth_maps.squeeze(-1)[None,:,:,:] # [b=1,m,h,w] to a batch
        batch['prompt']=[test_prompt]*depth_maps.shape[0] # copy by num_frames
        batch['mats_view']=mats_view[None,:,:,:] # [b=1,m,4,4] to a batch
        batch['mats_proj']=mats_proj[None,:,:,:] # [b=1,m,4,4] to a batch
        
        img_pred = gen_model.inference_gen(batch) #[b,m,h,w,3]
        print(f"shape of img_pred: {img_pred.shape}")
        
        save_test_res(img_pred)
            
        exit()
        
    
    def render_reverse(self):
        '''Render rgb, depth and reverse rgb images back to latents'''
        for cam_idx in range(len(self.datamanager.cameras)):
            CONSOLE.print(f"Rendering view {cam_idx}", style="bold yellow")
            current_cam = self.datamanager.cameras[cam_idx].to(self.device)
            if current_cam.metadata is None:
                current_cam.metadata = {}
            current_cam.metadata["cam_idx"] = cam_idx
            rendered_image = self._model.get_outputs_for_camera(current_cam)

            rendered_rgb = rendered_image['rgb'].to(torch.float16) # [512 512 3] 0-1
            rendered_depth = rendered_image['depth'].to(torch.float16) # [512 512 1]
            
            #self.noise_renderer.setup_noise("gaussctrl/noise_npy/noise1113_1944.npy")
            depth_pts_world = self.depth_map_screen_to_world(rendered_image['depth'],rendered_image["mat_view"])
            # [h,w,3]
            #depth_pts_world = rearrange(depth_pts_world,'b h w c -> (b h w) c').cpu().numpy()
            
            #mat_view = rendered_image["mat_view"].to(torch.float32).cpu().numpy()
            #mat_proj = rendered_image['mat_proj'].to(torch.float32).cpu().numpy()
            #mat_c2w = rendered_image['mat_c2w'].to(torch.float32).cpu().numpy()
            
            #multiviewcosis_rgb_mask = self.noise_renderer.render_noise_SS(rendered_depth32, mat_view, mat_proj, mat_c2w)
            #multiviewcosis_binary_mask = self.noise_renderer.binarize_mask(multiviewcosis_rgb_mask)
            
            #rgb_img = rendered_image['rgb'].to(torch.float32).cpu().numpy() # [512 512 3] 0-1

            #rgb_img[multiviewcosis_mask[...,:-1] != 0.0] = multiviewcosis_mask[...,:-1][multiviewcosis_mask[...,:-1] != 0.0]

            #uint8_rgb_img = (rgb_img * 255).astype(np.uint8)
            #self.noise_renderer.save_result_mask(uint8_rgb_img,"gaussctrl/noise_res_img")

            # reverse the images to noises
            self.pipe.unet.set_attn_processor(processor=AttnProcessor())
            self.pipe.controlnet.set_attn_processor(processor=AttnProcessor()) 
            init_latent = self.image2latent(rendered_rgb)
            disparity = self.depth2disparity_torch(rendered_depth[:,:,0][None]) 
            
            self.pipe.scheduler = self.ddim_inverser
            latent, _ = self.pipe(prompt=self.positive_prompt, #  placeholder here, since cfg=0
                                num_inference_steps=self.num_inference_steps, 
                                latents=init_latent, 
                                image=disparity, return_dict=False, guidance_scale=0, output_type='latent')

            # LangSAM is optional
            if self.config.langsam_obj != "":
                langsam_obj = self.config.langsam_obj
                langsam_rgb_pil = Image.fromarray((rendered_rgb.cpu().numpy() * 255).astype(np.uint8))
                masks, _, _, _ = self.langsam.predict(langsam_rgb_pil, langsam_obj)
                mask_npy = masks.clone().cpu().numpy()[0] * 1

            if self.config.langsam_obj != "":
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, mask_npy, depth_pts_world)
            else: 
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, None, depth_pts_world)
            if cam_idx == self.chunk_size -1:
                break
    
    def depth_map_screen_to_world(self, depth_map, mats_view):
        '''
        Args:
        depth_map [h,w,1] tensor
        mats_view [4,4] tensor
        Returns:
        depth_world [h,w,3] tensor
        '''
        #print(extract_camera_position_and_forward(view_mat.cpu().numpy()))
        h, w, _ = depth_map.shape
        
        # rays_d [h,w,3]
        rays_d = torch.from_numpy(
                    compute_ray_directions(resolution=(h, w)) 
                ).to(depth_map.device) 
        
        pts_view = rays_d * depth_map # [h,w,3]
        pts_view = torch.cat((pts_view, 
                            torch.ones((h, w, 1), dtype=rays_d.dtype, device=rays_d.device))
                            ,dim=-1) # [h,w,4]
        pts_view = rearrange(pts_view, 'h w c -> (h w) c').unsqueeze(-1) # [hw,4,1]
        
        # Normalize by w to get 3D coordinates in camera space
        inv_view_mat = torch.linalg.inv(mats_view) # [4,4]
        ###### ?????? validation not tested yet
        pts_world = inv_view_mat.to(torch.float32) @ pts_view.to(torch.float32)  # [hw, 4,1]
        pts_world /= pts_world[:,3:4,:] # divide by w
        pts_world = rearrange(pts_world.squeeze(-1), '(h w) c -> h w c',h=h)
        return pts_world[:,:,:3] # [h,w,3]
    
    def edit_images(self):
        '''Edit images with ControlNet and AttnAlign''' 
        # Set up ControlNet and AttnAlign
        self.pipe.scheduler = self.ddim_scheduler
        
        '''
        self.pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=2)) 
        '''
        self.pipe.unet.set_attn_processor(
                        processor=TriPlaneAttnProcessor(self_attn_coeff=0.6,bbox_length=8.0
                        ))
        #self.pipe.controlnet.set_attn_processor(
        #                processor=TriPlaneAttnProcessor(self_attn_coeff=0,
        #                )) 
        
        CONSOLE.print("Done Reset Attention Processor", style="bold blue")
        
        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")
        '''
        ref_disparity_list = []
        ref_z0_list = []
        for ref_idx in self.ref_indices:
            ref_data = deepcopy(self.datamanager.train_data[ref_idx]) 
            ref_disparity = self.depth2disparity(ref_data['depth_image']) 
            ref_z0 = ref_data['z_0_image']
            ref_disparity_list.append(ref_disparity)
            ref_z0_list.append(ref_z0) 
            
        ref_disparities = np.concatenate(ref_disparity_list, axis=0)
        ref_z0s = np.concatenate(ref_z0_list, axis=0)
        ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16).to(self.pipe_device)
        ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16).to(self.pipe_device)
        '''
        

        # Edit images in chunk
        for idx in range(0, len(self.datamanager.train_data), self.chunk_size): 
            chunked_data = self.datamanager.train_data[idx: idx+self.chunk_size]
            
            indices = [current_data['image_idx'] for current_data in chunked_data]
            mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
            unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
            pts_world= [current_data['pts_world'] for current_data in chunked_data] # [[h,w,3]]*chunk_size
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
            disparities = np.concatenate(depth_images, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)
            # [b,h,w,3]
            pts_world = torch.from_numpy(np.stack(pts_world, axis=0).copy()).to(torch.float16).to(self.pipe_device)

            z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
            z0s = np.concatenate(z_0_images, axis=0)
            latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

            #disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
            disp_ctrl_chunk = disparities_torch
            #latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
            latents_chunk = latents_torch
            ## 
            cross_attention_kwargs={}
            cross_attention_kwargs.update({"pts_world": pts_world})
            
            
            chunk_edited = self.pipe(
                                prompt=[self.positive_prompt] * (len(chunked_data)),
                                negative_prompt=[self.negative_prompts] * (len(chunked_data)),
                                latents=latents_chunk,
                                image=disp_ctrl_chunk,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                eta=self.eta,
                                output_type='pt',
                                cross_attention_kwargs=cross_attention_kwargs,
                            )
            chunk_edited = chunk_edited.cpu() 
            
            

            # Insert edited images back to train data for training
            for local_idx, edited_image in enumerate(chunk_edited):
                res_save_dir = "./gaussctrl/mv_res"
                os.makedirs(res_save_dir, exist_ok=True)
                img_pred_np = edited_image.cpu().numpy()
                filename = f"{res_save_dir}/{local_idx}.png"
                cv2.imwrite(filename, img_pred_np)
            
                
                global_idx = indices[local_idx]

                bg_cntrl_edited_image = edited_image
                if mask_images != []:
                    mask = torch.from_numpy(mask_images[local_idx])
                    bg_mask = 1 - mask

                    unedited_image = unedited_images[local_idx].permute(2,0,1)
                    bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 

                self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [512 512 3]
            exit()
        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")

    @torch.no_grad()
    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 512, 512]) -1~1
        latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        """
        Args: depth numpy array [1 512 512]
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
        return disparity_map[None]
    
    def depth2disparity_torch(self, depth):
        """
        Args: depth torch tensor
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity) # 0.00233~1
        disparity_map = torch.concatenate([disparity_map, disparity_map, disparity_map], dim=0)
        return disparity_map[None]

    def update_datasets(self, cam_idx, unedited_image, depth, latent, mask, pts_world=None):
        """Save mid results"""
        self.datamanager.train_data[cam_idx]["unedited_image"] = unedited_image 
        self.datamanager.train_data[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_data[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        if pts_world is not None:
            self.datamanager.train_data[cam_idx]["pts_world"] = pts_world.cpu().to(torch.float32).numpy()
        if mask is not None:
            self.datamanager.train_data[cam_idx]["mask_image"] = mask 

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step) # camera, data
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    
    def vis_pts(self, pts): # [num, 3]
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o')

        # Set labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Point Visualization')

        # Show the plot
        plt.show()

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

def save_test_res(img_pred):
    res_save_dir = "./gaussctrl/mv_res"
    os.makedirs(res_save_dir, exist_ok=True)
    b,m,h,w,c=img_pred.shape
    img_pred_np = img_pred[0].cpu().numpy()
    for i in range(m):
        filename = f"{res_save_dir}/{i}.png"
        cv2.imwrite(filename, img_pred_np[i])