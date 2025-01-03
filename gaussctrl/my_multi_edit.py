
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

CONSOLE = Console(width=120)


class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            ref0_frame_index = [0] * video_length
            ref1_frame_index = [1] * video_length
            ref2_frame_index = [2] * video_length
            ref3_frame_index = [3] * video_length
            
            hidden_states_ref0 = compute_attn(attn, query, key, value, video_length, ref0_frame_index, attention_mask)
            hidden_states_ref1 = compute_attn(attn, query, key, value, video_length, ref1_frame_index, attention_mask)
            hidden_states_ref2 = compute_attn(attn, query, key, value, video_length, ref2_frame_index, attention_mask)

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, ref3_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, ref3_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states_ref3 = torch.bmm(attention_probs, value)
        
        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * torch.mean(torch.stack([hidden_states_ref0, hidden_states_ref1, hidden_states_ref2, hidden_states_ref3]), dim=0) if not is_cross_attention else hidden_states_ref3 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@dataclass
class TestPipelineConfig(VanillaPipelineConfig):
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
    

class TestPipeline(VanillaPipeline):
    """Test pipeline"""

    config: TestPipelineConfig

    def __init__(
        self,
        config: TestPipelineConfig,
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
        self.ddim_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet).to(self.device).to(torch.float16)
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
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, mask_npy)
            else: 
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, None)
        
    def edit_images(self):
        '''Edit images with ControlNet and AttnAlign''' 
        # Set up ControlNet and AttnAlign
        self.pipe.scheduler = self.ddim_scheduler
        self.pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=2)) 
        CONSOLE.print("Done Reset Attention Processor", style="bold blue")
        
        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")
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

        # Edit images in chunk
        for idx in range(0, len(self.datamanager.train_data), self.chunk_size): 
            chunked_data = self.datamanager.train_data[idx: idx+self.chunk_size]
            
            indices = [current_data['image_idx'] for current_data in chunked_data]
            mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
            unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
            disparities = np.concatenate(depth_images, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

            z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
            z0s = np.concatenate(z_0_images, axis=0)
            latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

            disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
            latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
            
            chunk_edited = self.pipe(
                                prompt=[self.positive_prompt] * (self.num_ref_views+len(chunked_data)),
                                negative_prompt=[self.negative_prompts] * (self.num_ref_views+len(chunked_data)),
                                latents=latents_chunk,
                                image=disp_ctrl_chunk,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                eta=self.eta,
                                output_type='pt',
                            ).images[self.num_ref_views:]
            chunk_edited = chunk_edited.cpu() 

            # Insert edited images back to train data for training
            for local_idx, edited_image in enumerate(chunk_edited):
                global_idx = indices[local_idx]

                bg_cntrl_edited_image = edited_image
                if mask_images != []:
                    mask = torch.from_numpy(mask_images[local_idx])
                    bg_mask = 1 - mask

                    unedited_image = unedited_images[local_idx].permute(2,0,1)
                    bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 

                self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [512 512 3]
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

    def update_datasets(self, cam_idx, unedited_image, depth, latent, mask):
        """Save mid results"""
        self.datamanager.train_data[cam_idx]["unedited_image"] = unedited_image 
        self.datamanager.train_data[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_data[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
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
