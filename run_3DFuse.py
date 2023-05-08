import os
import numpy as np
import torch
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import imageio
import gradio as gr
import pose
from PIL import Image
from pose import circular_poses
from pose import get_K

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter
from run_img_sampling import SD
from misc import torch_samps_to_imgs
from pose import PoseConfig

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis

from pytorch3d.renderer import PointsRasterizationSettings

from semantic_coding import semantic_coding, semantic_karlo, semantic_sd
from pc_project import point_e, render_depth_from_cloud

from point_e.util import point_cloud
import random
import json

device_glb = torch.device("cuda")

def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }


def axis_angle_to_matrix(axis, angle):
    angle = torch.tensor(angle).cuda()
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    x, y, z = axis

    mat = torch.zeros((3, 3)).cuda()
    mat[0, 0] = x * x * C + c
    mat[0, 1] = x * y * C - z * s
    mat[0, 2] = x * z * C + y * s
    mat[1, 0] = y * x * C + z * s
    mat[1, 1] = y * y * C + c
    mat[1, 2] = y * z * C - x * s
    mat[2, 0] = z * x * C - y * s
    mat[2, 1] = z * y * C + x * s
    mat[2, 2] = z * z * C + c

    return mat

def total_variation_loss(image):
    """
    Compute the total variation loss for an input image.

    Args:
        image (torch.Tensor): NeRF-generated RGB output with shape (B, C, H, W),
                              where B is batch size, C is the number of channels (3 for RGB),
                              H is image height, and W is image width.

    Returns:
        tv_loss (torch.Tensor): The total variation loss.
    """
    # Calculate the differences in the horizontal and vertical directions
    diff_horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
    diff_vertical = image[:, :, 1:, :] - image[:, :, :-1, :]

    # Compute the L1-norm of the differences
    l1_horizontal = torch.sum(torch.abs(diff_horizontal))
    l1_vertical = torch.sum(torch.abs(diff_vertical))

    # Combine the horizontal and vertical L1-norms to compute the total variation loss
    tv_loss = l1_horizontal + l1_vertical

    return tv_loss

import torch
import torch.nn.functional as F

def laplacian(tensor):
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=tensor.device)
    laplacian_channels = []
    for i in range(tensor.shape[1]):
        channel = tensor[:, i:i+1, :, :]
        laplacian_channel = F.conv2d(channel, kernel, padding=1)
        laplacian_channels.append(laplacian_channel)
    return torch.cat(laplacian_channels, dim=1)

def laplacian_sharpness_loss(depth):
    laplacian_depth = laplacian(depth.unsqueeze(0).unsqueeze(0))
    
    sharpness_loss_depth = torch.mean(torch.abs(laplacian_depth))
    
    total_sharpness_loss = sharpness_loss_depth
    return total_sharpness_loss

def random_pose_variation(pose, max_angle):
    pose = torch.tensor(pose).float().cuda()
    # This function perturbs the input pose by rotating it around a random axis by a random angle.
    angle = random.uniform(-max_angle, max_angle)
    axis = torch.tensor([random.random(), random.random(), random.random()]).cuda()
    axis = axis / torch.norm(axis)
    rotation_matrix = axis_angle_to_matrix(axis, angle)

    # Apply the rotation only to the 3x3 submatrix of the pose
    perturbed_pose = pose.clone()
    perturbed_pose[:3, :3] = torch.mm(pose[:3, :3], rotation_matrix)
    # Convert to numpy
    perturbed_pose = perturbed_pose.cpu().numpy()    
    return perturbed_pose


def reproject_to_another_pose(image, depth, pose1, pose2, H, W, intrinsics):
    # Convert input numpy arrays to PyTorch tensors
    image = image.float()
    pose1 = torch.from_numpy(pose1).float().cuda()
    pose2 = torch.from_numpy(pose2).float().cuda()
    intrinsics = intrinsics.cuda()

    # Create meshgrid
    yy, xx = torch.meshgrid([torch.arange(0, H).float().cuda(), torch.arange(0, W).float().cuda()])
    xy_homogeneous = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3).t()

    # Compute coordinates in the camera frame
    camera_coordinates = torch.inverse(intrinsics) @ xy_homogeneous * depth.view(-1)

    # Compute coordinates in the world frame
    world_coordinates = pose1[:3, :3] @ camera_coordinates + pose1[:3, 3].unsqueeze(1)

    # Reproject the world coordinates to the new camera frame
    camera_coordinates_reprojected = torch.inverse(pose2[:3, :3]) @ (world_coordinates - pose2[:3, 3].unsqueeze(1))

    # Compute the new pixel coordinates
    pixel_coordinates_reprojected = intrinsics @ camera_coordinates_reprojected
    pixel_coordinates_reprojected = pixel_coordinates_reprojected[:2] / pixel_coordinates_reprojected[2]

    # Reshape and normalize the pixel coordinates
    pixel_coordinates_reprojected = pixel_coordinates_reprojected.t().reshape(H, W, 2)
    pixel_coordinates_reprojected = (pixel_coordinates_reprojected - torch.tensor([W / 2, H / 2]).cuda()) / torch.tensor([W / 2, H / 2]).cuda()

    # Sample the reprojected image using grid_sample
    reprojected_image = torch.nn.functional.grid_sample(image, pixel_coordinates_reprojected.unsqueeze(0), align_corners=False)

    return reprojected_image

def replace_color(image, old_color, new_color):
    width, height = image.size
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            if pixels[x, y] == old_color:
                pixels[x, y] = new_color

    return image


class SJC_3DFuse(BaseConf):
    family:     str = "sd"
    sd:         SD = SD(
        variant="v1",
        prompt="a comfortable bed",
        scale=100.0,
        dir="./results",
        alpha=0.3
    )
    lr:         float = 0.05
    n_steps:    int = 2000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False , bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=60.0, R=1.5)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight:       int = 0

    var_red:     bool = True
    exp_dir:     str = "./results"
    ti_step:     int = 800
    pt_step:     int = 800
    initial:    str = ""
    random_seed:     int = 0
    semantic_model:     str = "Karlo"
    bg_preprocess:     bool = True
    num_initial_image:     int = 4
    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()
        seed = cfgs.pop('random_seed')
        seed_everything(seed)
        initial = cfgs.pop('initial')
        exp_instance_dir=os.path.join(cfgs.pop('exp_dir'),initial)
        
        initial_prompt=cfgs['sd']['prompt']
        semantic_model = cfgs.pop('semantic_model')
        
        # Initial image generation
        image_dir=os.path.join(exp_instance_dir,'initial_image')

        # If instance0 initial image exists skip generation
        if not os.path.exists(os.path.join(image_dir,'instance0.png')):        
            if semantic_model == "Karlo":
                semantic_karlo(initial_prompt,image_dir,cfgs['num_initial_image'],cfgs['bg_preprocess'], seed)
            elif semantic_model == "SD":
                semantic_sd(initial_prompt,image_dir,cfgs['num_initial_image'],cfgs['bg_preprocess'], seed)
            else:
                raise NotImplementedError

        initial_images = []
        initial_depth_images = []
        self.num_initial_image=30
        for i in range(self.num_initial_image):
            initial_images.append(Image.open(os.path.join(image_dir,'instance{}.png'.format(i))))
            # Check if depth image exists
            if not os.path.exists(os.path.join(image_dir,'instance{}_depth.png'.format(i))):
                continue
            image =Image.open(os.path.join(image_dir,'instance{}_depth.png'.format(i)))
            image = torch.tensor(np.array(image)).squeeze(0).float() / 255.0
            print("Initial depth shape", image.shape)
            image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            image = image
            image = image.to("cuda")
            image = 1-image
            image *= 5.0
            initial_depth_images.append(image)
            

        
        # Optimization  and pivotal tuning for LoRA
        semantic_coding(exp_instance_dir,cfgs,self.sd,initial)
        
        
        # Load SD with Consistency Injection Module
        family = cfgs.pop("family")
        model = getattr(self, family).make()
        print(model.prompt)
        cfgs.pop("vox")
        vox = self.vox.make()
        
        cfgs.pop("pose")
        poser = self.pose.make()
        
        # Get coarse point cloud from off-the-shelf model
        print("Building point cloud")
        # Check if point cloud exists
        if not os.path.exists(os.path.join(exp_instance_dir,'points.npz')):
            points = point_e(device=device_glb,exp_dir=exp_instance_dir)
            # Use points.save to save the point cloud
            points.save(os.path.join(exp_instance_dir,"points.npz"))
            print("Saved point cloud")
        else:
            print("Loaded point cloud")
            points = point_cloud.PointCloud.load(os.path.join(exp_instance_dir,
                                                              "points.npz"))
            
        initial_latents = []
        for image in initial_images:
            # Replace pure white with pure green in image
            image = image.convert("RGB")
            image = replace_color(image, (255, 255, 255), (0, 255, 0))
            
            # PIL image to torch tensor
            image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image = image.to("cuda")
            # Resize image to 512x512
            image = torch.nn.functional.interpolate(image, (512, 512), mode="area")
            initial_latents.append(model.encode(image))

        # Score distillation
        pipeline = NeRF_Fuser(**cfgs, poser=poser,model=model,vox=vox,exp_instance_dir=exp_instance_dir, points=points, is_gradio=True, initial_latents=initial_latents,
                              initial_depth_images=initial_depth_images)
        next(pipeline.train())      

class WriteNerf(BaseConf):
    family:     str = "sd"
    sd:         SD = SD(
        variant="v1",
        prompt="a comfortable bed",
        scale=100.0,
        dir="./results",
        alpha=0.3
    )
    lr:         float = 0.05
    n_steps:    int = 2000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False , bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=60.0, R=1.5)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight:       int = 0

    var_red:     bool = True
    exp_dir:     str = "./results"
    ti_step:     int = 800
    pt_step:     int = 800
    initial:    str = ""
    random_seed:     int = 0
    semantic_model:     str = "Karlo"
    bg_preprocess:     bool = True
    num_initial_image:     int = 4
    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()
        seed = cfgs.pop('random_seed')
        seed_everything(seed)
        initial = cfgs.pop('initial')
        exp_instance_dir=os.path.join(cfgs.pop('exp_dir'),initial)
        cfgs.pop("vox")
        vox = self.vox.make()
        
        cfgs.pop("pose")
        poser = self.pose.make()

        family = cfgs.pop("family")
        model = getattr(self, family).make()
        
        K, poses, _, _ = poser.sample_train(500, "cuda")

        output = {}
        output["camera_angle_x"] = output["camera_angle_y"] = np.deg2rad(60)
        output["fl_x"] = output["fl_y"] = 0.5 / np.tan(output["camera_angle_x"] / 2)
        output["width"] = output["height"] = 512
        output["cx"] = output["cx"] = output["width"] / 2
        output["frames"] = []

        aabb = vox.aabb.T.cpu().numpy()
        ckpt_list = glob.glob(os.path.join(exp_instance_dir,'3d','ckpt','*'))
            # If ckpt exists, load it
        if len(ckpt_list) > 0:
            state_dict = torch.load(ckpt_list[0])
            vox.load_state_dict(state_dict)
            print("Loaded ckpt for voxnerf")
        vox = vox.to("cuda")
        vox.blend_bg_texture = False

        for i, render_pose in enumerate(poses):
            transform = np.array(render_pose)
            # Invert transform
            transform = np.linalg.inv(transform)
            transform = transform.tolist()
            frame = {
                "file_path" : "images/{}.png".format(i),
                "sharpness": 30.0,
                "transform_matrix": transform
            }
            y, _ = render_one_view(vox, aabb, 64, 64, K[i], render_pose, use_app_net=True)
            y = model.decode(y)
            y = torch_samps_to_imgs(y)[0]
            print("Y shape: {}".format(y.shape))
            y = Image.fromarray(y)
            # Make images dir if it doesn't exist
            os.makedirs(os.path.join(exp_instance_dir, "images"), exist_ok=True)
            y.save(os.path.join(exp_instance_dir, "images", "{}.png".format(i)))
            output["frames"].append(frame)
            if i > 100:
                continue
        # Convert frame to json
        json_contents = json.dumps(output)
        # Write json to file
        with open(os.path.join(exp_instance_dir, "transforms_train.json"), "w") as f:
            f.write(json_contents)            


import glob

class NeRF_Fuser:
    def __init__(
        self, poser, vox, model,
        lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
        depth_weight, var_red, exp_instance_dir, points, is_gradio, initial_depth_images,
        initial_latents, **kwargs
    ):
        print("Initializing pipeline")
        self.poser = poser
        self.vox = vox
        self.model = model
        self.lr = lr
        self.n_steps = n_steps
        self.emptiness_scale = emptiness_scale
        self.emptiness_weight = emptiness_weight
        self.emptiness_step = emptiness_step
        self.emptiness_multiplier = emptiness_multiplier
        self.depth_weight = depth_weight
        self.var_red = var_red
        self.exp_instance_dir = exp_instance_dir
        self.points = points
        self.is_gradio = is_gradio
        self.n_steps = 2000

        self.initial_latents = initial_latents
        self.initial_depth_images = initial_depth_images

        assert model.samps_centered()
        _, self.target_H, self.target_W = model.data_shape()
        self.bs = 1
        self.aabb = vox.aabb.T.cpu().numpy()
        self.vox = vox.to(device_glb)
        self.opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

        self.H, self.W = poser.H, poser.W
        self.Ks_, self.poses_, self.prompt_prefixes_, self.angles_list = poser.sample_train(n_steps, device_glb)

        self.ts = model.us[30:-10]

        self.fuse = EarlyLoopBreak(5)

        self.raster_settings = PointsRasterizationSettings(
            image_size=800,
            radius=0.02,
            points_per_pixel=10
        )

        self.ts = model.us[30:-10]
        self.calibration_value = 0.0

    def initial_overfit(self):
        poses = pose.circular_poses(1.25,0.0,len(self.initial_latents))
        for i in range(20):
            print("Overfitting, pose size", len(poses))
            print("Current step", i)
            for j, render_pose in enumerate(poses):
                expected_latents = self.initial_latents[j]
                y1, depth1, ws1, _ = render_one_view(self.vox, self.aabb, self.H, self.W, pose.get_K(64,64,60), render_pose, return_w=True, use_app_net=True)
                # Compute mse loss between ys and expected_latents
                loss = torch.nn.functional.mse_loss(y1, expected_latents)
                if len(self.initial_depth_images) > 0:
                    loss += torch.nn.functional.mse_loss(depth1, self.initial_depth_images[j])
                # Choose random integer between 4 and 9
                border_size = np.random.randint(4, 10)

                center_depth = depth1[border_size:-border_size, border_size:-border_size]
                border_depth_mean = (depth1.sum() - center_depth.sum()) / (64*64-(64-border_size*2)**2)
                center_depth_mean = center_depth.mean()
                depth_diff = center_depth_mean - border_depth_mean
                depth_loss = - torch.log(depth_diff + 1e-12)
                depth_loss = depth_loss*10
                loss += depth_loss
                
                loss.backward()
                print("i, j", i, j, "loss", loss.item(), "depth_loss", depth_loss.item())
                self.opt.step()
                self.opt.zero_grad()
                
            rgbs = self.model.decode(y1)
            # Convert to PIL image and save
            rgbs = rgbs[0].cpu().numpy().transpose(1,2,0)
            rgbs = (rgbs * 255).astype(np.uint8)
            rgbs = Image.fromarray(rgbs)
            rgbs.save(os.path.join(self.exp_instance_dir,"initial_overfit.png"))

    def train(self):
        with tqdm(total=self.n_steps) as pbar, \
            HeartBeat(pbar) as hbeat, \
                EventStorage(output_dir=os.path.join(self.exp_instance_dir,'3d')) as metric:
            self.metric = metric
            self.pbar = pbar
            self.hbeat = hbeat
            self.opt.zero_grad()
            
            # List files in instance_dir/3d/ckpt
            ckpt_list = glob.glob(os.path.join(self.exp_instance_dir,'3d','ckpt','*'))
            # If ckpt exists, load it
            if len(ckpt_list) > 0:
                state_dict = torch.load(ckpt_list[0])
                self.vox.load_state_dict(state_dict)
                print("Loaded ckpt for voxnerf")
            else:
                print("Found no voxnerf ckpt")
            print("Starting training poses_ length: ", len(self.poses_))
            self.initial_overfit()
            for j in range(5):
                for i in range(len(self.poses_)):

                    use_guidance = i % 5 != 0
                    if i < 1000: use_guidance = True
                    use_guidance = True
                    if (i % 1 == 0):
                        for g in self.opt.param_groups:
                            g['lr'] *= .999998

                    # TODO: Must fix
                    embed_fr = (i+j*len(self.poses_))/(self.n_steps-self.n_steps*0.7)
                                
                    if use_guidance:
                        self.train_one_step(self.poses_[i], self.angles_list[i], self.Ks_[i],
                                            self.prompt_prefixes_[i], i+j*len(self.poses_), embed_fr)
                    else:
                        self.train_one_step_no_guidance(self.poses_[i], self.angles_list[i], self.Ks_[i], i)
                    if i%2 == 0:                                                
                        self.opt.step()
                        self.opt.zero_grad()
                    if (i%1000 == 0):
                        metric.put_artifact(
                            "ckpt", ".pt","", lambda fn: torch.save(self.vox.state_dict(), fn)
                        )

            with EventStorage("result"):
                evaluate(self.model, self.vox, self.poser)
            
            if self.is_gradio:    
                yield gr.update(visible=True), f"Generation complete. Please check the video below. \nThe result files and logs are located at {exp_instance_dir}", gr.update(value=os.path.join(exp_instance_dir,'3d/result_10000/video/step_100_.mp4'))
            else :
                yield None
        
            metric.step()

            hbeat.done()

    def train_one_step_no_guidance(self, pose, angle, k, i, max_angle_variation=0.1):        
        # Render NeRF from the original pose
        y1, depth1, ws1 = render_one_view(self.vox, self.aabb, self.H, self.W, k, pose, return_w=True)
        y1 = self.model.decode(y1).float()
        tvl1 = total_variation_loss(y1)/50000.0 + total_variation_loss(depth1.unsqueeze(0).unsqueeze(0))/50000.0
        lsl1 = laplacian_sharpness_loss(depth1)/50000.0
        loss1 = tvl1 + lsl1

        # Render NeRF from a slightly perturbed pose
        perturbed_pose = random_pose_variation(pose, max_angle_variation)
        y2, depth2, ws2 = render_one_view(self.vox, self.aabb, self.H, self.W, k, perturbed_pose, return_w=True)
        y2 = self.model.decode(y2).float()
        tvl2 = total_variation_loss(y2)/50000.0 + total_variation_loss(depth2.unsqueeze(0).unsqueeze(0))/50000.0
        lsl2 = laplacian_sharpness_loss(depth2)/50000.0
        loss2 = tvl2 + lsl2

        # Resize depth and depth2 to 512x512
        depth1 = F.interpolate(depth1.unsqueeze(0).unsqueeze(0), size=(512,512), mode='bilinear')
        depth2 = F.interpolate(depth2.unsqueeze(0).unsqueeze(0), size=(512,512), mode='bilinear')
        # Restore original channel dimension
        depth1 = depth1.squeeze(0).squeeze(0)
        depth2 = depth2.squeeze(0).squeeze(0)

        # Reproject the results of the first pose to the second pose
        intrinsics = self.Ks_[i]
        # Convert intrinsics to cuda torch
        intrinsics = torch.tensor(intrinsics).float().to(device_glb)
        y1_reprojected = reproject_to_another_pose(y1, depth1, pose, perturbed_pose, 512,512, intrinsics)
        # Compute self-consistency loss
        scl = F.mse_loss(y1_reprojected, y2)

        #Project y2 to the first pose and compute the self-consistency loss
        y2_reprojected = reproject_to_another_pose(y2, depth2, perturbed_pose, pose, 512,512, intrinsics)
        scl += F.mse_loss(y2_reprojected, y1)

        scl = scl / 2.0
        scl = scl * 0.001

        # Combine losses
        loss = loss1 + loss2 + scl
        loss = loss/100.0
        print("Unsupervised loss: ", loss.item())

        if every(self.pbar, percent=1):
            with torch.no_grad():
                vis_routine(self.metric, y1, depth1,"",None)
       

        loss.backward()

        self.metric.step()
        self.pbar.update()

        self.pbar.set_description(str(loss))
        self.hbeat.beat()


    def train_one_step(self, pose, angle, k, prompt_prefix, i, embed_fr):
        depth_map = render_depth_from_cloud(self.points, 
                                                angle,
                                                self.raster_settings,
                                                device_glb,
                                                self.calibration_value)
        
        render_app_net = True
        if i > 500:
            render_app_net = True
   #     if i > 500 and i < 1000:
  #          self.vox.color.requires_grad = False
            #self.vox.density.requires_grad = False
 #       else:
#            self.vox.color.requires_grad = True
            #self.vox.density.requires_grad = True
        y, depth, ws, weight_entropy = render_one_view(self.vox, self.aabb, self.H, self.W, k,
                                       pose, return_w=True, embed_fr = embed_fr, use_app_net = render_app_net)

        p = f"{prompt_prefix} {self.model.prompt}"
        score_conds = self.model.prompts_emb([p])['c']
        negative_score_conds = self.model.prompts_emb(["blurry, distorted, noisy, low resolution, distorted anatomy, ugly, bad, low lighting"])['c']
        #score_conds = torch.cat([negative_score_conds['c'], score_conds['c']])
            
        with torch.no_grad():
            ts_frac = int((len(self.ts))*embed_fr)
            if ts_frac == 0: ts_frac = 1
            #chosen_σs = np.random.choice(self.ts[(ts_frac-1):ts_frac], self.bs, replace=False)
            noise_amount = 3.1*(1-(i/self.n_steps))
            chosen_σs = np.random.choice((noise_amount,), self.bs, replace=False)
            chosen_σs = chosen_σs.clip(0.5,12)
            print("Chosen σ: ", chosen_σs)
            chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
            chosen_σs = torch.as_tensor(chosen_σs, device=self.model.device, dtype=torch.float32)


            noise = torch.randn(self.bs, *y.shape[1:], device=self.model.device)

            zs = y + chosen_σs * noise
            Ds = self.model.denoise(zs, chosen_σs,depth_map.unsqueeze(dim=0),c=score_conds, uc=negative_score_conds)

            chosen_σs = chosen_σs.clamp(0.5,11.91)
            if self.var_red:
                grad = (Ds - y) / chosen_σs
            else:
                grad = (Ds - zs) / chosen_σs

            grad = grad.mean(0, keepdim=True)
            
        y.backward(-grad, retain_graph=True)
      #  tvl = total_variation_loss(y)
       # tvl = tvl /1000.0
       # tvl.backward(retain_graph=True)
        tvl = total_variation_loss(depth.unsqueeze(0).unsqueeze(0))
        tvl = tvl / 10000.0
        tvl.backward(retain_graph=True)

        #Entropy loss. TODO: Need a real global step
        lambda_entropy = 1e-3 * min(1, 2 * i/self.n_steps)
        entropy_loss = lambda_entropy * weight_entropy
        entropy_loss.backward(retain_graph=True)

        #self.depth_weight = 3e-4
        if self.depth_weight > 0:
            center_depth = depth[7:-7, 7:-7]
            border_depth_mean = (depth.sum() - center_depth.sum()) / (64*64-50*50)
            center_depth_mean = center_depth.mean()
            depth_diff = center_depth_mean - border_depth_mean
            depth_loss = - torch.log(depth_diff + 1e-12)
            depth_loss = self.depth_weight * depth_loss
            depth_loss.backward(retain_graph=True)

       # lsl = laplacian_sharpness_loss(depth)
       # lsl = lsl*0.0001
       # lsl.backward(retain_graph=True)

        #emptiness_loss = torch.log(1 + self.emptiness_scale * ws).mean()
        #emptiness_loss = self.emptiness_weight * emptiness_loss
        #if self.emptiness_step * self.n_steps <= i:
        #    emptiness_loss *= self.emptiness_multiplier
        #emptiness_loss.backward()

       # distance_to_zero = torch.abs(ws - 0)
       # distance_to_one = torch.abs(ws - 1)
       # min_distance = torch.min(distance_to_zero, distance_to_one)
       # alpha_loss = torch.mean(min_distance)
       # alpha_loss.backward(retain_graph=True)

        #print("Density grad norm: ", self.vox.density.grad.norm())
        # Print gradient norm of all params in the app_net
        #for param in self.vox.app_net.parameters():
         #   print("App net grad norm: ", param.grad.norm())
       # if (self.vox.density.grad != None):
        #    self.vox.density.grad /= 10.0
        #    if (self.vox.density.grad.norm().item() > 1000):
        #        self.vox.density.grad /= 50.0
    
        self.metric.put_scalars(**tsr_stats(y))

      
        if i % 50 == 0:
            with torch.no_grad():
                y = self.model.decode(y)
                vis_routine(self.metric, y, depth,p,depth_map[0])
       
        self.metric.step()
        self.pbar.update()

        self.pbar.set_description(p)
        self.hbeat.beat()

@torch.no_grad()
def evaluate(score_model, vox, poser):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break
        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose, use_app_net=True)
        y = score_model.decode(y)
        vis_routine(metric, y, depth,"",None)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "video", ".mp4","",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "img")[1])
    )

    metric.step()

def render_one_view(vox, aabb, H, W, K, pose, return_w=False, embed_fr = 1.0, use_app_net = False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    
    ro, rd, t_min, t_max = scene_box_filter_(ro, rd, aabb)

    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max, embed_fr, use_app_net)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    alphas = weights.clamp(1e-5, 1 - 1e-5)
    # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
    weight_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean() 
    if return_w:
        return rgbs, depth, weights, weight_entropy
    else:
        return rgbs, depth


def scene_box_filter_(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth,prompt,depth_map):
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    
    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png","",lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png",prompt, lambda fn: imwrite(fn, im))
    if depth_map != None:
        metric.put_artifact("PC_depth", ".png",prompt, lambda fn: imwrite(fn, depth_map.cpu().squeeze()))
    metric.put_artifact("depth", ".npy","",lambda fn: np.save(fn, depth))


if __name__ == "__main__":
    def force_cudnn_initialization():
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    force_cudnn_initialization()
    dispatch(SJC_3DFuse)
    #dispatch(WriteNerf)
