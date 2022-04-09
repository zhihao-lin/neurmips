from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_camera import *
from .utils_model import *
from .plane_geometry import PlaneGeometry
from .nerf.harmonic_embedding import HarmonicEmbedding
from .implicit_function import NeuralRadianceField
from .utils import *

def oscillate_ndc_grid(ndc_grid):
    '''
    oscillate NDC girds within pixel w/h when trainig -> anti-aliasing
    Args & Return:
        ndc_grid: (h, w, 3)
    '''
    h, w, _ = ndc_grid.size()
    device = ndc_grid.device
    half_pix_w = 1.0 / w
    half_pix_h = 1.0 / h
    noise_w = (torch.rand(h, w, device=device) - 0.5) * 2 * half_pix_w
    noise_h = (torch.rand(h, w, device=device) - 0.5) * 2 * half_pix_h
    ndc_grid[:,:,0] += noise_w
    ndc_grid[:,:,1] += noise_h
    return ndc_grid

def oscillated_points_along_rays(
    camera,
    xy_depth,
    ray_interval,
):  
    '''
    Args
        xy_depth: (..., 3)
    Return
        world points along rays oscillated within interval
        uniform distribution
    '''
    depths = xy_depth[..., -1]
    depth_upper = depths + ray_interval
    depth_lower = depths - ray_interval
    depth_upper = torch.clamp(depth_upper, min=1)
    depth_lower = torch.clamp(depth_lower, min=1)
    sample_weight = torch.rand_like(depths)
    depth_sampled = (1-sample_weight) * depth_lower + sample_weight * depth_upper

    xy_depth = torch.cat([xy_depth[...,:2], depth_sampled.unsqueeze(-1)], dim=-1)
    world_points_oscillated = unproject_points(camera, xy_depth)
    return world_points_oscillated

class Model_pixel(nn.Module):
    def __init__(
        self,
        plane_num: int, 
        plane_res: Tuple[int], # resolution of explicit pixels
        image_size: Tuple[int],
        anti_aliasing=False,
        init_lrf_neighbors:int=50,
        init_wh:float=1.0,
        init_points:int=None
    ):
        super().__init__()
        self.plane_num = plane_num
        self.image_size = image_size
        self.plane_geo = PlaneGeometry(plane_num)
        res_h, res_w = plane_res
        self.plane_content = nn.Parameter(torch.FloatTensor(plane_num, 1+3, res_h, res_w))
        self.plane_content.data.uniform_(-1, 1)
        self.anti_aliasing = anti_aliasing
        self.init = {
            'lrf_neighbors': init_lrf_neighbors,
            'wh': init_wh,
            'points': init_points
        }

    def initialize_geometry(self, points):
        sample_num = self.init['points']
        if sample_num != None and sample_num < points.size(0):
            sample_idx = torch.randperm(sample_num)
            points = points[sample_num]

        self.plane_geo.initialize(
            points,
            self.init['lrf_neighbors'],
            self.init['wh']
        )

    def compute_geometry_loss(self, points):
        return self.plane_geo(points)

    def forward(self, camera):
        '''
        Return:
            depth maps: image_size
        '''
        device = camera.device
        ndc_grid = get_ndc_grid(self.image_size).to(device)
        if self.training and self.anti_aliasing:
            ndc_grid = oscillate_ndc_grid(ndc_grid)
        ndc_points = ndc_grid.view(-1, 3) #(img_h*img_w=sample_n, 3)
        points_depth = get_depth_on_planes(
            self.plane_geo.basis(),
            self.plane_geo.position(),
            camera,
            ndc_points    
        ) # (plane_n, sample_n)
        xy_depth = ndc_points[None].repeat(self.plane_num, 1, 1)
        xy_depth[..., -1] = points_depth 
        
        # transform
        world_points  = unproject_points(camera, xy_depth)
        world2planes = get_transform_matrix(
            self.plane_geo.basis(),
            self.plane_geo.position()
        ) # (plane_n, 4, 4)
        plane_points = rotate_translate(world_points, world2planes)
        plane_points = plane_points[...,:-1] # (plane_n, sample_n, 2)

        # sample values on planes
        points_content, in_planes = grid_sample_planes(
            sample_points=plane_points, 
            planes_wh=self.plane_geo.size(),
            planes_content=self.plane_content,
            mode='bilinear' #bilinear/nearest
        ) # (plane_n, sample_n, 4), (plane_n, sample_n)

        points_content[in_planes == True]  = torch.sigmoid(points_content[in_planes])
        points_content[in_planes == False] = 0

        # compose: for each sample(pixel), calculate expected depth
        depth_sorted, sort_idx = torch.sort(points_depth, dim=0, descending=False) # ascending
        sample_n = points_content.size(1)
        content_sorted = points_content[sort_idx, torch.arange(sample_n)[None]]
        alpha_sorted, color_sorted = content_sorted[:,:,0], content_sorted[:,:,1:]
        alpha_comp = torch.cumprod(1-alpha_sorted, dim=0)
        alpha_comp = torch.cat([torch.ones(1, sample_n).to(device), alpha_comp], dim=0)[:-1,:] # cumulative product of (1 - alpha)
        alpha_weight = alpha_sorted * alpha_comp #(plane_n, sample_n)

        depth_expected = torch.sum(depth_sorted * alpha_weight, dim=0)
        depth_img = depth_expected.view(*self.image_size)   
        color_composite = torch.sum(color_sorted * alpha_weight.unsqueeze(-1), dim=0) #(sample, 3)
        color_img = color_composite.view(*self.image_size, 3)

        output = {
            'color': color_img,
            'depth': depth_img,
            'alpha': alpha_weight,
            'points_depth': points_depth, #(p, h*w),
            'in_planes': in_planes, #(p, h*w),
            'sort_idx': sort_idx #(p, h*w)
        }
        return output

class Model_rf(nn.Module):
    def __init__(
        self,
        plane_num: int,
        image_size: Tuple[int],
        # Radiance field 
        n_harmonic_functions_pos:int,
        n_harmonic_functions_dir:int,
        n_hidden_neurons_pos:int,
        n_hidden_neurons_dir:int,
        n_layers:int,
        # train & test
        n_train_sample:int, 
        n_infer_sample:int, 
        anti_aliasing:bool,
        premultiply_alpha: bool,
        # accelerate
        n_bake_sample:int,
        bake_res:int,
        filter_bar:float,
        white_bg:bool
    ):
        super().__init__()
        self.plane_num = plane_num
        self.plane_geo = PlaneGeometry(plane_num)
        self.image_size = image_size
        self.ndc_grid = get_ndc_grid(image_size)

        self.radiance_field = NeuralRadianceField(
            n_harmonic_functions_pos,
            n_harmonic_functions_dir,
            n_hidden_neurons_pos,
            n_hidden_neurons_dir,
            n_layers,
        )

        self.n_train_sample = n_train_sample 
        self.n_infer_sample = n_infer_sample
        self.anti_aliasing = anti_aliasing
        self.premultiply_alpha = premultiply_alpha

        self.planes_alpha = None
        self.n_bake_sample = n_bake_sample
        self.bake_res = bake_res
        self.filter_bar = filter_bar 
        self.white_bg = white_bg

    def compute_geometry_loss(self, points):
        return self.plane_geo(points)

    def bake_planes_alpha(self):
        resolution = self.bake_res
        planes_points = self.plane_geo.get_planes_points(resolution) #(plane_n, res, res, 3)
        planes_points = planes_points.view(-1, 3)
        points_total_n = (resolution ** 2) * self.plane_num
        sample_n = self.n_bake_sample
        chunk_n = math.ceil(points_total_n / sample_n)
        planes_alpha = []
        with torch.no_grad():
            for i in range(chunk_n):
                start = i * sample_n
                end = min((i+1)*sample_n, points_total_n)
                points = planes_points[start:end,:]
                dirs = torch.zeros_like(points)
                rgba = self.radiance_field(points, dirs)
                rgba = rgba.detach()
                alpha = rgba[..., -1]
                planes_alpha.append(alpha)
        planes_alpha = torch.cat(planes_alpha, dim=0)
        planes_alpha = planes_alpha.view(self.plane_num, 1, resolution, resolution)
        self.planes_alpha = planes_alpha #(plane_n, 1, res, res)
        torch.cuda.empty_cache()
        print('Baked planes alpha as [{} * {}]'.format(resolution, resolution))

    def ray_plane_intersect(self, camera, ndc_points):
        '''
        Return
            world_points: (plane_n, point_n, 3)
            planes_points: (plane_n, point_n, 2)
            planes_depth:  (plane_n, point_n)
            hit:          (plane_n, point_n)
        '''
        planes_basis = self.plane_geo.basis()
        planes_center = self.plane_geo.position() #(plane_n, 3)
        planes_depth, world_points = ray_plane_intersection(
            planes_basis,
            planes_center,
            camera, 
            ndc_points
        )

        xy_basis = planes_basis[:, :, :2] #(plane_n, 3, 2)
        planes_points = torch.bmm(world_points - planes_center.unsqueeze(1), xy_basis) #(plane_n, point_n, 2)
        
        in_planes = check_inside_planes(planes_points, self.plane_geo.size()) #(plane_n, point_n)
        hit = torch.logical_and(in_planes, planes_depth > 0) #(plane_n, point_n)
        return world_points, planes_points, planes_depth, hit

    def sort_depth_index(self, planes_depth):
        '''
        sort points along ray with depth to planes 
        Args
            planes_depth: (plane_n, point_n)
        Return
            sort_id_0, sort_id_0
        '''
        depth_sorted, sort_id_0 = torch.sort(planes_depth, dim=0, descending=False) # ascending
        point_n = planes_depth.size(1)
        sort_id_1 = torch.arange(point_n)[None].to(sort_id_0.device)
        sort_idx = [sort_id_0, sort_id_1]
        return depth_sorted, sort_idx

    def predict_points_rgba(self, camera, world_points, hit):
        '''
        Return
            points_rgba: (plane_n, point_n, 4)
        '''
        points = world_points[hit]
        view_dirs = get_normalized_direction(camera, points)
        rgba = self.radiance_field(points, view_dirs)
        points_rgba = world_points.new_zeros(*world_points.shape[:2], 4)
        points_rgba[hit] = rgba  
        return points_rgba

    def alpha_composite(self, rgb, alpha, depth):
        '''
        Return
            color: (point_n, 3)
            depth: (point_n)
        '''
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        depth = torch.sum(depth * alpha_weight, dim=0)   
        color = torch.sum(rgb * alpha_weight.unsqueeze(-1), dim=0) #(piont_n, 3)
        
        if self.white_bg:
            alpha_sum = torch.sum(alpha_weight, dim=0).unsqueeze(-1) #(point_n, 1)
            white = torch.ones_like(color) #(point_n, 3)
            color = color + (1 - alpha_sum) * white
        return color, depth

    def no_hit_output(self, ndc_points):
        if self.white_bg:
            color_bg = torch.ones_like(ndc_points)
        else:
            color_bg = torch.zeros_like(ndc_points)
        
        point_n = ndc_points.size(0)
        device = ndc_points.device 
        dummy_output = {
            'color': color_bg, #(point_n, 3)
            'depth': torch.zeros(point_n, device=device), #(point_n, )
            # 'points': torch.zeros(point_n, 3, device=device), #(point_n, 3)
            'eval_num': torch.zeros(point_n, device=device), #(point_n, )
        }
        return dummy_output

    def sample_baked_alpha(self, planes_points, hit):
        '''
        Return 
            alpha_sample: (plane_n, point_n)
        '''
        alpha_sample = grid_sample_planes(
            sample_points=planes_points, 
            planes_wh=self.plane_geo.size(),
            planes_content=self.planes_alpha,
            mode='nearest',
            padding_mode='border'
        ) # (plane_n, point_n, 1)
        alpha_sample = alpha_sample.squeeze(-1)
        alpha_sample[hit==False] = 0
        return alpha_sample

    def process_ndc_points(self, 
        camera, 
        ndc_points
    ):
        '''
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        '''
        world_points, _, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)
        
        rgba = self.predict_points_rgba(camera, world_points, hit)
        depth, sort_idx = self.sort_depth_index(planes_depth)
        rgba  = rgba[sort_idx]
        rgb, alpha = rgba[:,:,:-1], rgba[:,:,-1]
        color, depth = self.alpha_composite(rgb, alpha, depth)

        # compute unprojected points with predicted depth
        # points = unproject_points(camera, ndc_points, depth)
        # points = points.squeeze(0).detach()
        eval_num = torch.sum(hit, dim=0).float() #(sample_n)

        output = {
            'color': color, #(s, 3)
            'depth': depth, #(s, )
            # 'points': points, #(s, 3)
            'eval_num': eval_num, #(s,)
        }
        return output

    def process_ndc_points_with_alpha(self, 
        camera,
        ndc_points,
    ):
        '''
        Args
            ndc_points: (point_n, 3)
            camera: pytorch3d camera
        '''
        world_points, planes_points, planes_depth, hit = self.ray_plane_intersect(camera, ndc_points)
        if hit.any() == False:
            return self.no_hit_output(ndc_points)

        alpha_baked = self.sample_baked_alpha(planes_points, hit)
        depth, sort_idx = self.sort_depth_index(planes_depth)
        alpha = alpha_baked[sort_idx] #(plane_n, point_n)
        alpha_weight = compute_alpha_weight(alpha, normalize=self.premultiply_alpha)
        contrib = alpha_weight > self.filter_bar
        alpha[contrib == False] = 0
        
        world_points = world_points[sort_idx] #(plane_n, point_n, 3)
        hit = hit[sort_idx] #(plane_n, point_n)
        hit = torch.logical_and(hit, contrib)
        
        rgba = self.predict_points_rgba(camera, world_points, hit)
        # rgb = rgba[:,:,:-1]
        rgb, alpha = rgba[:,:,:-1], rgba[:,:,-1]
        color, depth = self.alpha_composite(rgb, alpha, depth)
    
        # compute unprojected points with predicted depth
        # points = unproject_points(camera, ndc_points, depth)
        # points = points.squeeze(0).detach()
        eval_num = torch.sum(hit, dim=0).float() #(point_n)

        output = {
            'color': color, #(s, 3)
            'depth': depth, #(s, )
            # 'points': points, #(s, 3)
            'eval_num': eval_num, #(s,)
        }
        return output

    def process(self, camera, ndc_points):
        out = None
        if self.planes_alpha is not None:
            out = self.process_ndc_points_with_alpha(camera, ndc_points)
        else:
            out = self.process_ndc_points(camera, ndc_points)
        return out 

    def ndc_points_full(self, camera):
        '''
        Return:
            NDC points: (img_h*img_w, 3)
        '''
        device = camera.device
        self.ndc_grid = self.ndc_grid.to(device)
        ndc_grid = self.ndc_grid.clone()
        if self.training and self.anti_aliasing:
            ndc_grid = oscillate_ndc_grid(ndc_grid)
        ndc_points = ndc_grid.view(-1, 3) #(img_h*img_w, 3)
        return ndc_points

    def forward_train(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        # sample_idx = torch.randperm(img_pixel_num)[:self.n_train_sample] #(n_train_sample, )
        sample_idx = torch.rand(self.n_train_sample)
        sample_idx = (sample_idx * img_pixel_num).long()
        ndc_points = ndc_points_full[sample_idx] #(n_train_sample, 3)
        output = self.process(camera, ndc_points)
        output['sample_idx'] = sample_idx
        return output 

    def forward_full_image(self, camera, ndc_points_full):
        img_pixel_num = ndc_points_full.size(0)
        if self.n_infer_sample > 0:
            sample_num = self.n_infer_sample
        else:
            sample_num = img_pixel_num
        
        chunk_num = math.ceil(img_pixel_num / sample_num)
        chunk_outputs = []
        for i in range(chunk_num):
            start = i * sample_num
            end = min((i+1) * sample_num, img_pixel_num)
            ndc_points = ndc_points_full[start:end]
            chunk_out = self.process(camera, ndc_points)
            chunk_outputs.append(chunk_out)
        
        # aggregate output from all chunks
        img_wh = self.image_size
        shapes = {
            'color': [*img_wh, 3],
            'depth': [*img_wh],
            # 'points': [-1, 3],
            'eval_num': [-1]
        }
        output = {
            key: torch.cat([
                chunk_out[key] for chunk_out in chunk_outputs
            ], dim=0).view(*shape)
            for key, shape in shapes.items()
        }
        return output 

    def forward(self, camera):
        ndc_points_full = self.ndc_points_full(camera)

        if self.training: 
            output = self.forward_train(camera, ndc_points_full)
        else: 
            output = self.forward_full_image(camera, ndc_points_full)
        return output

class RadianceField(nn.Module):
    def __init__(
        self,
        n_harmonic_functions_pos:int=6,
        n_harmonic_functions_dir:int=4,
        n_hidden_neurons_pos:int=256,
        n_hidden_neurons_dir:int=128,
        n_layers:int=8,
        pos_dim:int=3
    ):
        '''
        This module follows the design of nerf-pytorch3d/implicit_function.py 
        '''
        super().__init__()
        self.harmonic_embedding_pos = HarmonicEmbedding(n_harmonic_functions_pos)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        dir_dim = 3
        embedding_dim_xyz = n_harmonic_functions_pos * 2 * pos_dim + pos_dim
        embedding_dim_dir = n_harmonic_functions_dir * 2 * dir_dim + dir_dim

        layers = []
        for i in range(n_layers):
            in_dim = n_hidden_neurons_pos
            out_dim = n_hidden_neurons_pos
            acti = nn.ReLU(True)
            if i == 0:
                in_dim = embedding_dim_xyz
            layers += [nn.Linear(in_dim, out_dim), acti]
        self.pos_net = nn.Sequential(*layers)

        self.alpha_layer = nn.Linear(n_hidden_neurons_pos, 1)
        self.color_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons_pos + embedding_dim_dir, n_hidden_neurons_dir),
            nn.ReLU(True),
            nn.Linear(n_hidden_neurons_dir, 3)
        )

    def forward(self, 
        points, 
        directions
    ):
        '''
        Args
            points: (..., 3)
            directions: normalized (..., 3)
        Return 
            RGBA values: (..., 4)        
        '''
        harmonic_pos = self.harmonic_embedding_pos(points)
        harmonic_pos = torch.cat([harmonic_pos, points], dim=-1)
        harmonic_dir = self.harmonic_embedding_dir(directions)
        harmonic_dir = torch.cat([harmonic_dir, directions], dim=-1)

        features = self.pos_net(harmonic_pos)
        alpha = self.alpha_layer(features)

        features_dir = torch.cat([features, harmonic_dir], dim=-1)
        color = self.color_layer(features_dir)

        output = torch.cat([color, alpha], dim=-1)
        output = torch.sigmoid(output)
        return output

def test_radiance_field():
    points = torch.randn(1024, 3)
    direcs = torch.randn(1024, 3)

    model = RadianceField()
    output = model(points, direcs)
    print(output.size())

if __name__ == '__main__':
    test_radiance_field()
