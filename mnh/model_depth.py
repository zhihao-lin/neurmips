from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_camera import *
from .utils_model import grid_sample_planes
from .model_plane import PlaneGeometry

class DepthModel(nn.Module):
    def __init__(
        self,
        plane_num: int, 
        plane_res: Tuple[int], # resolution of explicit pixels
        image_size: Tuple[int],
        init_lrf_neighbors:int=50,
        init_wh:float=1.0,
        init_points:int=None
    ):
        super().__init__()
        self.plane_num = plane_num
        self.image_size = image_size
        self.plane_geo = PlaneGeometry(plane_num)
        alpha_h, alpha_w = plane_res
        self.plane_alpha = nn.Parameter(torch.FloatTensor(plane_num, 1, alpha_h, alpha_w))
        self.plane_alpha.data.uniform_(-1, 1)
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

    def freeze_geometry(self):
        '''Prevent updating the parameters in palne_geo'''
        for param in self.plane_geo.parameters():
            param.requires_grad = False

    def compute_geometry_loss(self, points):
        return self.plane_geo(points)

    def forward(self, camera):
        '''
        Return:
            depth maps: image_size
        '''
        device = camera.device
        ndc_grid = get_ndc_grid(self.image_size).to(device)
        ndc_points = ndc_grid.view(-1, 3) #(img_h*img_w=sample_n, 3)
        points_depth = get_depth_on_planes(
            self.plane_geo.basis(),
            self.plane_geo.position(),
            camera,
            ndc_points    
        ) # (plane_n, sample_n)

        K_matrix = camera.get_projection_transform().get_matrix()
        f1 = K_matrix[0, 2 ,2]
        f2 = K_matrix[0, 3, 2]
        sdepth = (f1 * points_depth + f2) / points_depth
        xy_depth = ndc_points[None].repeat(self.plane_num, 1, 1) # (plane_n, smaple_n, 3)
        xy_depth[:, :, -1] = sdepth        
        # xy_depth[:,:,-1] = points_depth
        # world_points = camera.unproject_points(xy_depth)
        cam2world = camera.get_full_projection_transform().inverse().get_matrix()
        world_points = transform_points_batch(xy_depth, cam2world)
        
        world2planes = get_transform_matrix(
            self.plane_geo.basis(),
            self.plane_geo.position()
        ) # (plane_n, 4, 4)
        # project to each planes
        plane_points = transform_points_batch(world_points, world2planes)
        plane_points = plane_points[...,:-1] # (plane_n, sample_n, 2)

        # sample values on planes
        points_alpha, in_planes = grid_sample_planes(
            sample_points=plane_points, 
            planes_wh=self.plane_geo.size(),
            planes_content=self.plane_alpha,
            mode='bilinear' #bilinear/nearest
        ) # (plane_n, sample_n, 1), (plane_n, sample_n)

        points_alpha[in_planes == True]  = torch.sigmoid(points_alpha[in_planes])
        points_alpha[in_planes == False] = 0
        # compose: for each sample(pixel), calculate expected depth
        points_alpha = points_alpha.squeeze()
        depth_sorted, sort_idx = torch.sort(points_depth, dim=0, descending=False) # ascending
        plane_n, sample_n = points_alpha.size()
        alpha_sorted = points_alpha[sort_idx, torch.arange(sample_n)[None]]
        alpha_comp = torch.cumprod(1 - alpha_sorted, dim=0) / (1 - alpha_sorted) # cumulative (1 - alpha)
        alpha = alpha_sorted * alpha_comp
        depth_expected = torch.sum(depth_sorted * alpha, dim=0)
        depth_map = depth_expected.view(*self.image_size)    
        output = {
            'depth': depth_map,
            'alpha': alpha,
            'points_depth': points_depth, #(p, s),
            'in_planes': in_planes #(p, s)
        }
        return output


def test_depth_model():
    from .utils import Timer
    from .depth import get_replica_datasets
    device = 'cuda'
    folder = '../BlenderProc/mnh/captures/room0_f13-x-10'
    train_dataset, test_dataset = get_replica_datasets(folder)
    model = DepthModel(
        plane_num=100,
        plane_res=[128, 128],
        image_size=[128, 128],
    ).to(device)
    data = train_dataset[0]
    camera = data['camera'].to(device)
    depth_gt = data['depth'].to(device)
    timer = Timer()
    out = model(camera)
    depth_pred = out['depth']
    time_per_frame = timer.get_time()
    print('forward time: {:.5f} s'.format(time_per_frame))
    print('FPS: {:.3f}'.format(1/time_per_frame))


if __name__ == '__main__':
    test_depth_model()