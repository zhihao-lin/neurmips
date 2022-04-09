import os
import argparse
import vedo
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from pytorch3d.renderer import PerspectiveCameras, FoVPerspectiveCameras
from typing import List
from .utils_camera import *
from .utils import get_image_tensors

def get_points_from_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[3:]
        points = []
        for line in lines:
            xyz_str = line.split(' ')[1:4]
            xyz= [float(s) for s in xyz_str]
            points.append(xyz)
        points = np.array(points)
        points = torch.FloatTensor(points)
        return points

def filter_points(points, rate=0.98):
    '''
    filter out outlier points 
    '''
    center = torch.mean(points, dim=0)
    dist2center = torch.sum((points - center[None])**2, dim=-1) 
    ths = torch.quantile(dist2center, rate)
    inlier = dist2center < ths 
    points = points[inlier]
    return points

def ndc_to_screen(intrinsic):
    W, H, fx, fy, px, py = intrinsic
    # convert from NDC space to screen space
    half_w, half_h = W/2, H/2
    fx_new = fx * half_w
    fy_new = fy * half_h
    px_new = -(px * half_w) + half_w
    py_new = -(py * half_h) + half_h
    return [W, H, fx_new, fy_new, px_new, py_new]


class ReplicaDataset(Dataset):
    def __init__(
        self,
        folder:str,
        focal_length:float=2.7778,
        read_points:bool=False,
        filter_sparse_rate:float=0.98,
        sample_points:int=10000,
        fp16:bool = False
        # znear=0.1,
        # zfar=1000,
        # aspect_ratio=1.0,
        # fov=0.69111,
        
    ):
        R = np.load(os.path.join(folder, 'R.npy'))
        T = np.load(os.path.join(folder, 'T.npy'))
        R, T = torch.tensor(R), torch.tensor(T)
        if fp16:
            R, T = R.half(), T.half()
        self.R = R 
        self.T = T 
        self.focal_length = ((focal_length, focal_length),)
        self.principal_point = ((0, 0),)

        intrinsic = [512, 512, focal_length, focal_length, 0, 0]
        self.intrinsic = ndc_to_screen(intrinsic)  # W, H, fx, fy, px, py
        
        cameras = []
        for i in range(R.size(0)):
            # cam = FoVPerspectiveCameras(
            #     znear=znear,
            #     zfar=zfar,
            #     aspect_ratio=aspect_ratio,
            #     fov=fov,
            #     degrees=False,
            #     R=R[i][None],
            #     T=T[i][None]
            # )
            cam  = PerspectiveCameras(
                focal_length=self.focal_length,
                principal_point=self.principal_point,
                R = R[i][None],
                T = T[i][None]
            )
            cameras.append(cam)
        self.cameras = cameras 

        images = get_image_tensors(os.path.join(folder, 'images'))
        depths = np.load(os.path.join(folder, 'depth.npy'))
        depths = torch.tensor(depths) #(N, h, w)
        if fp16:
            images = images.half()
            depths = depths.half()
        self.images = images
        self.depths = depths

        self.sparse_points = None
        self.dense_points = None
        self.have_points = read_points
        if read_points:
            sparse_path = os.path.join(folder, 'triangulate/points3D.txt')
            sparse_points = get_points_from_file(sparse_path)
            sparse_points = filter_points(sparse_points, filter_sparse_rate)
            dense_path = os.path.join(folder, 'dense/points3D.txt')
            dense_points = get_points_from_file(dense_path)
            if fp16:
                sparse_points = sparse_points.half()
                dense_points = dense_points.half()
            self.sparse_points = sparse_points
            self.dense_points = dense_points

        self.sample_points = sample_points 

    def get_camera_centers(self):
        R, T = self.R, self.T 
        centers = torch.bmm(R, -T.unsqueeze(-1)).squeeze(-1) 
        return centers

    def __len__(self):
        return len(self.cameras)
    
    def __getitem__(self, index):
        if self.have_points:
            dense_n = self.dense_points.size(0)
            # sample_idx = torch.randperm(dense_n)[:self.sample_points]
            sample_idx = torch.rand(self.sample_points)
            sample_idx = (sample_idx * dense_n).long()
            points = self.dense_points[sample_idx]
        else: 
            points = torch.zeros(self.sample_points, 3)
    
        data = {
            'camera': self.cameras[index],
            'color': self.images[index],
            'depth': self.depths[index],
            'points': points,
        }
        return data


def unproject_depth_points(depth, camera):
    '''
    Unproject depth points into world coordinates 
    '''
    # print(camera.get_full_projection_transform().get_matrix())
    size = list(depth.size())
    ndc_grid = get_ndc_grid(size).to(depth.device) #(h, w)
    ndc_grid[..., -1] = depth
    xy_depth = ndc_grid.view(1, -1, 3)
    points = camera.unproject_points(xy_depth)[0]
    return points

def dataset_to_depthpoints(dataset, point_num=None):
    '''
    Unproject all depth points within dataset into world coordinates 
    Args
        dataset
    Return
        points: (point_num, 3)
    '''
    points_all = []
    for i in range(len(dataset)):
        data = dataset[i]
        color = data['color']
        depth = data['depth']
        camera = data['camera']
        points = unproject_depth_points(depth, camera)
        points_all.append(points)
    
    points = torch.cat(points_all, dim=0)
    if point_num is not None:
        sample_idx = torch.randperm(points.size(0))[:point_num]
        points = points[sample_idx]
    return points

def dataset_to_rgbd(dataset, point_num=None):
    points_all = []
    colors_all = []
    for i in range(len(dataset)):
        data = dataset[i]
        color = data['color']
        color = color.view(-1, 3)
        depth = data['depth']
        camera = data['camera']
        points = unproject_depth_points(depth, camera)
        points_all.append(points)
        colors_all.append(color)
    
    points = torch.cat(points_all, dim=0)
    colors = torch.cat(colors_all, dim=0)

    if point_num is not None:
        sample_idx = torch.randperm(points.size(0))[:point_num]
        points = points[sample_idx]
        colors = colors[sample_idx]
    return points, colors

def chamfer_uni(points_source, points_target):
    dist_vec = points_source.unsqueeze(1) - points_target.unsqueeze(0) #(s_n, t_n, 3)
    dist = torch.sum(dist_vec**2, dim=-1)
    dist_min,_ = torch.min(dist, dim=1)
    cd_loss = torch.sum(dist_min).item()
    return cd_loss 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='dataset')
    parser.add_argument('-folder', default='data/replica/room0/train')
    parser.add_argument('-out')
    args = parser.parse_args()
    folder = args.folder

    if args.mode == 'camera':
        from .utils_vedo import get_vedo_cameras_cones
        folder = args.folder
        train_dataset = ReplicaDataset(os.path.join(folder, 'train'))
        valid_dataset = ReplicaDataset(os.path.join(folder, 'valid'))
        
        points_cfg = {
            'num': 50000,
            'r': 5
        }
        points_t, colors_t = dataset_to_rgbd(train_dataset, point_num=points_cfg['num'])
        points_v, colors_v = dataset_to_rgbd(valid_dataset, point_num=points_cfg['num'])
        points = torch.cat([points_t, points_v], dim=0)
        colors = torch.cat([colors_t, colors_v], dim=0)
        colors = colors.numpy()
        points = vedo.Points(points, c=colors, r=points_cfg['r'])

        cones_cfg = {
            'r': 0.05,
            'height': 0.15,
            'alpha': 0.5
        }
        cam_train = get_vedo_cameras_cones(
            train_dataset.R, train_dataset.T, 
            cones_cfg['r'], cones_cfg['height'], 
            color=(0, 0, 1), alpha=cones_cfg['alpha']
        )
        cam_valid = get_vedo_cameras_cones(
            valid_dataset.R, valid_dataset.T, 
            cones_cfg['r'], cones_cfg['height'], 
            color=(1, 0, 0), alpha=cones_cfg['alpha']
        )

        vedo.show(points, cam_train, cam_valid)

    
    if args.mode == 'dataset':
        dataset = ReplicaDataset(folder, read_points=True, sample_points=10000)
        print('sparse points', dataset.sparse_points.size())
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
        for i, data in enumerate(dataloader):
            data = data[0]
            depth = data['depth']
            camera = data['camera']
            points = data['points']
            K = camera.get_projection_transform().get_matrix()
            # print(K)
            d_min, d_max = depth.min(), depth.max()
            print('d_min: {:.3f} | d_max: {:.3f}'.format(d_min, d_max))
            print('points size: {}'.format(points.size()))

    if args.mode == 'test':
        dataset = ReplicaDataset(folder, read_points=True, sample_points=10000)
        pt_gt = dataset_to_depthpoints(dataset, point_num=10000)
        pt_sp = dataset.sparse_points 
        pt_dp = dataset.dense_points 
        
        print('# sparse points = {}'.format(pt_sp.size(0)))
        print('# dense  points = {}'.format(pt_dp.size(0)))
        
        pt_sp = vedo.Points(pt_sp, c=[0, 0, 1])
        pt_dp = vedo.Points(pt_dp, c=[1, 0, 0])
        pt_gt  = vedo.Points(pt_gt, c=[0.5, 0.5, 0.5])
        vedo.show(pt_sp, pt_gt)
        vedo.show(pt_dp, pt_gt)
        vedo.show(pt_sp, pt_dp)


if __name__ == '__main__':
    main()


    