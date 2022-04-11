import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_camera import *
import vedo

def get_cube_vertices(side_len:int=2):
    vertices = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                v = torch.FloatTensor([x, y, z])
                vertices.append(v)
    vertices = torch.stack(vertices)
    veritces = vertices * side_len/2
    return vertices

def sample_cube_points(
    points_per_plane:int,
    side_len: int
):
    cube_v = get_cube_vertices(side_len)
    points = torch.tensor([])
    for dim in range(3):
        for val in [-1, 1]:
            plane_v = cube_v[cube_v[:, dim] == val]
            o = plane_v[0][None] #(1, 3)
            d1 = plane_v[1] - plane_v[0]
            d2 = plane_v[2] - plane_v[0]
            d = torch.stack([d1, d2]) #(2, 3)
            sample_w = torch.rand(points_per_plane, 2)
            plane_points = o + sample_w @ d
            points = torch.cat([points, plane_points], dim=0)
    return points

class CubePointsData():
    def __init__(
        self,
        points_per_plane:int=1000,
        points_per_batch:int=1000,
        cube_len:float=2.0
    ):
        self.points_per_batch = points_per_batch
        cube_points = sample_cube_points(points_per_plane, cube_len)
        point_n = cube_points.size(0)
        order = torch.randperm(point_n)
        self.points = cube_points[order]

    def __len__(self):
        point_n = self.points.size(0)
        return point_n // self.points_per_batch
    
    def __getitem__(self, index):
        start = index * self.points_per_batch
        end = (index + 1) * self.points_per_batch
        return self.points[start:end]

class ScenePointData():
    def __init__(
        self,
        path:str,
        points_total:int,
        points_batch:int,
        bounds:list,
        lrf_neighbor:int=50
    ):
        mesh = vedo.Mesh(path)
        mesh.crop(bounds=bounds)
        points = mesh.points()
        index = torch.randperm(points.shape[0])[:points_total]
        points = torch.tensor(points[index])
        lrf = get_points_lrf(points, lrf_neighbor)
        normal = lrf[:,:,-1]
        # normalize
        points = points - torch.mean(points, dim=0)
        r = torch.sqrt(torch.max(torch.sum(points**2, dim=1)))
        points = points/r
        self.points = points
        self.normal = normal
        self.points_total = points_total
        self.points_batch = points_batch

    def show(self):
        points = vedo.Points(self.points, r=2, c=(0.5,0.5,0.5), alpha=0.5)
        vedo.show([points], axes=True)
    
    def __len__(self):
        return self.points_total // self.points_batch

    def __getitem__(self, index):
        start = index * self.points_batch
        end = (index + 1) * self.points_batch
        return self.points[start:end], self.normal[start:end]

def test_scene_point():
    scene_points = ScenePointData(
        path='data/mesh/room_0.ply',
        points_total=60000,
        points_batch=1000,
        bounds=[0, 6, 0, 2.5, -2, 1]
    )
    print('len: {}'.format(len(scene_points)))
    print('point batch: {}'.format(scene_points[0].size()))
    scene_points.show()

def gt_position_rotation(side_len:float=2.0):
    position, rotation = [], []
    for dim in range(3):
        for val in [-1, 1]:
            pos = torch.zeros(3)
            pos[dim] = val #position = z-axis \
            position.append(pos)

            y = torch.zeros(3)
            y[dim-1] = 1
            x = torch.cross(y, pos)
            rot = torch.stack([x, y, pos], dim=1)
            rotation.append(rot)
    position = torch.stack(position) * side_len / 2
    rotation = torch.stack(rotation)
    return position, rotation

def orthonormal_basis_from_xy(xy):
    '''
    compute orthonormal basis from xy vector: (n, 3, 2)
    '''
    x, y = xy[:,:,0], xy[:,:,1]
    z = torch.cross(x, y, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    xyz = torch.stack([x,y,z], dim=-1)
    return xyz

def orthonormal_basis_from_yz(yz):
    '''
    compute orthonormal basis from yz vector: (n, 3, 2)
    '''
    y, z = yz[:,:,0], yz[:,:,1]
    x = torch.cross(y, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    xyz = torch.stack([x,y,z], dim=-1)
    return xyz

def project_points_on_plane(
    points,
    center,
    xyz,
    wh:list,
):     
    '''
    points: (n, 3)
    center: (3, )
    xyz: orthonormal basis (3, 3)
    length: scalar
    '''
    center = center.view(1, -1)
    points_trans = torch.matmul(points - center, xyz)
    points_trans = points_trans[points_trans[:,0].abs() < wh[0]/2]
    points_trans = points_trans[points_trans[:,1].abs() < wh[1]/2]
    points_trans[:, -1] = 0
    points_project = torch.matmul(points_trans, xyz.T) + center
    return points_project

def farthest_point_sample(
    points,
    sample_n: int
):
    '''
    Input:
        points: (point_n, dim)
    Return:
        idx: (sample_n)
        points_sample: (sample_n, dim)
    '''
    idx = 0
    sample_set= [idx]
    dist2set = torch.tensor([]).to(points.device)
    for i in range(sample_n - 1):
        dist = points - points[idx]
        dist = torch.sum(dist**2, dim=1)[:, None]
        dist2set = torch.cat([dist2set, dist], dim=1)
        min_dist, _ = torch.min(dist2set, dim=1) #(point_n,)
        _, max_id = torch.max(min_dist, dim=0)
        idx = max_id.item()
        sample_set.append(idx) 

    points_sample = points[sample_set]
    sample_set = torch.LongTensor(sample_set)
    return sample_set, points_sample

def test_fps():
    sample_n = 100
    points = torch.randn(10000, 3)
    points = F.normalize(points, dim=1)
    idx, points_sample = farthest_point_sample(points, sample_n)
    points_all = vedo.Points(points, r=3, c=(0.5, 0.5, 0.5), alpha=0.8)
    points_fps = vedo.Points(points_sample, r=7, c=(1, 0, 0), alpha=1)
    vedo.show(points_all, points_fps)

def get_points_lrf(
    points,
    neighbor_num:int,
    indices,
    chunk_size:int=200
):
    '''
    Input:
        points: (point_n, 3)
        indices: (sample_n,) index of partial points -> reduce computation
    Output:
        Local reference frame at each point computed by PCA
        lrf: (point_n, 3, 3) basis are aranged in columns
    '''
    samples = points[indices] #(sample_n, 3)
    dist = samples.unsqueeze(1) - points.unsqueeze(0) #(s, p, 3)
    dist = torch.sum(dist**2, dim=-1) #(s, p)
    dist_n, neighbor_idx = torch.topk(dist, k=neighbor_num, dim=-1, largest=False)
    neighbors = points[neighbor_idx].cpu() #(s, n, 3)
    lrf_list = []
    sample_n = samples.size(0)
    chunk_n = math.ceil(sample_n/chunk_size)
    for i in range(chunk_n):
        start = i * chunk_size 
        end = min((i+1)*chunk_size, sample_n)
        U, S, V_t = torch.pca_lowrank(neighbors[start:end])
        lrf_list.append(V_t)
        # U:(s, n, n), S:(s, min(n,3)), V_t:(s, 3, 3)
    lrf = torch.cat(lrf_list, dim=0).to(points.device)
    return lrf

class PlaneModel(nn.Module):
    def __init__(
        self,
        plane_num:int=6,
        device:torch.device='cpu'
    ):
        super().__init__()
        self.plane_num = plane_num
        self.center = nn.Parameter(torch.FloatTensor(plane_num, 3))
        self.xy = nn.Parameter(torch.FloatTensor(plane_num, 3, 2))
        self.wh = nn.Parameter(torch.FloatTensor(plane_num, 2))
    
    def initialize(self, points):
        lrf = get_points_lrf(points, neighbor_num=50) #(point_n, 3, 3)
        sample_idx, center = farthest_point_sample(points, self.plane_num)
        self.center.data = center
        xyz = lrf[sample_idx]
        self.xy.data = xyz[:,:,:2]
        self.wh.data[:] = 0.5

    def forward(self, points, normal):
        '''
        Input:
            points: (point_num, 3) xyz 
            normal: (point_num, 3)
        Return:
            loss: mean square of distance from every point to nearest plane
        '''
        # compute orthonormal basis
        x, y = self.xy[:,:,0], self.xy[:,:,1]
        z = torch.cross(x, y, dim=-1)
        y = torch.cross(z, x, dim=-1)
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        z = F.normalize(z, dim=-1)
        
        diff = points.unsqueeze(0) - self.center.unsqueeze(1) #(plane_n, point_n, 3)
        dist_x = torch.bmm(diff, x.unsqueeze(-1)).squeeze() #(plane_n, point_n)
        dist_y = torch.bmm(diff, y.unsqueeze(-1)).squeeze()
        dist_z = torch.bmm(diff, z.unsqueeze(-1)).squeeze()  
        
        dist_x = torch.abs(dist_x) - self.wh[:, 0].unsqueeze(-1)/2
        dist_x[dist_x < 0] = 0
        dist_y = torch.abs(dist_y) - self.wh[:, 1].unsqueeze(-1)/2
        dist_y[dist_y < 0] = 0
        distance = dist_x**2 + dist_y**2 + dist_z**2

        min_dist, min_id = torch.min(distance, dim=0) #(point_n)
        loss_point2plane = torch.mean(min_dist)

        distance, _ = torch.sort(distance, dim=1)
        nearest_num = int(distance.size(1)/self.plane_num)
        loss_plane2point = torch.mean(distance[:,:nearest_num])

        loss_area = torch.mean(torch.abs(self.wh[:, 0] * self.wh[:, 1]))

        plane_normal = torch.cross(self.xy[:,:,0], self.xy[:,:,1], dim=-1) #(plane_n, 3)
        nearest_plane_normal = plane_normal[min_id] #(point_n, 3)
        cosine_sim = torch.abs(F.cosine_similarity(normal, nearest_plane_normal, dim=-1))
        loss_normal = 1 - torch.mean(cosine_sim)

        output = {
            'loss_point2plane': loss_point2plane,
            'loss_plane2point': loss_plane2point,
            'loss_area': loss_area,
            'loss_normal': loss_normal,
            'min_dist': min_dist,
            'min_id': min_id
        }
        return output

def test_plane_model():
    points = sample_cube_points(1000, 2)
    lrf = get_points_lrf(points, neighbor_num=20)
    normal = lrf[:, :, -1]
    print(points.size())
    model = PlaneModel()
    model.initialize(points)
    output = model(points, normal)
    output_string(output)
    visualize(points, model)

def test_plane_geometry():
    points = torch.randn(220000, 3)
    plane_n = 200
    model = PlaneGeometry(plane_n)
    model.initialize_with_box(
        points,
        lrf_neighbors=50,
        wh=0.5,
        box_factor=1.0,
        random_rate=0.2
    )

    visualize(points, model, alpha=0.5)


def visualize(
    points,
    model,
    r:float=2,
    c:list=(0.5,0.5,0.5),
    alpha:float=0.5,
    screenshot_name:str=None
):
    objs = []
    points = points.cpu().numpy() 
    points = vedo.Points(points, r=r, c=c, alpha=1)
    objs.append(points)
    
    center = model.center.detach().cpu().numpy()
    xyz = orthonormal_basis_from_xy(model.xy.detach()).detach().cpu().numpy()
    wh = model.wh.detach().cpu().numpy()

    colors = np.random.rand(model.plane_num, 3)
    for i in range(model.plane_num):
        c = center[i]
        x, y = xyz[i,:,0], xyz[i,:,1]
        x_s, y_s = x*(wh[i, 0]/2), y*(wh[i, 1]/2)
        verts = [c+x_s+y_s, c-x_s+y_s, c-x_s-y_s, c+x_s-y_s]
        faces = [[0,1,2], [2,3,0]]
        plane = vedo.Mesh([verts, faces], c=colors[i], alpha=alpha)
        objs.append(plane)
        
    vedo.show(*objs ,axes=1)
    if screenshot_name:
        vedo.io.screenshot(screenshot_name)

def output_string(output:dict):
    string = ""
    for key, value in output.items():
        if 'loss' in key:
            string += "{}: {:.5f} |".format(key, value.cpu().item())
    print(string)

if __name__ == '__main__':
    test_plane_geometry()