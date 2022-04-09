import os
import sys
import h5py
import numpy as np
import torch
import argparse 
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras
from .utils_rotation import rotation_to_quaternion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-blender_root', default='../BlenderProc')
    parser.add_argument('-replica_data', default='../Replica-Dataset/dataset/')
    parser.add_argument('-output_path', default='data/test')
    # camera intrinsic
    parser.add_argument('-znear', type=float, default=0.1)
    parser.add_argument('-zfar', type=float, default=1000)
    parser.add_argument('-aspect_ratio', type=float, default=1.0)
    parser.add_argument('-fov', type=float, default=0.69111)
    args = parser.parse_args()
    
    current_path = os.path.abspath('.')
    folder_root = get_folder_path(current_path, args.output_path)

    blender_root = args.blender_root   
    exe = os.path.join(blender_root, 'run.py')
    config_file = os.path.join(blender_root, 'mnh/config.yaml')
    cam_pose_path = os.path.join(blender_root, 'mnh/camera_pose')
    replica_path = os.path.join(current_path, args.replica_data)
    folder_blender = get_folder_path(folder_root, 'blenderproc')

    # generate raw data with BlenderProc
    self_path = 'mnh/generate_replica_data.py'
    os.system(f'cp {self_path} {folder_root}')
    os.system(f'cp {config_file} {folder_blender}')
    sample_write_view(cam_pose_path)
    os.system(f'cp {cam_pose_path} {folder_blender}')
    hdf5_path = get_folder_path(folder_blender, 'hdf5')
    os.system(f'python3 {exe} {config_file} {replica_path} {hdf5_path}')
    
    # Parse data and store
    pos_rot = np.genfromtxt(cam_pose_path)
    pos, rot = pos_rot[:,:3], pos_rot[:,3:]
    R, T = convert_to_RT(pos, rot)
    R, T = R.numpy(), T.numpy()
    np.save(os.path.join(folder_root, 'R.npy'), R)
    np.save(os.path.join(folder_root, 'T.npy'), T)

    depth = get_arrays_from_hdf5_folder(hdf5_path, 'depth')
    np.save(os.path.join(folder_root, 'depth.npy'), depth)
    colors = get_arrays_from_hdf5_folder(hdf5_path, 'colors') #(n, h, w, 3)
    view_n = colors.shape[0]
    folder_img = get_folder_path(folder_root, 'images')
    img_names = []
    for i in range(view_n):
        img = Image.fromarray(colors[i])
        name = '{:0>5}.png'.format(i)
        path = os.path.join(folder_img, name)
        img.save(path)
        img_names.append(name)

    # construct sparse files for colmap, 
    # in order to reconstruct sparse 3D points
    folder_sparse = get_folder_path(folder_root, 'sparse')
    points_path = os.path.join(folder_sparse, 'points3D.txt')
    with open(points_path, 'w') as file:
        pass 
    intrinsic_path = os.path.join(folder_sparse, 'cameras.txt')
    with open(intrinsic_path, 'w') as file:
        h, w = colors.shape[1:3]
        px, py = w/2, h/2
        camera = FoVPerspectiveCameras()
        K = camera.compute_projection_matrix(
            znear=args.znear,
            zfar=args.zfar,
            aspect_ratio=args.aspect_ratio,
            fov=args.fov,
            degrees=False
        ) #(1, 4, 4)
        s1, s2 = K[0,0,0].item(), K[0,1,1].item()
        fx, fy = s1*w/2, s2*h/2
        line = f'1 PINHOLE {w} {h} {fx} {fy} {px} {py}\n'
        file.write(line)
        
    extrinsic_path = os.path.join(folder_sparse, 'images.txt')
    with open(extrinsic_path, 'w') as file:
        R[:,:,:2] *= -1
        T[:,:2] *= -1
        view_n = R.shape[0]
        for i in range(view_n):
            r = R[i]
            q = rotation_to_quaternion(r)
            qw, qx, qy, qz = q 
            t = T[i]
            tx, ty, tz = t
            img_id = i + 1 
            cam_id = 1 
            name = img_names[i]
            line = f'{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}\n\n'
            file.write(line)

def sample_write_view(cam_pose_path):
    # Replica 
    ############### modify for generate canera poses ###############
    views_n = 100
    splits = [50, 100]
    positions = np.zeros((views_n, 3))
    rotations = np.zeros((views_n, 3))

    positions[:splits[0]] = random_sample_values(
        splits[0] - 0, 
        x=[2.0, 4.0], 
        y=[-2.0, -1], 
        z=[0.0, 0.5]
    )
    rotations[:splits[0]] = random_sample_values(
        splits[0] - 0, 
        x=[65, 75], 
        y=[0, 0], 
        z=[-200, -160]
    ) * (np.pi/180)
    # ----------------------------------------
    positions[splits[0]:splits[1]] = random_sample_values(
        splits[1] - splits[0], 
        x=[2.8, 4.4], 
        y=[-3.2, -2.0], 
        z=[-0.1, 0.5]
    )
    rotations[splits[0]:splits[1]] = random_sample_values(
        splits[1] - splits[0], 
        x=[65, 75], 
        y=[0, 0], 
        z=[-210, -160]
    ) * (np.pi/180)
    ############### modify for generate canera poses ###############

    dump_locations_rotations(
        positions,
        rotations,
        cam_pose_path
    )

def random_sample_values(
    number: int, 
    x: list, y: list, z: list,
):  
    '''
    xyz are list with (min, max)
    Return 
        values: (num, 3)
    '''
    values = np.zeros((number, 3))
    
    a = np.random.rand(number)
    values[:, 0] = a * x[0] + (1-a) * x[1]
    a = np.random.rand(number)
    values[:, 1] = a * y[0] + (1-a) * y[1]
    a = np.random.rand(number)
    values[:, 2] = a * z[0] + (1-a) * z[1]
    return values

def linear_sample_values(
    number: int, 
    x: list, y: list, z: list,
):  
    '''
    xyz are list with (min, max)
    Return 
        values: (num, 3)
    '''
    values = np.zeros((number, 3))
    a = np.linspace(0, 1, number)

    values[:, 0] = a * x[0] + (1-a) * x[1]
    values[:, 1] = a * y[0] + (1-a) * y[1]
    values[:, 2] = a * z[0] + (1-a) * z[1]
    return values

def dump_locations_rotations(
    positions,
    rotations,
    path
):
    positions = np.array(positions).astype(np.float)
    rotations = np.array(rotations).astype(np.float)
    views_n = positions.shape[0]
    
    with open(path, 'w') as file:
        for i in range(views_n):
            p = positions[i]
            r = rotations[i]
            line = '{} {} {} {} {} {}\n'.format(
                p[0], p[1], p[2], r[0], r[1], r[2]
            )
            file.write(line)

def dump_cam2world_matrix(
    positions,
    rotations,
    path
):
    positions = np.array(positions).astype(np.float)
    rotations = np.array(rotations).astype(np.float)
    views_n = positions.shape[0]
    matrix = get_cam2world_matrix(positions, rotations)
    
    matrix *= np.array([-1, -1, 1, 1]).reshape(1, -1, 1)
    matrix = np.transpose(matrix, (0, 2, 1))
    matrix = matrix.reshape(views_n, -1)

    with open(path, 'w') as file:
        for i in range(views_n):
            m = matrix[i]
            m = [str(m[j]) for j in range(len(m))]
            line = ' '.join(m) + '\n'
            file.write(line)

def get_cam2world_matrix(
    positions, 
    rotations
): 
    '''
    Args
        cam location in world (n, 3)
        world2cam rotation (n, 3, 3)
    Return 
        transformation (cam2world) (n, 4, 4)
    '''
    n = rotations.shape[0]
    matrix = np.zeros((n, 4, 4))
    R = np.transpose(rotations, (0, 2, 1))
    matrix[:, :3, :3] = R
    matrix[:, -1, :3] = positions
    matrix[:, -1, -1] = 1
    return matrix

##### functions for loading data generated by BlenderProc
def get_array_from_hdf5(path, key, remove_noise=True):
    with h5py.File(path, 'r') as data:
        array = np.array(data[key])
        if key in ['distance', 'depth']:
            array = array[:,:,0]
        if remove_noise == True:
            bound = 1000
            array[array >= bound] = np.mean(array[array < bound])
        return array

def get_arrays_from_hdf5_folder(folder, key):
    '''
    folder: folder path of *.hdf5 files
    '''
    file_num = len([name for name in os.listdir(folder) if name.endswith('.hdf5')])
    files = [os.path.join(folder, '{}.hdf5'.format(i)) for i in range(file_num)]
    arrays = [get_array_from_hdf5(file, key) for file in files]
    arrays = np.stack(arrays, axis=0)
    return arrays

def euler2mat(euler):
    '''
    Transform euler angle in to rotation matrix 
    Input:
        Euler angle in radian: (3, ) 
    Output:
        Rotation matrix: (3, 3)
    '''
    mats = []
    for ax in range(3):
        rad = euler[ax]
        cos = np.cos(rad)
        sin = np.sin(rad)
        axes = [0, 1, 2]
        axes.remove(ax)
        rot = np.eye(3)
        rot[axes[0], axes[0]] = cos
        rot[axes[0], axes[1]] = sin
        rot[axes[1], axes[0]] = -sin
        rot[axes[1], axes[1]] = cos
        mats.append(rot)
    
    mat_x, mat_y, mat_z = mats
    mat = mat_z @ mat_y @ mat_x
    return mat

def convert_to_RT(position, rotation):
    '''
    convert position, rotation read from config file to 
    R, T which are used to initialize cameras
    Args
        position, rotation: numpy array list
    Return:
        tensor
    '''
    R = [euler2mat(rot) for rot in rotation]
    R = torch.FloatTensor(R)
    position = torch.FloatTensor(position)
    # according to the difference in OpenGL camera
    position *= torch.tensor([[-1, 1, -1]])
    o2center = - position.unsqueeze(1) #(n, 1, 3)
    T = torch.bmm(o2center, R).squeeze(1)
    return R, T

def get_folder_path(*argv):
    folder_path = os.path.join(*argv)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

if __name__ == '__main__':
    main()