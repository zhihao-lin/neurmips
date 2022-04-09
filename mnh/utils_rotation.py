import numpy as np 
import torch 
import vedo 
import math

def get_sphere_points(lat, lon, r: float):
    '''
    Args
        latitude:  (n,) [-90, 90]
        longitude: (n,) [-180, 180]
    Return
        points: (n, 3) 
    '''
    
    lat, lon = (lat/180)*np.pi, (lon/180)*np.pi
    points = np.stack((
        np.cos(lat)*np.cos(lon), 
        np.cos(lat)*np.sin(lon), 
        np.sin(lat)), axis=-1
    )
    points = points * r
    return points

def zy_to_rotations(vec_z, vec_y):  
    '''
    Args
        vec_z: (n, 3)
        vec_y: (n, 3)
    Return 
        Roation matrix (n, 3, 3)
        R = [[x1, y1, z1],
            [x2, y2, z2],
            [x3, y3, z3]]
    '''
    vec_x = np.cross(vec_y, vec_z)
    vec_y = np.cross(vec_z, vec_x)
    r = np.stack((vec_x, vec_y, vec_z), axis=-1)
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r = r / r_norm
    return r

def quaternion_to_matrix(q):
    '''
    re-write pytorch3d codes in numpy;
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Return
        matrix R: (n, 3, 3)
        transform points p with p' = pR (follow pytorch3d convention)
    '''
    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two_s = 2.0 / (q * q).sum(-1)
    
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )

    matrix = o.reshape(o.shape[0], 3, 3)
    matrix = np.transpose(matrix, (0, 2, 1))
    return matrix

def rotation_to_quaternion(R):
    # Input: rotation matrix (3 x 3)
    # [x1, y1, z1]
    # [x2, y2, z2]
    # [x3, y3, z3]
    # Output: quaternion coordinates
    # [Qw, Qx, Qy, Qz]
    q0 = 0.5*math.sqrt(1+R[0,0]+R[1,1]+R[2,2])
    q1 = (R[1,2]-R[2,1])/(4*q0)
    q2 = (R[2,0]-R[0,2])/(4*q0)
    q3 = (R[0,1]-R[1,0])/(4*q0)
    return [q0, q1, q2, q3]

def test():
    n = 20
    r = 1
    lat = np.zeros((n,))
    lat[:] = 30
    lon = np.linspace(0, 360, n)
    positions = get_sphere_points(lat, lon, r)
    
    # axes = - positions
    # theta = np.zeros((n,))
    # quaternion = get_quaternion(theta, axes)
    # rotations = quaternion_to_matrix(quaternion)
    
    vec_z = -positions
    vec_y = np.zeros((n, 3))
    vec_y[:, -1] = 1
    rotations = zy_to_rotations(vec_z, vec_y)

    show_position_rotation(positions, rotations, arrow_len=0.1)