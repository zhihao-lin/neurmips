import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra
import vedo
from mnh.dataset_replica import ReplicaDataset
from mnh.dataset_TanksAndTemples import TanksAndTemplesDataset
from mnh.model_plane import visualize
from mnh.stats import StatsLogger, WandbLogger
from mnh.utils import *
from mnh.utils_vedo import get_vedo_alpha_plane
from mnh.utils_video import load_video_cameras
from teacher_forward import *

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs/teacher')
TEST_CONFIG = 'test'
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints/teacher')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')


@hydra.main(config_path=CONFIG_DIR, config_name=TEST_CONFIG)
def main(cfg: DictConfig):
    # Set random seed for reproduction
    set_fp16 = False
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set device for training
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cfg.cuda))
    else:
        device = torch.device('cpu')
        
    # set DataLoader objects
    train_path = os.path.join(CURRENT_DIR, cfg.data.path, 'train')
    valid_path = os.path.join(CURRENT_DIR, cfg.data.path, 'valid')
    test_path  = os.path.join(CURRENT_DIR, cfg.data.path, 'test')
    train_dataset, valid_dataset = None, None
    if 'replica' in cfg.data.path:
        train_dataset = ReplicaDataset(folder=train_path, read_points=True, sample_points=cfg.data.sample_points)
        valid_dataset = ReplicaDataset(folder=valid_path)
    elif 'Tanks' in cfg.data.path or 'BlendedMVS' in cfg.data.path:
        train_dataset = TanksAndTemplesDataset(
            folder=train_path, 
            read_points=True, 
            sample_rate=cfg.data.sample_rate,
            batch_points=cfg.data.batch_points,
        )
        valid_dataset = TanksAndTemplesDataset(
            folder=valid_path,
        )
    elif 'Synthetic' in cfg.data.path:
        train_dataset = TanksAndTemplesDataset(
            folder=train_path, 
            read_points=True, 
            sample_rate=cfg.data.sample_rate,
            batch_points=cfg.data.batch_points,
        )
        valid_dataset = TanksAndTemplesDataset(
            folder=test_path,
        )
    datasets = {
        'train': train_dataset,
        'valid': valid_dataset
    }

    model = get_model_from_config(cfg)
    model.to(device)
    model.eval()

    # load checkpoints
    checkpoint_default = '{}.pth'.format(cfg.name)
    checkpoint_name = checkpoint_default if cfg.checkpoint == '' else cfg.checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print('Load from checkpoint: {}'.format(checkpoint_name))
        loaded_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(loaded_data['model'])
        # used after finetuning
        if 'alpha' in loaded_data:
            model.planes_alpha = loaded_data['alpha'].to(device)
    else:
        # initialize plane position, rotation and size
        print('[Init] initialize plane geometry ...')
        points = train_dataset.dense_points.to(device)
        print('#points= {}'.format(points.size(0)))
        # model.plane_geo.initialize_with_box(
        #     points, 
        #     lrf_neighbors=cfg.model.init.lrf_neighbors,
        #     wh=cfg.model.init.wh,
        #     box_factor=cfg.model.init.box_factor, 
        #     random_rate=cfg.model.init.random_rate,
        # )
        model.plane_geo.initialize(
            points,
            lrf_neighbors=cfg.model.init.lrf_neighbors,
            wh=cfg.model.init.wh,
        )
        del points 
        torch.cuda.empty_cache()

    if set_fp16:
        torch.set_default_dtype(torch.float16)
        model = model.half()
        model.ndc_grid = model.ndc_grid.half()
    if cfg.model.bake == True:
        model.bake_planes_alpha()
    output_dir = os.path.join(CURRENT_DIR, 'output_images/teacher', cfg.name)
    os.makedirs(output_dir, exist_ok=True)
    
    print('Test [{}] ...'.format(cfg.test.mode))
    if cfg.test.mode == 'test_model':
        print('- Parameter number: {}'.format(parameter_number(model)))
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.optimizer.lr
        )
        train_data = train_dataset[0]
        train_stats, _ = forward_pass(train_data, model, device, cfg, optimizer, training=True)
        print(train_stats)
        print('- Train: forward + backprop')
        valid_data = valid_dataset[0]
        valid_stats, _ = forward_pass(valid_data, model, device, cfg, training=False)
        print(valid_stats)
        print('- Validation: forward')
        print('- Image inference FPS: {:.3f}'.format(valid_stats['FPS']))

    if cfg.test.mode == 'complexity':
        from mnh.complexity import calculate_complexity
        net = model.radiance_field
        inputs = {
            'points': torch.randn(10000, 3, device=device),
            'directions': torch.randn(10000, 3, device=device)
        }
        calculate_complexity(net, inputs)

    if cfg.test.mode == 'evaluate':
        stats_logger= StatsLogger()
        model.eval()
        for split, dataset in datasets.items():
            # alpha_dist = []
            for i in range(len(dataset)):
                data = dataset[i]
                stats, _ = forward_pass(data, model, device, cfg)
                stats_logger.update(split, stats)
                # alpha_dist.append(alpha)
            stats_logger.print_info(split)
            # alpha_dist = torch.stack(alpha_dist, dim=0)
            # alpha_dist = torch.mean(alpha_dist, dim=0)
            # print('Alpha: ', alpha_dist)

    if cfg.test.mode == 'render':
        folder = {}
        splits = ['valid']
        keys = ['color', 'depth']
        for key in keys:
            folder[key] = {}
            for split in splits:
                if cfg.test.folder == '':
                    path = os.path.join(output_dir, key, split)
                else:
                    path = os.path.join(output_dir, cfg.test.folder, key, split)
                os.makedirs(path, exist_ok=True)
                folder[key][split] = path
        
        stats_logger = StatsLogger()
        for split in splits:
            print('saving [{}] images ...'.format(split))
            dataset = datasets[split]
            depths_all = []
            for i in range(len(dataset)):
                data = dataset[i]
                stats, images = forward_pass(data, model, device, cfg)
                
                for label in ['gt', 'pred']:
                    img = images['color_{}'.format(label)]
                    img = tensor2Image(img)
                    path = os.path.join(folder['color'][split], '{}-{:0>5}-{}.png'.format(split, i, label))
                    img.save(path)
                depth = images['depth_pred']
                depths_all.append(depth)
                stats_logger.update(split, stats)
            
            depths_all = torch.stack(depths_all, dim=0)
            depths_all = to_numpy(depths_all)
            path = os.path.join(folder['depth'][split], 'depth.npy')
            np.save(path, depths_all)

            stats_logger.print_info(split)
            avg_time = stats_logger.stats[split]['time'].get_mean()
            print('FPS: {:.3f}'.format(1 / avg_time))

    if cfg.test.mode == 'geometry':
        if cfg.test.vis.id == '':
            img_path = None
        else:
            img_path = os.path.join(output_dir, '{}-geometry-{}.png'.format(cfg.name, cfg.test.vis.id))
        visualize(
            train_dataset.dense_points,
            model.plane_geo,
            r=cfg.test.vis.r,
            c=cfg.test.vis.c,
            alpha=cfg.test.vis.alpha,
            screenshot_name=img_path
        )

    if cfg.test.mode == 'planes':
        model.bake_planes_alpha(200)
        model = model.cpu()
        plane_num = model.plane_num
        plane_geo = model.plane_geo
        centers = plane_geo.center.detach().numpy()
        rotations = plane_geo.basis().detach().numpy()
        wh = plane_geo.wh.detach().numpy()
        planes_alpha = model.planes_alpha.squeeze().cpu().detach().numpy()
        
        planes = []
        for i in range(plane_num):
            print(f'{i+1}/{plane_num}')
            plane = get_vedo_alpha_plane(
                centers[i],
                rotations[i],
                wh[i],
                planes_alpha[i],
                color=(0.5, 0.5, 0.5)
            )
            planes.append(plane)
        
        vedo.show(*planes)

    if cfg.test.mode == 'content':
        alpha_folder = os.path.join(output_dir, 'alpha')
        os.makedirs(alpha_folder, exist_ok=True)
        
        model.eval()
        model.bake_planes_alpha(200)
        for i in range(cfg.model.plane_num):
            alpha = model.planes_alpha[i]
            img = tensor2Image(alpha)
            path = os.path.join(alpha_folder, "{}-alpha-{:0>5d}.png".format(cfg.name, i))
            img.save(path)


    if cfg.test.mode == 'camera':
        cameras, points = [], []
        color = {
            'train': 'blue',
            'valid': 'red'
        }
        cam_vis = {
            'r': 0.02,
            'height': 0.1,
            'alpha':0.2
        }
        for split, dataset in datasets.items():
            pts, _ = dataset_to_depthpoints(dataset)
            pts = vedo.Points(pts, r=1, c=(0.5, 0.5, 0.5), alpha=1)
            points.append(pts)
            for i in range(len(dataset)):
                data = dataset[i]
                cam = data['camera'] # Pytorch3D camera 
                center = cam.get_camera_center()[0] #(3, )
                rot_mat = cam.R #(1, 3, 3)
                axis = - rot_mat[0, :, -1]
                cam = vedo.Cone(
                    pos=list(center), 
                    axis=list(axis), 
                    r=cam_vis['r'], 
                    height=cam_vis['height'],
                    alpha=cam_vis['alpha'],
                    c=color[split] 
                )
                cameras.append(cam)
            
        vedo.show(cameras, points, axes=1)
        if cfg.test.vis.id != '':
            path = os.path.join(output_dir, '{}-camera-{}.png'.format(cfg.name, cfg.test.vis.id))
            vedo.io.screenshot(path)

    if cfg.test.mode == 'assign':
        folder = os.path.join(output_dir, 'assign')
        os.makedirs(folder, exist_ok=True)

        colors = torch.rand(cfg.model.plane_num, 3).to(device)
        for split, dataset in datasets.items():
            for i in range(len(dataset)):
                data = dataset[i]
                camera = data['camera'].to(device)
                out = model(camera)
                alpha_weight = out['alpha'] #(plane_n, h*w)
                sort_idx = out['sort_idx'] #(plane_n, h*w)

                color_sorted = colors[sort_idx] #(plane_n, h*w, 3)
                color_composite = torch.sum(color_sorted * alpha_weight.unsqueeze(-1), dim=0)
                color_img = color_composite.view(*cfg.data.image_size, 3)
                color_img = tensor2Image(color_img)
                path = os.path.join(folder, '{}-{}-{}-assign.png'.format(
                    cfg.name, split, i
                ))
                color_img.save(path)
    
    if cfg.test.mode == 'error':
        folder = os.path.join(output_dir, 'error')
        os.makedirs(folder, exist_ok=True)

        color_factor = 1 # adjust the ratio between true color and error
        color_high = torch.tensor([1.0, 0.0, 0.0])[None, None]
        color_low  = torch.tensor([0.0, 0.0, 1.0])[None, None]
        for split, dataset in datasets.items():
            for i in range(len(dataset)):
                data = dataset[i]
                _, images = forward_pass(data, model, device, cfg)
                img_gt   = images['color_gt'].cpu()
                img_pred = images['color_pred'].cpu()
                
                error = torch.sum(img_gt**2 - img_pred**2, dim=-1)
                error = (error / error.max()).unsqueeze(-1) #(h, w, 1)
                error_map = error*color_high + (1-error)*color_low
                error_map = error_map * color_factor + img_gt
                error_map = tensor2Image(error_map / error_map.max())
                path = os.path.join(folder, '{}-{:0>5}-error.png'.format(split, i))
                error_map.save(path)

    if cfg.test.mode == 'store':
        import math 
        from mnh.utils_camera import get_transform_matrix, get_camera_k, get_normalized_direction
        output_dir = os.path.join(output_dir, 'store')
        # camera extrinsic & intrinsic

        splits = ['train', 'valid']
        for split in splits:
            data_dir = os.path.join(output_dir, split)
            os.makedirs(data_dir, exist_ok=True)
            dataset = datasets[split]
            R, T = dataset.R, dataset.T 
            cam_n = R.size(0)
            cam_ext = torch.zeros(cam_n, 4, 4)
            cam_ext[:, :3, :3] = R
            cam_ext[:, -1, :3] = T
            cam_ext[:, -1, -1] = 1
            cam_ext = cam_ext.numpy()
            np.save(os.path.join(data_dir, 'cam_ext.npy'), cam_ext)
            cam_int = get_camera_k(dataset[0]['camera']).numpy()
            np.save(os.path.join(data_dir, 'cam_int.npy'), cam_int)
        
            camera = dataset[0]['camera']
            proj_trans = camera.get_projection_transform()
            proj_mat = proj_trans.get_matrix().squeeze().numpy() #(4, 4)
            np.save(os.path.join(data_dir, 'projection_matrix'), proj_mat)
        
        # plane geometry
        print('- Save geometry ...')
        geo_dir = os.path.join(output_dir, 'plane_geometry')
        os.makedirs(geo_dir, exist_ok=True)
        plane_geo = model.plane_geo
        center = to_numpy(plane_geo.center)
        np.save(os.path.join(geo_dir, 'center.npy'), center)
        basis = to_numpy(plane_geo.basis())
        np.save(os.path.join(geo_dir, 'rotation.npy'), basis)
        wh = to_numpy(plane_geo.wh)
        np.save(os.path.join(geo_dir, 'wh.npy'), wh)
        world2plane = get_transform_matrix(plane_geo.basis(), plane_geo.center)
        world2plane = to_numpy(world2plane)
        np.save(os.path.join(geo_dir, 'world2plane.npy'), world2plane)

        planes_verts = []
        plane_num = cfg.model.plane_num 
        for i in range(plane_num):
            c = center[i]
            x, y = basis[i,:,0], basis[i,:,1]
            x_s, y_s = x*(wh[i, 0]/2), y*(wh[i, 1]/2)
            verts = [c+x_s+y_s, c-x_s+y_s, c-x_s-y_s, c+x_s-y_s]
            planes_verts.append(verts)
        planes_verts = np.array(planes_verts)
        np.save(os.path.join(geo_dir, 'planes_vertices.npy'), planes_verts)

        # plane rgba
        print('- Save texture(RGBA) ...')
        cam_ref = datasets['train'][0]['camera'].to(device)
        plane_geo = model.plane_geo
        resolution = cfg.model.bake_res
        plane_num = cfg.model.plane_num

        planes_points = plane_geo.get_planes_points(resolution) #(plane_n, res, res, 3)
        planes_points = planes_points.view(-1, 3)
        sample_num = cfg.model.n_bake_sample
        points_total = (resolution ** 2) * plane_num
        chunk_num = math.ceil(points_total / sample_num)
        planes_rgba = []
        with torch.no_grad():
            for i in range(chunk_num):
                start = i * sample_num
                end = min((i+1)*sample_num, points_total)
                points = planes_points[start:end]
                directions = get_normalized_direction(cam_ref, points)
                rgba = model.radiance_field(points, directions)
                rgba = rgba.detach()
                planes_rgba.append(rgba)
        planes_rgba = torch.cat(planes_rgba, dim=0)
        planes_rgba = planes_rgba.view(plane_num, resolution, resolution, 4)
        
        tex_dir = os.path.join(output_dir, 'texture')
        os.makedirs(tex_dir, exist_ok=True)
        for i in range(plane_num):
            rgba = planes_rgba[i]
            img = tensor2Image(rgba)
            name = os.path.join(tex_dir, '{:0>5}.png'.format(i))
            img.save(name)

    if cfg.test.mode == 'video':
        traj_path = os.path.join(CURRENT_DIR, cfg.video.traj_path)
        _, cameras = load_video_cameras(traj_path)
        
        folder_vdo = os.path.join(output_dir, 'video', cfg.video.name)
        os.makedirs(folder_vdo, exist_ok=True)
        
        model.eval()
        frame_num = len(cameras)
        for i in tqdm(range(frame_num)):
            camera = cameras[i].to(device)
            with torch.no_grad():
                out = model(camera)
            rgb_pred = out['color']
            img = tensor2Image(rgb_pred)
            path = os.path.join(folder_vdo, 'frame_{:0>5}.png'.format(i))
            img.save(path)
            
        img_path = os.path.join(folder_vdo, 'frame_%05d.png')
        vdo_path = os.path.join(folder_vdo, 'video-{}.mp4'.format(cfg.video.name))
        command = 'ffmpeg -r {} -i {} {}'.format(cfg.video.fps, img_path, vdo_path)
        os.system(command)
        
if __name__ == '__main__':
    main()