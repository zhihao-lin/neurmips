import os
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra
import copy
import vedo
from mnh.dataset_replica import ReplicaDataset
from mnh.dataset_tat import TanksAndTemplesDataset
from mnh.utils_vedo import visualize_geometry
from mnh.stats import StatsLogger
from mnh.utils import *
from mnh.utils_model import freeze_model
import teacher_forward
from experts_forward import *

CURRENT_DIR = os.path.realpath('.')
CONFIG_DIR = os.path.join(CURRENT_DIR, 'configs')
TEST_CONFIG = 'test'
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')


@hydra.main(config_path=CONFIG_DIR, config_name=TEST_CONFIG)
def main(cfg: DictConfig):
    # Set random seed for reproduction
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
        train_dataset = ReplicaDataset(folder=train_path, read_points=True, batch_points=cfg.data.batch_points)
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

    teacher = teacher_forward.get_model_from_config(cfg)
    teacher.to(device)
    model = get_model_from_config(cfg)
    model.to(device)

    # load checkpoints
    stats_logger = None
    checkpoint_teacher = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.teacher)
    pretrained_teacher = os.path.isfile(checkpoint_teacher)
    if pretrained_teacher: 
        print('Load teacher from checkpoint: {}'.format(checkpoint_teacher))
        loaded_data = torch.load(checkpoint_teacher, map_location=device)
        teacher.load_state_dict(loaded_data['model'])
    else:
        print('WARNING: no pretrained weight for teacher network')
    
    checkpoint_experts = os.path.join(CHECKPOINT_DIR, cfg.checkpoint.experts)
    if cfg.train.resume and os.path.isfile(checkpoint_experts):
        print('Resume training from checkpoint: {}'.format(checkpoint_experts))
        loaded_data = torch.load(checkpoint_experts, map_location=device)
        model.load_state_dict(loaded_data['model'])
    else:
        if pretrained_teacher:
            print('[Init] Copy plane geometry from teacher ...')
            model.plane_geo = copy.deepcopy(teacher.plane_geo)
        else:
            print('[Init] Initialize plane geometry')
            points = train_dataset.dense_points.to(device)
            model.plane_geo.initialize(
                points, 
                lrf_neighbors=cfg.model.init.lrf_neighbors,
                wh=cfg.model.init.wh,
            )
            del points

    if cfg.model.accelerate.bake == True:
        model.bake_planes_alpha()
    output_dir = os.path.join(CURRENT_DIR, 'output_images', cfg.name, 'experts')
    os.makedirs(output_dir, exist_ok=True)
    
    print('Test [{}] ...'.format(cfg.test.mode))
    if cfg.test.mode == 'test_model':
        print('- Parameter number: {}'.format(parameter_number(model)))
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.optimizer.lr
        )
        train_data = train_dataset[0]
        
        stats = learn_from_teacher(train_data, model, teacher, device, cfg, optimizer)
        print(stats)
        print('- Learn from teacher')
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
        batch_size = 10000
        net = model.plane_radiance_field[0]
        calculate_complexity(net, batch_size=batch_size)

    if cfg.test.mode == 'evaluate':
        stats_logger= StatsLogger()
        model.eval()
        for split, dataset in datasets.items():
            for i in range(len(dataset)):
                data = dataset[i]
                stats, _ = forward_pass(data, model, device, cfg)
                stats_logger.update(split, stats)
            stats_logger.print_info(split)

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
    
    if cfg.test.mode == 'alpha_map':
        model.bake_planes_alpha()
        dir_alpha = os.path.join(output_dir, 'alpha_maps')
        os.makedirs(dir_alpha, exist_ok=True)
        planes_alpha = model.planes_alpha.squeeze()
        for i in range(model.n_plane):
            a = planes_alpha[i]
            a = tensor2Image(a)
            a.save(os.path.join(dir_alpha, '{:0>4d}.png').format(i))

    if cfg.test.mode == 'geometry':
        if cfg.test.vis.id == '':
            img_path = None
        else:
            img_path = os.path.join(output_dir, '{}-geometry-{}.png'.format(cfg.name, cfg.test.vis.id))
        train_points = dataset_to_depthpoints(train_dataset)
        visualize_geometry(
            train_points,
            model.plane_geo,
            r=cfg.test.vis.r,
            c=cfg.test.vis.c,
            alpha=cfg.test.vis.alpha,
            screenshot_name=img_path
        )
        
                
if __name__ == '__main__':
    main()