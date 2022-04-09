import torch 
import torch.nn.functional as F 
from mnh.utils import *
from mnh.model_mnh import Model_rf
from mnh.utils_model import *
from omegaconf import DictConfig

def get_model_from_config(cfg: DictConfig):
    model = Model_rf(
        plane_num=cfg.model.plane_num,
        image_size=cfg.data.image_size,
        # # Radiance field 
        n_harmonic_functions_pos=cfg.model.n_harmonic_functions_pos,
        n_harmonic_functions_dir=cfg.model.n_harmonic_functions_dir,
        n_hidden_neurons_pos=cfg.model.n_hidden_neurons_pos,
        n_hidden_neurons_dir=cfg.model.n_hidden_neurons_dir,
        n_layers=cfg.model.n_layers,
        # train & test
        n_train_sample=cfg.model.n_train_sample,
        n_infer_sample=cfg.model.n_infer_sample,
        anti_aliasing=cfg.model.anti_aliasing,
        premultiply_alpha=cfg.model.premultiply_alpha,
        # accelerate
        n_bake_sample=cfg.model.n_bake_sample,
        bake_res=cfg.model.bake_res, 
        filter_bar=cfg.model.filter_bar,
        white_bg=cfg.data.white_bg
    )
    return model 

def forward_pass(
    data, 
    model,
    device,
    cfg, # config object
    optimizer=None,
    training:bool=False,
    **kwargs,
):
    camera   = data['camera'].to(device)
    color_gt = data['color'].to(device)
    # depth_gt = data['depth'].to(device)
    points_dense = data['points'].to(device)

    timer = Timer(cuda_sync= not training)
    # timer = Timer(cuda_sync=True)
    if training:
        model.train()
        out = model(camera)
        sample_idx = out['sample_idx']
        color_gt = color_gt.view(-1, 3)[sample_idx]
        # depth_gt = depth_gt.view(-1)[sample_idx]
    else:
        with torch.no_grad():
            model.eval()
            out = model(camera)
    time = timer.get_time()
    # timer.print_time('Inference')

    depth_pred = out['depth']
    color_pred = out['color']
    # points_pred = out['points']
    # points = torch.cat([points_dense, points_pred])

    mse_color = F.mse_loss(color_pred, color_gt)
    # mse_depth = F.mse_loss(depth_pred, depth_gt) 
    loss_geo = model.compute_geometry_loss(points_dense)
    loss_point2plane = loss_geo['loss_point2plane']
    loss_area = loss_geo['loss_area']
    # timer.print_time('calculate loss')

    if training: #training
        optimizer.zero_grad()
        loss =  mse_color * cfg.loss_weight.color
        # loss += mse_depth * cfg.loss_weight.depth 
        loss += loss_point2plane * cfg.loss_weight.point2plane
        loss += loss_area * cfg.loss_weight.area
        loss.backward()
        optimizer.step()
    # timer.print_time('backforward')
    psnr = compute_psnr(color_gt, color_pred)
    ssim = compute_ssim(color_gt, color_pred) if not training else 0

    # alpha = out['alpha'].detach().cpu()
    # alpha_dist = torch.histc(alpha, bins=10, min=0, max=1) / alpha.numel()
    # time = timer.get_time()
    stats = {
        'mse_color': mse_color.detach().cpu().item(), 
        # 'mse_depth': mse_depth.detach().cpu().item(),
        'mse_point2plane': loss_point2plane.detach().cpu().item(),
        'loss_area': loss_area.detach().cpu().item(),
        'psnr': psnr,
        'ssim': ssim,
        'FPS': 1/time,
        'time': time,
        'eval_num': out['eval_num'].mean().detach().cpu().item()
    }
    images = {
        # 'depth_gt': depth_gt/depth_gt.max(),
        'depth_pred': depth_pred,#/depth_pred.max(),
        'color_gt': color_gt,
        'color_pred': color_pred 
    }
    return stats, images