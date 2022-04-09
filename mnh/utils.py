from typing import Tuple, List
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.metrics import structural_similarity as calculate_ssim

def parameter_number(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def compute_psnr(gt: torch.Tensor, pred: torch.Tensor):
#     """
#     Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
#     """
#     mse = F.mse_loss(gt, pred)
#     psnr = 10.0 * torch.log10(gt.max()**2/mse)
#     return psnr

def compute_psnr(gt, pred):
    mse = torch.mean((gt - pred)**2)
    device = gt.device
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(device))
    psnr = psnr.cpu().item()
    return psnr 

def compute_ssim(gt, pred):
    '''image size: (h, w, 3)'''
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    ssim = calculate_ssim(pred, gt, data_range=gt.max() - gt.min(), multichannel=True)
    return ssim

def total_variation(
    images, 
    norm:str='1'
):  
    '''
    Args
        images: (n, d, h, w)
        norm: L1/L2
    '''
    dims = len(images.size())
    dim_h, dim_w = dims - 2, dims - 1
    d_h = images[...,1:,:] - images[...,:-1,:] #(..., h-1, w)
    d_h = torch.cat([d_h, torch.zeros_like(d_h)[...,0,:].unsqueeze(dim_h)], dim=dim_h)
    d_w = images[...,:,1:] - images[...,:,:-1] #(..., h, w-1)
    d_w = torch.cat([d_w, torch.zeros_like(d_w)[...,:,0].unsqueeze(dim_w)], dim=dim_w)
    
    var = 0
    if str(norm) == '1':
        var = torch.mean(torch.abs(d_h) + torch.abs(d_w))
    elif str(norm) == '2':
        var = torch.mean(d_h**2 + d_w**2)
    return var

def tensor2Image(
    tensor, 
    image_size: Tuple[int, int] = None, 
    resample: str = 'nearest'
):
    '''
    Args:
        tensor: shape (h, w, 3) in range (0, 1)
        image_size: (h, w)
        resample: 'nearest' or 'bilinear'
    Return:
        PIL image
    '''
    img = tensor.squeeze().detach().cpu()#.clamp(0.0, 1.0)
    img = (img * 255).numpy().astype(np.uint8)
    img = Image.fromarray(img)
    if image_size != None: 
        image_size = (int(image_size[0]), int(image_size[1]))
        resample_mode = Image.NEAREST if resample == 'nearest' else Image.BILINEAR
        img = img.resize(image_size, resample=resample_mode)
    return img

def output_images(
    output_dir: str,
    images_tensor: dict,
    image_size: Tuple[int, int] = None,
    prefix: str = '',
    postfix: str = ''
):
    for name, image_tensor in images_tensor.items():
        image = tensor2Image(image_tensor, image_size)
        out_name = '{}{}{}.png'.format(prefix, name, postfix)
        out_path = os.path.join(output_dir, out_name)
        image.save(out_path)

def generate_gif(
    gif_name: str,
    images: List[torch.tensor],
    size: Tuple[int, int],
    duration=50
):
    images = [tensor2Image(img, size) for img in images]
    images[0].save(
        gif_name,
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=duration,
        loop=0
    )

def gaussian_kernel(
    kernel_size:int,
    sigma:float
):
    shift = (kernel_size - 1) / 2
    side = torch.arange(kernel_size).float() - shift
    x, y = torch.meshgrid(side, side) #(k, k)
    kernel = torch.exp(-(x**2 + y**2)/ (sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_blur(
    image:torch.tensor,
    kernel_size:int, 
    sigma:float=1.0
):  
    '''
    Input & output:
        Image (h, w, dim=3)
    '''
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel[None, None].to(image.device) #(1, 1, k, k)
    image = image.permute(2, 0, 1).unsqueeze(1) #(3, 1, h, w)
    padding = int((kernel_size - 1) / 2)
    image = F.conv2d(image, kernel, padding=padding) #(3, 1, h, w)
    image = image.squeeze().permute(1, 2, 0) #(h, w, 3)
    return image

def list2txt(list, path):
    '''
    output list values into text file (rows) 
    '''
    with open(path, 'w') as f:
        for i in range(len(list)):
            val = list[i]
            f.write('{}\n'.format(val))

def is_image_file(file_name):
    if file_name.endswith('.png') or file_name.endswith('.jpg'):
        return True
    else:
        return False

def get_image_tensors(folder, channels:int=3):
    names = sorted(os.path.join(folder, name) for name in os.listdir(folder) if is_image_file(name))
    images = []
    for i in range(len(names)):
        img = Image.open(names[i])
        img_array = np.array(img)[...,:channels]
        img_tensor = torch.FloatTensor(img_array)
        img_tensor /= 255.0
        images.append(img_tensor)
    images = torch.stack(images, dim=0)
    return images

def random_sample_points(points, rate:float):
    points_n = points.size(0)
    sample_n = int(points_n * rate)
    sample_idx = torch.randperm(points_n)[:sample_n]
    points = points[sample_idx]
    return points

def to_numpy(tensor):
    array = tensor.detach().to('cpu').numpy()
    return array

class Timer:
    def __init__(self, cuda_sync:bool=False):
        self.cuda_sync = cuda_sync 
        self.reset()
    
    def reset(self):
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.start = time.time()
    
    def get_time(self, reset=True):
        if self.cuda_sync:
            torch.cuda.synchronize()
        now = time.time()
        interval = now - self.start
        if reset:
            self.reset()
        return interval

    def print_time(self, info, reset=True):
        interval = self.get_time(reset)
        print('{:.5f} | {}'.format(interval, info))

def test_gaussian():
    img_path = 'tests/paintings/smile.jpg'
    out_path = 'mnh/gaussian.png'
    img = Image.open(img_path)
    img = torch.FloatTensor(np.array(img))
    img_c = img.clone()
    img = gaussian_blur(img, 5, 2)
    print('diff: {}'.format(torch.sum(img - img_c)))
    img = img.numpy().astype(np.uint8)
    img = Image.fromarray(img)
    img.save(out_path)

def test_tv():
    images = torch.randn(100, 4, 128, 128)
    tv_l1 = total_variation(images, norm='1')
    tv_l2 = total_variation(images, norm='2')
    print('TV-L1: {}'.format(tv_l1))
    print('TV-L2: {}'.format(tv_l2))

if __name__ == '__main__':
    test_tv()