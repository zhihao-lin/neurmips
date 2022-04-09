import os
import torch
from matplotlib import pyplot as plt
from PIL import Image
from .utils_camera import *
from .utils import calc_mse
import numpy as np
from .dataset import get_datasets
from .utils import tensor2Image, calc_mse, generate_gif, gaussian_blur
from .model import planes_center_gt, SimpleModel_v2

def plot_loss_gradient():
    data_dir = 'grad_check/'
    data_name = 'test-0'
    out_dir = 'grad_check/0714-gaussian-blur'
    case_name = 'baseline-min'
    out_dir = os.path.join(out_dir, '{}'.format(case_name))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')
    image_size = [400, 400]
    eps = 1e-4
    steps = 50
    # gaussian kernel
    kernel_size = 9
    sigma = 20

    train_data = get_datasets(data_dir, data_name, image_size)[0]
    image, camera, cam_id = train_data[0].values()
    image, camera = image.to(device), camera.to(device)
    # image = gaussian_blur(image, kernel_size, sigma)
    model = SimpleModel_v2(image_size, device, kernel_size, sigma)
    model.to(device)
    
    start, end = -8, 10 
    texture_mse, pos = [], []
    images_pred = []
    grad_back, grad_num = [], []

    for i in np.linspace(start, end, steps):
        planes_center = planes_center_gt.clone()
        planes_center[0, 2] += i
        model.planes_center.data = planes_center.to(device)
        output = model(camera)
        img_pred = output['image_pred']
        tex_mse = calc_mse(image, img_pred)
        images_pred.append(img_pred)
        pos.append(i)
        texture_mse.append(tex_mse.item())

        model.zero_grad()
        tex_mse.backward()
        grad_b = model.planes_center.grad.clone()
        grad_n = numerical_gradients(
            model,
            image,
            camera, 
            eps=eps
        )
        grad_back.append(grad_b)
        grad_num.append(grad_n)
    
    texture_mse = np.array(texture_mse)
    grad_back = torch.stack(grad_back).cpu()
    grad_num  = torch.stack(grad_num).cpu()

    grad_step = (texture_mse[1:] - texture_mse[:-1])*steps/(end-start)
    plt.title('Step gradient')
    plt.plot(pos[:-1], grad_step)
    name = os.path.join(out_dir, 'grad_step.png')
    plt.savefig(name)
    plt.close()

    gif_name = os.path.join(out_dir, 'animation.gif')
    gif_size = [200, 200]
    generate_gif(
        gif_name,
        images_pred, 
        gif_size
    )
    
    plt.plot(pos, texture_mse, c='r', label='texture_mse')
    plt.legend()
    name = os.path.join(out_dir, 'tex_mse.png')
    plt.savefig(name)
    plt.close()

    w, h  = 320, 240
    grad_back_all = Image.new('RGB', (w*5, h*3))
    grad_all = Image.new('RGB', (w*5, h*3))
    for p in range(5):
        for d in range(3):
            label = '[{},{}]'.format(p, d)
            plt.title(label)
            plt.plot(pos, grad_back[:, p,d], label='backprop', c='r')
            plt.legend()
            name = os.path.join(out_dir, 'grad_back_{}.png'.format(label))
            plt.savefig(name)
            img = Image.open(name)
            img = img.resize([w, h])
            grad_back_all.paste(img, [p*w, d*h])

            plt.plot(pos, grad_num[:, p,d], label='numerical', c='b')
            plt.legend()
            name = os.path.join(out_dir, 'grad_{}.png'.format(label))
            plt.savefig(name)
            plt.close()
            img = Image.open(name)
            img = img.resize([w, h])
            grad_all.paste(img, [p*w, d*h])
    
    name = os.path.join(out_dir, 'grad_back_all.png')
    grad_back_all.save(name)
    name = os.path.join(out_dir, 'grad_all.png')
    grad_all.save(name)


def numerical_gradients(
    model,
    image, 
    camera, 
    eps=1e-5
):  
    '''
    Return numerical gradients
    '''
    model.zero_grad()
    planes_center = model.planes_center.data.clone()
    
    plane_n, dim_n = 5, 3
    grad_numerical = torch.zeros(plane_n, dim_n).to(image.device)
    for p in range(plane_n):
        for d in range(dim_n):
            planes_center_l = planes_center.clone()
            planes_center_l[p,d] -= eps
            planes_center_r = planes_center.clone()
            planes_center_r[p,d] += eps

            model.planes_center.data = planes_center_l
            out = model(camera)
            loss_l = calc_mse(image, out['image_pred'])
            model.planes_center.data = planes_center_r
            out = model(camera)
            loss_r = calc_mse(image, out['image_pred'])
            grad = (loss_r-loss_l) / (eps*2)
            grad_numerical[p,d] = grad
    
    return grad_numerical.detach()

if __name__ == '__main__':
    plot_loss_gradient()