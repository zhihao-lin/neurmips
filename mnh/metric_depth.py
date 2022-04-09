import os
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from .plot import save_multi_images
import argparse

data_root = 'output_images/Replica/'
scenes = ['apartment_0', 'apartment_1', 'apartment_2', 'frl_0', 'kitchen', 'room_0', 'room_2']
methods = ['nerf', 'nex', 'KiloNeRF', 'ours', 'ours-distill']

def load_depth(method, scene):
    path = os.path.join(data_root, method, scene, 'depth.npy')
    depth = np.load(path)
    return depth

def load_rgb(method, scene):
    folder_path = os.path.join(data_root, method, scene, 'images')
    rgb = load_folder_images(folder_path)
    return rgb

def is_image_name(name):
    image_names = ['png', 'PNG', 'jpg', 'JPG']
    return (name[-3:] in image_names)

def load_folder_images(folder):
    names = sorted(os.path.join(folder, name) for name in os.listdir(folder) if is_image_name(name))
    images = []
    for i in range(len(names)):
        img = Image.open(names[i])
        img = np.array(img).astype(np.float32)
        img /= 255.0
        images.append(img)
    images = np.stack(images)
    return images

def save_image(path, array, size=None):
    '''
    array: numpy array whose values are in (0, 1)
    '''
    array = (array*255).astype(np.uint8)
    image = Image.fromarray(array)
    if size != None:
        image = image.resize(size)
    image.save(path)

def evaluate_metrics():
    method_names = {
        'nerf': 'NeRF',
        'nex': 'NeX',
        'KiloNeRF': 'KiloNeRF',
        # 'kilonerf-th0': 'KiloNeRF (tau=0.0)',
        # 'kilonerf-th3': 'KiloNeRF (tau=3.0)',
        'ours-distill': 'NeurMips'
    }
    method_error = {}
    for scene in scenes:
        print('\n{}\n{}'.format(scene, '='*25))
        gt = load_depth('GT', scene)
        for method, name in method_names.items():
            if method not in method_error:
                method_error[method] = []
            pred = load_depth(method, scene)
            error = pred - gt
            method_error[method].append(error)
            print_metrics(name, error)
    
    print('\n{}\n{}'.format('Average', '='*25))
    for method in method_error:
        error_map = np.stack(method_error[method], axis=0)
        print_metrics(method_names[method], error_map)

def print_metrics(name, error_map):
    metrics = '{:>10} '.format(name)
    metrics += '| MAE: {:.3f}'.format(np.mean(np.abs(error_map)))
    metrics += '| RMSE: {:.3f}'.format(np.sqrt(np.mean(error_map**2))) 
    metrics += '| Median: {:.3f}'.format(np.median(np.abs(error_map)))
    for thresh in [0.05, 0.10, 0.25, 0.50]:
        metrics += '| Out.({:}): {:.3f}'.format(thresh, np.sum(np.abs(error_map) > thresh) / error_map.size)
    print(metrics)

def plot_cumulative_error():
    errors = np.linspace(0.0, 0.5, 50)
    folder = os.path.join(data_root, 'plots/depth/err-0.0-0.5')
    os.makedirs(folder, exist_ok=True)
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Times New Roman'
    })
    method_names = {
        'nerf': 'NeRF',
        'nex': 'NeX',
        'KiloNeRF': 'KiloNeRF',
        # 'kilonerf-th0': 'KiloNeRF (tau=0.0)',
        # 'kilonerf-th3': 'KiloNeRF (tau=3.0)',
        'ours-distill': 'NeurMips'
    }
    method_error = {}
    for method in method_names:
        method_error[method] = []
    for scene in scenes:
        plt.title(scene)
        plt.ylabel('Percentage (%)')
        plt.xlabel('Absolute Error')
        plt.grid()
        gt = load_depth('GT', scene)
        for method, label in method_names.items():
            pred = load_depth(method, scene)
            err_map = np.abs(gt - pred)
            method_error[method].append(err_map)
            percentage =  calculate_percentage(err_map, errors)
            plt.plot(errors, percentage, label=label)
        plt.legend()
        plt.savefig(os.path.join(folder, '{}.png'.format(scene)))
        plt.close()
        print('plot: {}'.format(scene))
    
    plt.title('All scenes (Replica)')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Absolute error')
    plt.grid()
    for method, label in method_names.items():
        err_maps = np.stack(method_error[method], axis=0)
        percentage =  calculate_percentage(err_maps, errors)
        plt.plot(errors, percentage, label=label)
    plt.legend()
    plt.savefig(os.path.join(folder, 'all.png'))
    plt.close()
    print('plot: All')

def get_percentiles(array, q_list):
    percentiles = []
    for q in q_list:
        percentiles.append(np.percentile(array, q))
    return percentiles

def calculate_percentage(array, values):
    percentage = []
    for v in values:
        p = np.sum(array <= v) / array.size * 100
        percentage.append(p)
    percentage = np.array(percentage)
    return percentage

def plot_error_map(method, normal_factor=7.0):
    dir_method = os.path.join(data_root, method)
    for scene in scenes:
        depth_gt = load_depth('GT', scene)
        depth_pred = load_depth(method, scene)
        error = (depth_gt - depth_pred)**2
        stack = np.concatenate([
            depth_gt, 
            depth_pred, 
            error
        ], axis=2)/normal_factor

        dir_error = os.path.join(dir_method, scene, 'error')
        os.makedirs(dir_error, exist_ok=True)
        for i in range(stack.shape[0]):
            img = stack[i]
            img = img.clip(0, 1)
            img = cm.inferno(img)
            path = os.path.join(dir_error, '{:0>3d}.png'.format(i))
            save_image(path, img, size=(600, 200))

def test():
    depth_1 = load_depth('KiloNeRF', 'apartment_0')
    depth_2 = load_depth('kilonerf-th0', 'apartment_0')
    depth_3 = load_depth('kilonerf-th3', 'apartment_0')

    print(np.sum(np.abs(depth_1 - depth_2)))
    print(np.sum(np.abs(depth_1 - depth_3)))

if __name__ == '__main__':
    # test()
    evaluate_metrics()
    # plot_cumulative_error()