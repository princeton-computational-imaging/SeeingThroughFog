import glob
import os
import scipy
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import sys
import gc
import time
import argparse

import multiprocessing
WORKERS = multiprocessing.cpu_count()-1 or 1


def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar Fog Simulation Filename')
    parser.add_argument('--root', '-r', help='Enter the root folder', default='./example_data/')
    parser.add_argument('--depth_folder', '-d', help='Data folder precise Depth', default='DepthData')
    parser.add_argument('--image_folder', '-i', help='Data folder Images', default='ImageData')
    parser.add_argument('--beta', '-b', type=float, help='Enter the fog density beta', default=0.16)
    parser.add_argument('--parallel', '-p', type=bool, help='Parallel execution', default=False)
    args = parser.parse_args()
    args.destination_folder = 'hazing/image_beta%.5f'%args.beta
    global hazed

    return args






def boxfilter(img, r):
    # r = 2 * r + 1
    return cv2.boxFilter(img, -1, (r, r))



def guidedfilter3(I, p, r, eps):
    """
    Simple matlab code https://github.com/clarkzjw/GuidedFilter/blob/master/MATLAB/guidedfilter_color.m converted to numpy and optimized
    A more extensive faster Guided Filter used for the Experiments can be found in https://github.com/tody411/GuidedFilter
    """
    [hei, wid] = p.shape[0], p.shape[1]
    N = boxfilter(np.ones([hei, wid]), r)

    mean_I = boxfilter(I, r) / N[:, :, np.newaxis]
    mean_p = boxfilter(p, r) / N

    mean_Ip = boxfilter(I * p[:, :, np.newaxis], r) / N[:, :, np.newaxis]

    cov_Ip = mean_Ip - mean_I * mean_p[:, :, np.newaxis]

    # var_I = boxfilter(np.matmul(I,I),r) / N[:,:,np.newaxis] - np.matmul(mean_I, mean_I)
    var_I_rg = boxfilter(I[:, :, 0] * I[:, :, 1], r) / N - mean_I[:, :, 0] * mean_I[:, :, 1]
    var_I_rb = boxfilter(I[:, :, 0] * I[:, :, 2], r) / N - mean_I[:, :, 0] * mean_I[:, :, 2]
    var_I_gb = boxfilter(I[:, :, 1] * I[:, :, 2], r) / N - mean_I[:, :, 1] * mean_I[:, :, 2]
    var_I = boxfilter(I * I, r) / N[:, :, np.newaxis] - mean_I * mean_I
    var_I_rr = var_I[:, :, 0]
    var_I_gg = var_I[:, :, 1]
    var_I_bb = var_I[:, :, 2]

    a = np.zeros([hei, wid, 3])
    Sigma = np.array([[var_I_rr, var_I_rg, var_I_rb],
                      [var_I_rg, var_I_gg, var_I_gb],
                      [var_I_rb, var_I_gb, var_I_bb]])
    eps = eps * np.eye(3)
    Sigma = Sigma + eps[:, :, np.newaxis, np.newaxis]  # + 1e-2
    Sigma = np.moveaxis(np.moveaxis(Sigma, 2, 0), 3, 1)
    Sigma_inv = np.linalg.inv(Sigma)
    a = np.squeeze(np.matmul(cov_Ip[:, :, np.newaxis, :], Sigma_inv))
    b = mean_p - a[:, :, 0] * mean_I[:, :, 0] - a[:, :, 1] * \
        mean_I[:, :, 1] - a[:, :, 2] * mean_I[:, :, 2]

    q = (boxfilter(a[:, :, 0], r) * I[:, :, 0]
         + boxfilter(a[:, :, 1], r) * I[:, :, 1]
         + boxfilter(a[:, :, 2], r) * I[:, :, 2]
         + boxfilter(b, r)) / N
    return q


def transmittance(depth, beta):
    return np.e**(-beta * depth.astype(np.float32))

def grey_scale(pixel_bgr):
    grey_scale_ = 0.299 * pixel_bgr[..., 2] + 0.587 * \
        pixel_bgr[..., 1] + 0.114 * pixel_bgr[..., 0]
    return grey_scale_[..., np.newaxis]


def median_pixel(image):
    pixel_vector = image.reshape(
        (image.shape[0] * image.shape[1], image.shape[2]))
    return np.median(pixel_vector, 0)


def topk(array_1d, k):
    if k >= array_1d.shape[0]:
        return array_1d
    return array_1d[np.argpartition(array_1d, -k)[-k:]]

def dark_channel(image, kernel_size):
    print(image.shape)
    image= np.min(image,2)
    
    dc= scipy.ndimage.minimum_filter(image, kernel_size)
    return dc

def topk_2d(array_2d, k):
    result = []
    for each in range(array_2d.shape[1]):
        result.append(topk(array_2d[..., each], k))

    return np.array(result)

def brightes_pixel(image,path):
    y = grey_scale(image)
    index = np.transpose(np.where(y==np.max(y)))[0]
    pixel = image[index[0],index[1],:].copy()
    return pixel

def atmospheric_light(image):
    dark = dark_channel(image, 10)
    dark_median = np.median(topk_2d(dark, 210),1)
    dark_filter = dark_median == dark
    return np.max(np.max(image[dark_filter],1),0)


def fogify(image, depth, beta, atmospheric_light_):
    get_rect_left = np.where((np.not_equal(image[:, :, 0], 0) & np.not_equal(image[:, :, 1], 0) & np.not_equal(image[:, :, 2], 0)))
    fog_image = image.copy()
    transmittance_ = transmittance(depth, beta)
    transmittance_ = np.clip((transmittance_ * 255), 0, 255).astype(np.uint8)
    transmittance_ = cv2.bilateralFilter(transmittance_, 9, 75, 75)
    transmittance_ = transmittance_.astype(np.float32) / 255
    transmittance_ = np.clip(transmittance_, 0, 1)
    image = np.clip(image, 0,255)
    transmittance_ = guidedfilter3(image.astype(np.float32)/255, transmittance_, 20, 1e-3)
    transmittance_ = transmittance_[:, :, np.newaxis]
    fog_image[get_rect_left] = np.clip(image[get_rect_left] * transmittance_[get_rect_left] + atmospheric_light_ *
                        (1 - transmittance_[get_rect_left]), 0, 255).astype(np.uint8)
    return fog_image

name = 'image_lab_all'
cv2.namedWindow(name,cv2.WINDOW_NORMAL)
def load_image(image_path):
    return cv2.imread(image_path)

class Foggify:

    def __init__(self,args):
        self.args = args

    def fogify_path_tuple(self, image_file):
        image_path, depth_path = os.path.join(self.args.root,self.args.image_folder,image_file), \
                                 os.path.join(self.args.root,self.args.depth_folder,image_file.replace('.png','.mat'))

        image, depth = load_image(image_path), scipy.io.loadmat(depth_path)["depth_map"]
        file_name = image_path.split('/')[-1]
        atmospheric_light_ = atmospheric_light(image)
        fog_image = image
        output_file = os.path.join(self.args.root, self.args.destination_folder)
        if not os.path.isdir(output_file):
            os.makedirs(output_file)
        output_file = os.path.join(output_file, file_name)
        print(output_file)
        fog_image = fogify(fog_image, depth, self.args.beta, atmospheric_light_)

        cv2.imwrite(output_file, fog_image)
        gc.collect()
    



def main():
    args = parsArgs()
    images = os.listdir(os.path.join(args.root,args.image_folder))

    fogClass = Foggify(args)
    if args.parallel:
        print("parallel execution with {} workers".format(WORKERS))
        pool= multiprocessing.Pool(processes=WORKERS)
        pool.map(fogClass.fogify_path_tuple, images)
        pool.close()
        pool.join()
    else:
        for each in images:
            fogClass.fogify_path_tuple(each)


if __name__ == "__main__":
    main()
