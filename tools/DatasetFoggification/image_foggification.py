import os
import gc
import cv2
import scipy
import scipy.io
import scipy.ndimage
import numpy as np
import argparse
import socket

from tqdm import tqdm

import multiprocessing

WORKERS = multiprocessing.cpu_count() - 1 or 1

DEBUG = False


def parsArgs():
    parser = argparse.ArgumentParser(description='Image foggification')
    parser.add_argument('-r', '--root', help='root folder of dataset', default='/Users/Hahner/trace/datasets/nuScenes')
    parser.add_argument('-d', '--depth_folder', help='Data folder precise Depth', default='depth/TRI/png')
    parser.add_argument('-i', '--image_folder', help='Data folder Images', default='samples')
    parser.add_argument('-b', '--beta', type=float, help='fogdensity parameter', default=0.16)
    parser.add_argument('-p', '--parallel', type=bool, help='Parallel execution', default=False)
    parser.add_argument('-s', '--seed', help='random seed', default=None)
    parser.add_argument('-k', '--k', type=int, help='k for atmospheric light estimation', default=210)
    parser.add_argument('-t', '--inverse_transmittance', help='specify if the transmittance has to be inverted',
                        default=False)

    args = parser.parse_args()

    args.destination_folder = 'hazing/image_beta%.5f' % args.beta

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
    mean_p = boxfilter(p, r) / N[:, :, np.newaxis]

    mean_Ip = boxfilter(I * p, r) / N[:, :, np.newaxis]

    cov_Ip = mean_Ip - mean_I * mean_p

    # var_I = boxfilter(np.matmul(I,I),r) / N[:,:,np.newaxis] - np.matmul(mean_I, mean_I)
    var_I_rg = boxfilter(I[:, :, 0] * I[:, :, 1], r) / N - mean_I[:, :, 0] * mean_I[:, :, 1]
    var_I_rb = boxfilter(I[:, :, 0] * I[:, :, 2], r) / N - mean_I[:, :, 0] * mean_I[:, :, 2]
    var_I_gb = boxfilter(I[:, :, 1] * I[:, :, 2], r) / N - mean_I[:, :, 1] * mean_I[:, :, 2]

    var_I = boxfilter(I * I, r) / N[:, :, np.newaxis] - mean_I * mean_I

    var_I_rr = var_I[:, :, 0]
    var_I_gg = var_I[:, :, 1]
    var_I_bb = var_I[:, :, 2]

    Sigma = np.array([[var_I_rr, var_I_rg, var_I_rb],
                      [var_I_rg, var_I_gg, var_I_gb],
                      [var_I_rb, var_I_gb, var_I_bb]])

    eps = eps * np.eye(3)

    Sigma = Sigma + eps[:, :, np.newaxis, np.newaxis]  # + 1e-2
    Sigma = np.moveaxis(np.moveaxis(Sigma, 2, 0), 3, 1)
    Sigma_inv = np.linalg.inv(Sigma)

    a = np.squeeze(np.matmul(cov_Ip[:, :, np.newaxis, :], Sigma_inv))

    b = (mean_p - (a[:, :, 0] * mean_I[:, :, 0])[:, :, np.newaxis]
         - (a[:, :, 1] * mean_I[:, :, 1])[:, :, np.newaxis]
         - (a[:, :, 2] * mean_I[:, :, 2])[:, :, np.newaxis])

    q = ((boxfilter(a[:, :, 0], r) * I[:, :, 0])[:, :, np.newaxis]
         + (boxfilter(a[:, :, 1], r) * I[:, :, 1])[:, :, np.newaxis]
         + (boxfilter(a[:, :, 2], r) * I[:, :, 2])[:, :, np.newaxis]
         + boxfilter(b, r)) / N[:, :, np.newaxis]

    return q


def get_transmittance(depth, beta):
    return np.e ** (-beta * depth.astype(np.float32))


def grey_scale(pixel_bgr):
    grey_scale_ = 0.299 * pixel_bgr[..., 2] + 0.587 * \
                  pixel_bgr[..., 1] + 0.114 * pixel_bgr[..., 0]
    return grey_scale_[..., np.newaxis]


def median_pixel(image):
    pixel_vector = image.reshape(
        (image.shape[0] * image.shape[1], image.shape[2]))
    return np.median(pixel_vector, 0)


def dark_channel(image, kernel_size):
    gray = np.min(image, 2)

    dc = scipy.ndimage.minimum_filter(gray, kernel_size)

    if DEBUG:
        cv2.imshow('Input', image)
        cv2.imshow('Minimum', gray)
        cv2.imshow('Dark Channel', dc)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return dc


def topk(array_1d, k):
    if k >= array_1d.shape[0]:
        return array_1d

    return array_1d[np.argpartition(array_1d, -k)[-k:]]


def topk_2d(array_2d, k):
    result = []

    for each in range(array_2d.shape[1]):
        result.append(topk(array_2d[..., each], k))

    return np.array(result)


def brightes_pixel(image):
    y = grey_scale(image)

    index = np.transpose(np.where(y == np.max(y)))[0]
    pixel = image[index[0], index[1], :].copy()

    return pixel


def atmospheric_light(image, k):
    dark = dark_channel(image, 10)
    dark_median = np.median(topk_2d(dark, k), 1)
    dark_filter = dark_median == dark

    return np.max(np.max(image[dark_filter], 1), 0)


def fogify(image, depth, beta, atmospheric_light_estimation, inverse_transmittance):
    transmittance = get_transmittance(depth, beta)
    transmittance = np.clip((transmittance * 255), 0, 255).astype(np.uint8)
    transmittance = cv2.bilateralFilter(transmittance, 9, 75, 75)
    transmittance = transmittance.astype(np.float32) / 255
    transmittance = np.clip(transmittance, 0, 1)

    if inverse_transmittance:
        transmittance = 1 - transmittance

    # image = np.clip(image, 0, 255)

    transmittance = guidedfilter3(image.astype(np.float32) / 255, transmittance, 20, 1e-3)

    fog_image = np.clip(image * transmittance +
                        atmospheric_light_estimation * (1 - transmittance), 0, 255).astype(np.uint8)

    return fog_image


class Foggify:

    def __init__(self, args):
        self.args = args

    def fogify_path_tuple(self, image_file, dst_folder, inverse_transmittance, extra_folder=None):

        if extra_folder:
            image_path = os.path.join(self.args.root, self.args.image_folder, extra_folder, image_file)
            depth_path = os.path.join(self.args.root, self.args.depth_folder, extra_folder, image_file)
            depth_path = depth_path.replace('jpg', 'png')
        else:
            image_path = os.path.join(self.args.root, self.args.image_folder, image_file)
            depth_path = os.path.join(self.args.root, self.args.depth_folder, image_file)

        image = cv2.imread(image_path)
        depth = cv2.imread(depth_path)

        # resize depth image to input image
        depth = cv2.resize(depth, image.shape[:2][::-1])

        file_name = image_path.split('/')[-1]

        atmospheric_light_estimation = atmospheric_light(image, self.args.k)  # TODO: almost always leads to 255

        fog_image = image

        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)

        output_file = os.path.join(dst_folder, file_name)

        fog_image = fogify(fog_image, depth, self.args.beta, atmospheric_light_estimation, inverse_transmittance)

        if DEBUG:
            print(f'Atmospheric Light Estimation: {atmospheric_light_estimation}')
            print(output_file)

            cv2.imshow('Foggy Image', fog_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(output_file, fog_image)
        gc.collect()


def main():
    args = parsArgs()
    args.seed = 0

    hostname = socket.gethostname()

    if 'MacBook' in hostname or '.ee.ethz.ch' in hostname:
        args.root = '/Users/Hahner/trace/datasets/nuScenes'
    else:  # assume CVL host
        args.root = '/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/nuScenes'

    cam_folders = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for folder in cam_folders:

        src_folder = os.path.join(args.root, args.image_folder, folder)
        print(f'Source folder: {src_folder}')

        betas = [0.005, 0.01, 0.02, 0.03, 0.06]
        betas.reverse()

        for b in betas:

            args.beta = b

            dst_folder = f'{src_folder}_DENSE_TRI_beta_{args.beta:.3f}'

            images = sorted(os.listdir(src_folder))

            fogClass = Foggify(args)

            print(f'Starting to generate images using beta={b} in {dst_folder}')

            if args.parallel:

                print("parallel execution with {} workers".format(WORKERS))
                pool = multiprocessing.Pool(processes=WORKERS)
                pool.map(fogClass.fogify_path_tuple, images)
                pool.close()
                pool.join()

            else:

                for image in images if DEBUG else tqdm(images):
                    fogClass.fogify_path_tuple(image, dst_folder, args.inverse_transmittance, extra_folder=folder)


if __name__ == "__main__":
    main()
