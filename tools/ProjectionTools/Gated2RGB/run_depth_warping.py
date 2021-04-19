from tools.ProjectionTools.Gated2RGB.lib.warp_gatedimage import WarpingClass
from tools.ProjectionTools.Gated2RGB.lib.data_loader import load_vehicle_speed, load_time, load_stearing_ange
from tools.Raw2LUTImages.conversion_lib.process import Rectify_image
from tools.CreateTFRecords.generic_tf_tools.resize import resize
from tools.ProjectionTools.Gated2RGB.lib.image_transformer import disparity2depth_psm
import cv2
import os
import numpy as np
import argparse


def parsArgs():
    parser = argparse.ArgumentParser(description='Gated2RGB projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder', default='./example_data')
    parser.add_argument('--depth_folder', '-d', help='Data folder precise depth', default='psmnet_sweden', choices=['cam_stereo_sgm', 'psmnet_sweden'])
    parser.add_argument('--debug', '-deb', type=bool, help='Save human readable image', default=True)
    parser.add_argument('--suffix', '-s', type=str, help='Define suffix for warped images', default='psm_warped')
    args = parser.parse_args()


    return args

class DepthWarpingWrapper():
    # Class loads Raw Images
    gated_keys = ['gated0_raw', 'gated1_raw', 'gated2_raw']
    image_keys = ['cam_stereo_left'] # Add raw
    history_images = ['cam_stereo_left_raw_history_%d'%i for i in range(-6,5)]

    def __init__(self, source_dir=None, dest_root=None, suffix=None, DEBUG=True, depthfolder='psm_sweden'):
        self.source_dir = source_dir
        self.dest_root = dest_root
        self.suffix = suffix
        self.r = resize('RGB2Gatedv2')
        self.DEBUG = DEBUG
        self.depth_folder = depthfolder
        self.WarpGated = WarpingClass()
        self.WarpGated.InitTransformer(source_dir)
        self.RL = Rectify_image(self.source_dir, 'calib_cam_stereo_left.json')
        self.RR = Rectify_image(self.source_dir, 'calib_cam_stereo_right.json')
        self.RG = Rectify_image(self.source_dir, 'calib_gated_bwv.json', DEBUG=False)



    def read_data_and_process(self, entry_id, vehicle_speed, delay, angle):

        dist_images = {}
        gated_images = {}
        dist_images_shape = {}
        gated_images_shape = {}
        if self.DEBUG == True:
            for folder in self.image_keys:
                file_path = os.path.join(self.source_dir, folder, entry_id + '.tiff')
                img = self.r.crop(self.RL.process_lut(file_path))
                img_height, img_width, _ = img.shape
                dist_images[folder] = img
                dist_images_shape[folder] = ([img_height, img_width, 3])


        for folder in self.gated_keys:
            file_path = os.path.join(self.source_dir, folder, entry_id + '.tiff')
            if self.DEBUG==True:
                img = self.RG.process_rect_lut_gated8(file_path)
            else:
                img = self.RG.process_rect_gated(file_path)
            img_height, img_width = img.shape
            img = img[:,:, np.newaxis]
            img = np.concatenate([img]*3, axis=2)

            if 'psmnet' in self.depth_folder:
                # Take care PSMNet was trained on half the resolution! Therefore, the disparity has to be multiplied by two!!
                # Also the cam_stereo_sgm disparity maps are caclulated on half the resolution
                depth_single = cv2.resize(disparity2depth_psm(2*np.load(os.path.join(self.source_dir, self.depth_folder, entry_id + '.npz'))['arr_0']), (1920, 1024)) #
            else:
                depth_single = cv2.resize(disparity2depth_psm(np.load(os.path.join(self.source_dir, self.depth_folder, entry_id + '.npz'))['arr_0']), (1920, 1024)) #

            img = self.WarpGated.process_image_ego_motion((768, 1280), img, depth_single, vehicle_speed, angle, delay[folder.split('_')[0]], self.RG.PC.K)
            gated_images[folder] = img
            gated_images_shape[folder] = ([img_height, img_width, 1])


        data = {}
        data['image_data'] = dist_images
        data['gated_data'] = gated_images
        data['image_shape'] = dist_images_shape
        data['gated_shape'] = gated_images_shape
        data['name'] = entry_id

        return data

    def save_gated_data(self, data, key):
        if self.dest_root is not None:
            alpha = 0.5
            if self.DEBUG==True:
                for folder in self.gated_keys:
                    overlay1 = cv2.addWeighted(data['image_data']['cam_stereo_left'], alpha,
                                           data['gated_data'][folder], 1 - alpha, 0)
                    path = os.path.join(self.dest_root, folder.split('_')[0] + '_' + self.suffix + '_debug')
                    if not os.path.exists(path):
                       os.makedirs(path)
                    cv2.imwrite(os.path.join(path, key + '.png'), overlay1)
                output = np.max((data['gated_data']['gated0_raw'],data['gated_data']['gated1_raw'],data['gated_data']['gated2_raw']),axis=-1).astype(np.uint8).transpose((1,2,0))
                print(output.shape, data['image_data']['cam_stereo_left'].shape)
                overlay = cv2.addWeighted(data['image_data']['cam_stereo_left'], alpha,
                                           cv2.cvtColor(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR), 1 - alpha, 0)
                return overlay, data['image_data']['cam_stereo_left'], output
            else:
                for folder in self.gated_keys:
                    path = os.path.join(self.dest_root, folder.split('_')[0] + '_' + self.suffix)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    print(data['gated_data'][folder].dtype, np.min(data['gated_data'][folder]), np.max(data['gated_data'][folder]))
                    cv2.imwrite(os.path.join(path, key + '.tiff'), data['gated_data'][folder])
                return None, None, None


if __name__ == '__main__':
    args = parsArgs()
    if args.debug:
        cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)
    T = DepthWarpingWrapper(source_dir=args.root, dest_root=args.root, suffix=args.suffix, DEBUG=args.debug, depthfolder=args.depth_folder)
    T2 = None
    if args.debug:
        T2 = DepthWarpingWrapper(source_dir=args.root, dest_root=args.root, suffix=args.suffix+'_no_correction', DEBUG=args.debug, depthfolder=args.depth_folder)
        cv2.namedWindow('DEBUG', cv2.WINDOW_NORMAL)

    # Read files
    files = os.listdir(os.path.join(args.root, 'cam_stereo_left'))
    print(files)
    for key in files:
        key = key.split('.tiff')[0]
        print(key)
        delta0 = float(load_time('gated0',key)[1] - load_time('rgb', key)[1])/10**9
        delta1 = float(load_time('gated1',key)[1] - load_time('rgb', key)[1])/10**9
        delta2 = float(load_time('gated2',key)[1] - load_time('rgb', key)[1])/10**9
        delays = {
            'gated0': delta0,
            'gated1': delta1,
            'gated2': delta2
        }
        speed = load_vehicle_speed(args.root, key)/3.6 # conversion from km/h to m/s.
        angle = load_stearing_ange(args.root, key)/520*30 # conversion from steering angle to heading. Assumption of 3 steering wheel rotations from end to end and a maximum heading of 30°.


        data = T.read_data_and_process(key, speed, delays, angle)
        img1, rgb1, output1 = T.save_gated_data(data, key)


        if args.debug == True:
            delays2 = {
                'gated0': 0,
                'gated1': 0,
                'gated2': 0
            }
            data2 = T2.read_data_and_process(key, speed, delays2, 0)
            img2, rgb2, output2 = T2.save_gated_data(data2, key)
            cv2.imshow('DEBUG', np.hstack((img1, img2)))
            print(speed, angle, delays['gated0'])
            cv2.waitKey()
            cv2.imshow('DEBUG', np.hstack((output1, output2)))
            cv2.waitKey()
            cv2.imshow('DEBUG', np.vstack((np.hstack((rgb1, output1)),np.hstack((img1, img2)))))
            cv2.waitKey()
