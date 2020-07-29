import numpy as np
import os
import argparse
from conversion_lib.plot_utils import PlotLut
from conversion_lib.process import Rectify_image, conversion_params
import cv2

def parsArgs():
    parser = argparse.ArgumentParser(description='RawData Converter')
    parser.add_argument('--root', '-r', help='Enter the root folder', default='./example_data/')
    parser.add_argument('--cam_file', '-c', help='Enter the root folder', default='calib_cam_stereo_left.json')
    parser.add_argument('--image_folder', '-i', help='Data folder Images', default='cam_stereo_left')
    parser.add_argument('--meta_folder', '-m', help='Enter the fog density beta', default='labeltool_labels')
    parser.add_argument('--dest_folder', '-d', help='Enter the fog density beta', default='cam_stereo_left_lut')
    parser.add_argument('--DEBUG', '-D', help='Enter the fog density beta', default=False)
    args = parser.parse_args()
    global hazed

    return args

if __name__ == '__main__':

    args = parsArgs()
    if args.DEBUG == True:
        p = PlotLut()
        p.add_plot(conversion_params["decomp_kneepoints"])
        p.show_plot()
    RI = Rectify_image(args.root, args.cam_file)
    if not os.path.exists(os.path.join(args.root, args.dest_folder)):
        os.makedirs(os.path.join(args.root, args.dest_folder))

    for sample in os.listdir(os.path.join(args.root, args.image_folder)):
        image_lut = RI.process_lut(os.path.join(args.root, args.image_folder, sample), os.path.join(args.root, args.meta_folder, sample.replace('.tiff', '.json')))


        cv2.imwrite(os.path.join(args.root, args.dest_folder, sample.replace('.tiff', '.png')), image_lut)

