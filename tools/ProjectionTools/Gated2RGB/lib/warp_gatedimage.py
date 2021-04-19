import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tools.CreateTFRecords.generic_tf_tools.resize import resize
from tools.ProjectionTools.Gated2RGB.lib.image_transformer import ImageTransformer
# You need to import the resize class from the first Calib read and projection tools.
import json

# Projections are created in the gated frame


def pad_gated_to_psm(img_in):
    img_out = np.lib.pad(img_in, ((0, 0), (216, 296), (0, 0)), mode='constant', constant_values=0)
    return img_out




def process_points(DEBUG=False):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coresponding_points.txt'), 'r') as f:
        points = json.load(f)

    X, Y = points['pos1'], points['pos2']
    if DEBUG:
        for x, y in zip(X, Y):
            print(x, y)
        print(len(X), len(Y))

    return X,Y



class WarpingClass():

    def __init__(self):
        self.r = resize('RGB2Gatedv2')
        self.X, self.Y = process_points()
        dst_pts = np.asarray([[x, y] for x, y in self.X]).astype(np.float32).reshape(-1, 1, 2)
        src_pts = np.asarray([[x, y - 768] for x, y in self.Y]).astype(np.float32).reshape(-1, 1, 2)
        print(dst_pts.shape, src_pts.shape)

        self.M, self.mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10)
        self.matchesMask = self.mask.ravel().tolist()

    def InitTransformer(self, root):
        tf_file = "calib_tf_tree_full.json"
        cam_file_gated = "calib_gated_bwv.json"
        cam_file_rgb_left = "calib_cam_stereo_left.json"
        cam_file_rgb_right = "calib_cam_stereo_right.json"
        rgb_left_frame = "cam_stereo_left_optical"
        rgb_right_frame = "cam_stereo_right_optical"
        gated_frame = "bwv_cam_optical"
        self.it = ImageTransformer(source_frame=gated_frame, target_frame=rgb_left_frame, target_cam_file=os.path.join(root, cam_file_rgb_left), source_cam_file=os.path.join(root, cam_file_gated), tf_file=os.path.join(root, tf_file), source_cam_file_stereo=None)#os.path.join(root_dir, cam_file_rgb_right))


    def process_images(self, image, gated_image, DEBUG=False):
        img1 = self.r.crop(image)  # queryImage
        img0 = np.zeros((img1.shape[0], img1.shape[1] + 30, 3))
        img0[:img1.shape[0], :img1.shape[1], :] = img1
        img1 = img0
        img2 = cv2.resize(gated_image, (img1.shape[1], img1.shape[0]))  # trainImage
        if DEBUG:
            h, w, c = img1.shape

            # Example how to transpose a 2d Box from camera frame to gated frame.
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            print(pts)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            plt.imshow(img2, 'gray'), plt.title('Projected Box'), plt.show()

        # warp image from gated to rgb
        img22 = cv2.warpPerspective(img2.copy(), self.M, (img1.shape[1], img1.shape[0]))
        return img1[:,:-30], img22[:,:-30]

    def process_image(self, image_shape, gated_image):
        # Uses constant image homography to map images
        img2 = cv2.resize(gated_image, (image_shape[1]+30, image_shape[0]))  # trainImage
        # warp image from gated to rgb
        img22 = cv2.warpPerspective(img2.copy(), self.M, (image_shape[1]+30, image_shape[0]))
        return img22[:,:-30]

    def process_image_ego_motion(self, image_shape, gated_image, depth, vehicle_speed, angle, delta_time, CameraMatrix):
        # Uses ego motion corrected depth dependend image mapping -> recommended

        img22 = self.r.crop(self.it.transform_with_target_depth(gated_image, None, depth, vehicle_speed=vehicle_speed, delay=delta_time, angle=angle))
        return img22
