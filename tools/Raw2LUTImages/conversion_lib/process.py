from conversion_lib.pinhole_camera_model import PinholeCameraModel
from conversion_lib.basic_utils import read_image_intrinsic, read_tiff_image, save_tiff_image, check_image, read_meta_file, parse_day_night, apply_clahe_8bit
import conversion_lib.decompand as decompand
import numpy as np
import cv2


def fb(x, bitdepth=16):
    return int(x*2**bitdepth)

def gamma_custom(exponent, num=100):
    values = np.linspace(0.0051, 0.999, num)

    lut = [[0,0], [fb(0.0025), fb(0.1)], [fb(0.005), fb(0.25)]]
    y = 0.25
    x = 0.005
    alpha = (y-1)/(x**exponent-1.0)
    beta = 1-alpha
    for i in values:
        lut.append([fb(i), fb(alpha*i**exponent+beta)])
    return lut

conversion_params = {
    "decomp_kneepoints": [[1023, 1023], [2559, 4095], [3455, 32767], [3967, 65535]],
    "comp_kneepoints": [[1023, 1023], [4095,2559], [32767,3455], [65535,3967]],
    "lut_kneepoints": [[512, 30720], [2048, 53760]],
    "lut_kneepoints_daytime": [[fb(0.005), fb(0.05)], [fb(0.01), fb(0.2)], [fb(0.03), fb(0.35)], [fb(0.05), fb(0.4)], [fb(0.1), fb(0.5)], [fb(0.2), fb(0.7)], [fb(0.3), fb(0.8)], [fb(0.4), fb(0.9)], [fb(0.5), fb(0.98)]],
    "lut_kneepoints_nighttime": [[fb(0.0025), fb(0.1)], [fb(0.005), fb(0.25)], [fb(0.01), fb(0.4)], [fb(0.1), fb(0.8)],[fb(0.2), fb(0.9)], [fb(0.3), fb(0.98)]],
    "lut_kneepoints_gated": [[fb(0.0025,bitdepth=10), fb(0.1,bitdepth=10)], [fb(0.005,bitdepth=10), fb(0.25,bitdepth=10)], [fb(0.01,bitdepth=10), fb(0.3,bitdepth=10)], [fb(0.1,bitdepth=10), fb(0.4,bitdepth=10)],[fb(0.2,bitdepth=10), fb(0.5,bitdepth=10)], [fb(0.3,bitdepth=10), fb(0.6,bitdepth=10)]],
    "lut_collorwise": {
        "r": gamma_custom(0.2),
        "g": gamma_custom(0.2),
        "b": gamma_custom(0.2),
    }


}


def create_lut_from_kneepoints(kneepoints, bit_depth=16, start_point=None,DEBUG=False):
    lut_kneepoints = kneepoints[:]
    if start_point is None:
        start_point = [0, 0]
    end_point = [2**bit_depth, 2**bit_depth]
    lut_kneepoints.append(end_point)
    lut = np.zeros((2**bit_depth,), dtype=np.uint16)
    counter = 0
    for idx, kneepoint in enumerate(lut_kneepoints):
        if counter == 0:
            first_point = start_point
            second_point = kneepoint
        else:
            first_point = lut_kneepoints[idx-1]
            second_point = lut_kneepoints[idx]
        counter += 1
        if DEBUG:
            print(first_point)
            print(second_point)
        m = (second_point[1] - first_point[1])/(float(second_point[0] - first_point[0]))
        c = first_point[1] - m*first_point[0]

        lut[first_point[0]:second_point[0]] = np.floor(m*np.arange(first_point[0],second_point[0]) + c).astype(np.uint16)

    return lut

def apply_clahe_8bit(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:,:,0])
    clahe2 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    image[:, :, 1] = clahe2.apply(image[:,:,1])
    image[:, :, 2] = clahe2.apply(image[:,:,2])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return image

class Rectify_image():

    def __init__(self, root, cam_file, DEBUG=False):
        self.PC = PinholeCameraModel()
        self.data_camera = read_image_intrinsic(root, cam_file)
        self.PC.fromJsonDict(self.data_camera)
        decompand_lut_kneepoints = decompand.loadKneepoints(conversion_params['decomp_kneepoints'])
        compand_lut_kneepoints = decompand.loadKneepoints(conversion_params['comp_kneepoints'])
        self.decompand_lut = decompand.create_decompand_lut(decompand_lut_kneepoints)
        self.compand_lut = decompand.create_decompand_lut(compand_lut_kneepoints)
        self.daytime_lut = create_lut_from_kneepoints(conversion_params["lut_kneepoints_daytime"])
        self.nighttime_lut = create_lut_from_kneepoints(conversion_params["lut_kneepoints_nighttime"])
        self.gated_lut = create_lut_from_kneepoints(conversion_params["lut_kneepoints_gated"], bit_depth=10)
        self.DEBUG = DEBUG

    def process_lut(self, image_path, meta_path=None):
        """
        Takes a raw data image and converts it to a 8 bit RGB image with for visual inspection.
        Images are applied with an hand crafted image enhancement process including a gamma correction and
        contrast enhancement. This approach reimplements the process for the images in cam_stereo_left_lut.
        """
        image_raw = read_tiff_image(image_path)
        if meta_path is not None:
            meta = read_meta_file(meta_path)
        else:
            meta = None
        if self.DEBUG:
            check_image(image_raw)
        image = self.decompand_lut[image_raw]

        if parse_day_night(meta):
            image_lut = self.daytime_lut[image]
        else:
            image_lut = self.nighttime_lut[image]
        image_bayer = cv2.cvtColor(image_lut, cv2.COLOR_BAYER_GB2BGR)

        image_bit = np.right_shift(image_bayer, 8).astype(np.uint8)
        image_bit = apply_clahe_8bit(image_bit)
        self.PC.rectifyImage(image_bit, image_bit)
        return image_bit

    def process_rect8(self, image_path):
        """
        Takes a raw data image and converts it to a 8 bit RGB image with for visual inspection.
        Images are only rectified and bitshifted.
        """
        image_raw = read_tiff_image(image_path)
        if self.DEBUG:
            check_image(image_raw)
        image_bayer = cv2.cvtColor(image_raw, cv2.COLOR_BAYER_GB2BGR)
        self.PC.rectifyImage(image_bayer, image_bayer)
        image_bit = np.right_shift(image_bayer, 4).astype(np.uint8)

        return image_bit

    def process_rect(self, image_path):
        """
        Takes a raw data image and converts it to a rectified 12 bit RGB image
        """
        image_raw = read_tiff_image(image_path)
        if self.DEBUG:
            check_image(image_raw)
        image_bayer = cv2.cvtColor(image_raw, cv2.COLOR_BAYER_GB2BGR)
        self.PC.rectifyImage(image_bayer, image_bayer)
        return image_bayer

    def process_rect_decompand(self, image_path):
        """
        Takes a raw data image and converts it to a decompanded rectified 16 bit image.
        """
        image_raw = read_tiff_image(image_path)
        image_bayer = cv2.cvtColor(image_raw, cv2.COLOR_BAYER_GB2BGR)
        self.PC.rectifyImage(image_bayer, image_bayer)
        image_decomp = self.decompand_lut[image_bayer]
        return image_decomp

    def process_rect_gated(self, image_path):
        """
        Takes a raw data gated image and converts it to a rectified 10 bit grayscale image
        """
        image_raw = read_tiff_image(image_path)
        if self.DEBUG:
            check_image(image_raw)
        self.PC.rectifyImage(image_raw, image_raw)
        return image_raw

    def process_rect_lut_gated8(self, image_path):
        """
        Takes a raw data gated image and converts it to a rectified bit shifted 8 bit grayscale image
        """
        image_raw = read_tiff_image(image_path)
        if self.DEBUG:
            check_image(image_raw)
        image_raw = self.gated_lut[image_raw]
        image_raw = np.right_shift(image_raw, 2).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image_raw)

    def process_comp(self, image):
        return self.compand_lut[image]