import cv2
import os
import numpy as np
import json


def read_meta_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def parse_day_night(data, DEBUG=False):
    if DEBUG:
        print(data['daytime'])
    if data is None:
        return True
    day = data['daytime']['day']
    night = data['daytime']['night']
    assert day != night
    return day

def read_tiff_image(total_path):
    """
    reads raw 16bit tiff images
    :param path: folder Path
    :param name: image name
    :return: dtype= unit16 numpy array containing raw images
    """
    #return cv2.cvtColor(cv2.imread(os.path.join(path, name), -1), cv2.__getattribute__("COLOR_BAYER_" + "GR2RGB"))
    return cv2.imread(total_path, -1)

def save_tiff_image(path, name, image):
    """
    :param path: destination folder
    :param name: image name .png, .tiff denotes saving operation
    :param image: input image which has to be saved
    :return: None
    """
    cv2.imwrite(os.path.join(path, name), image)

def read_image_intrinsic(path_dataset, name_camera_calib):

    with open(os.path.join(path_dataset, name_camera_calib), 'r') as f:
        data_camera = json.load(f)

    return data_camera

def check_image(image, name='image'):

    print('#### entered %s check loop ####'%name)
    print('max', np.max(image))
    print('min', np.min(image))
    print('shape', np.shape(image))
    print(image.dtype)


BGR2LAB=[[0.1804375, 0.3575761, 0.4124564]/np.sum([0.1804375, 0.3575761, 0.4124564]),
         [0.0721750, 0.7151522, 0.2126729]/np.sum([0.0721750, 0.7151522, 0.2126729]),
         [0.0193339, 0.1191920, 0.9503041]/np.sum([[0.0193339, 0.1191920, 0.9503041]])
]

a=np.asarray(BGR2LAB)

LAB2BGR=np.linalg.inv(BGR2LAB)



def proj_BGR2LAB(image):
    img=np.copy(image).astype(np.float64)
    img2=np.copy(img)
    for i in range(img.shape[0]):
        a=np.matmul(BGR2LAB,img[i,:,:].transpose([1,0])).transpose([1,0])
        img2[i, :,:]=a

    return np.clip(img2, 0, 2**16).astype(np.uint16)

def proj_LAB2BGR(image):
    img=np.copy(image).astype(np.float64)
    img2=np.copy(img)
    for i in range(img.shape[0]):
        a=np.matmul(LAB2BGR,img[i,:,:].transpose([1,0])).transpose([1,0])
        img2[i, :,:]=a

    return np.clip(img2, 0, 2 ** 16).astype(np.uint16)

def apply_clahe_16bit(image):
    image = (2**-8*image).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:,:,0])
    clahe2 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    image[:, :, 1] = clahe2.apply(image[:,:,1])
    image[:, :, 2] = clahe2.apply(image[:,:,2])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return image

def apply_clahe_8bit(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:,:,0])
    clahe2 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    image[:, :, 1] = clahe2.apply(image[:,:,1])
    image[:, :, 2] = clahe2.apply(image[:,:,2])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return image