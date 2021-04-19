import cv2
import numpy as np



def get_iou(bb1, bb2):
    """
    Taken from: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    #print  bb1['x1'], bb1['x2']
    assert bb1['x1'] <= bb1['x2']
    #print bb1['y1'], bb1['y2']
    assert bb1['y1'] <= bb1['y2']
    #print bb2['x1'], bb2['x2']
    assert bb2['x1'] <= bb2['x2']
    #print bb2['y1'], bb2['y2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if bb1_area == bb2_area == 0:
        return 1.0
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    #print iou, bb1_area, bb2_area, intersection_area
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class resize():

    def __init__(self, mode='default'):
        """
        Setup standart resize mode in initialization.
        Resizes image to desired frame with r.crop(image)
        Resizes boxes accordingly using r.crop_bboxes(boxes)
        Returns correct image projection matrix through r.get_image_scaling()
        """
        if mode == 'default':
            # no crop
            self.crop_height_lower = 0
            self.crop_height_higher = 1024
            self.crop_width_lower = 0
            self.crop_width_higher = 1920
            self.scaling_fx = 1.0  # None
            self.scaling_fy = 1.0  # None
            self.dsize = (1920, 1024)  # 2239
        if mode=='Sweden2PSMNet':
            self.crop_height_lower = 0
            self.crop_height_higher = 1024
            self.crop_width_lower = 0
            self.crop_width_higher = 1920
            self.scaling_fx = 1.0/2.0
            self.scaling_fy = 1.0/2.0
            self.dsize = (960, 512)
        if mode == 'RGB2Gatedv2':
            self.crop_height_lower = 202
            self.crop_height_higher = 970
            self.crop_width_lower = 280
            self.crop_width_higher = 1560
            self.scaling_fx = 1.0  # None
            self.scaling_fy = 1.0  # None
            self.dsize = (1280, 768)
        if mode == 'RGB2Gatedv2Fogchamber':
            self.crop_height_lower = 258
            self.crop_height_higher = 770
            self.crop_width_lower = 510
            self.crop_width_higher = 1534
            self.scaling_fx = 1.0  # None
            self.scaling_fy = 1.0  # None
            self.dsize = (1024, 512)

    def crop(self, image):

        image_cropped = np.copy(
            image[self.crop_height_lower:self.crop_height_higher, self.crop_width_lower:self.crop_width_higher, :])

        if self.scaling_fx is not None:
            image_cropped = cv2.resize(image_cropped, self.dsize, interpolation=cv2.INTER_AREA)

        return image_cropped

    def crop_bboxes(self, bbox):
        """

        :param bbox: [ymax, xmax, ymin, xmin]
        :return: bbox_scaled: [ymax, xmax, ymin, xmin]
        """
        if bbox is not None:
            bbox[0] = np.clip(bbox[0] - self.crop_height_lower, 0, self.crop_height_higher - self.crop_height_lower)
            bbox[2] = np.clip(bbox[2] - self.crop_height_lower, 0, self.crop_height_higher - self.crop_height_lower)
            bbox[1] = np.clip(bbox[1] - self.crop_width_lower, 0, self.crop_width_higher - self.crop_width_lower)
            bbox[3] = np.clip(bbox[3] - self.crop_width_lower, 0, self.crop_width_higher - self.crop_width_lower)
            if self.scaling_fx is not None:
                bbox[0] = int(self.scaling_fy * bbox[0])
                bbox[2] = int(self.scaling_fy * bbox[2])
                bbox[1] = int(self.scaling_fx * bbox[1])
                bbox[3] = int(self.scaling_fx * bbox[3])
               
            
            return bbox

    def crop_bboxes_inverted_xy(self, bbox, truncation):
        """

        :param bbox: [xmax, ymax, xmin, ymin]
        :return: bbox_scaled: [xmax, ymax, xmin, ymin]
        """
        box_in = dict()
        box_in['x1'] = bbox[0]-1
        box_in['x2'] = bbox[2]
        box_in['y1'] = bbox[1]-1
        box_in['y2'] = bbox[3]

        if bbox is not None:
            box_reference = dict()
            box_reference['y1'] = np.clip(bbox[1], self.crop_height_lower, self.crop_height_higher)
            box_reference['y2'] = np.clip(bbox[3], self.crop_height_lower, self.crop_height_higher)
            box_reference['x1'] = np.clip(bbox[0], self.crop_width_lower, self.crop_width_higher)
            box_reference['x2'] =  np.clip(bbox[2], self.crop_width_lower, self.crop_width_higher)

            bbox[1] = np.clip(bbox[1] - self.crop_height_lower, 0, self.crop_height_higher - self.crop_height_lower)
            bbox[3] = np.clip(bbox[3] - self.crop_height_lower, 0, self.crop_height_higher - self.crop_height_lower)
            bbox[0] = np.clip(bbox[0] - self.crop_width_lower, 0, self.crop_width_higher - self.crop_width_lower)
            bbox[2] = np.clip(bbox[2] - self.crop_width_lower, 0, self.crop_width_higher - self.crop_width_lower)
            
            truncation_new = 1 - get_iou(box_in, box_reference)
            #print('truncation_new', truncation_new,  get_iou(box_in, box_reference))
            truncation = np.clip(truncation + truncation_new, 0, 1)
            if self.scaling_fx is not None:
                bbox[0] = float(self.scaling_fx * bbox[0])
                bbox[2] = float(self.scaling_fx * bbox[2])
                bbox[1] = float(self.scaling_fy * bbox[1])
                bbox[3] = float(self.scaling_fy * bbox[3])

            return bbox, truncation

    def get_image_scaling(self):
        """
        Takes given croppging parameters and
        :return: Image Projection Matrix for rescaled image
        """
        return np.asarray([[self.scaling_fx, 0, -self.scaling_fx * self.crop_width_lower],
                           [0, self.scaling_fy, -self.scaling_fy * self.crop_height_lower], [0, 0, 1]])

def AddPaddingToImage(input_image, dst_shape, method='REFLECTION'):
    """Adds a border to the image by specified padding method to obtain
       a image which has the intended shape


    input_image -- image to upsample
    dst_shape -- aimed output shape of image
    method -- padding method i.e. reflection padding
    """
    input_image_shape = np.array(
        [1, np.size(input_image, 0), np.size(input_image, 1), np.size(input_image, 2)])

    if np.array_equal(input_image_shape, dst_shape):
        return [input_image, [0, 0]]

    if np.less(input_image_shape, dst_shape).all():
        raise ValueError(
            "Error: Input image shape is smaller than input shape expected by network.")

    diff_shape = np.subtract(dst_shape, input_image_shape)

    # Calculate paddings -- consider odd differences!
    padding_top = int(diff_shape[1] / 2 + diff_shape[1] % 2)
    padding_bottom = int(diff_shape[1] / 2)

    padding_left = int(diff_shape[2] / 2 + diff_shape[2] % 2)
    padding_right = int(diff_shape[2] / 2)

    paddings = [padding_top, padding_left]

    # do the upsampling by reflection
    if method is 'REFLECTION':
        upsampled_image = cv2.copyMakeBorder(
            input_image,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_REFLECT)
    else:
        raise ValueError("Error: Upsampling method {} is not implemented!".format(method))

    return upsampled_image, paddings



def RemovePaddingFromImage(image, dst_shape, padding):
    """Removes the padding from the image

    image -- image containing the padded border
    dst_shape -- aimed output shape of image
    padding -- pixel paddings of top and left
    """
    y1 = dst_shape[0] + padding[0]
    y2 = dst_shape[1] + padding[1]
    output = image[0, padding[0]:y1, padding[1]:y2,:]
    return output

