import numpy as np
import cv2
from lib.utils import rotx_matrix, roty_matrix, rotz_matrix

# Assign color to each label for visualization
LABEL_TO_CLR = {"Don't care region": (10, 10, 10),
                'RidableVehicle': (128, 255, 10),
                'Obstacle': (120, 120, 120),
                'Pedestrian': (10, 10, 255),
                'PassengerCar': (255, 10, 10),
                'LargeVehicle': (10, 255, 10),
                'Vehicle': (10, 120, 120)
                }

def project_3d_to_2d(points3d, P):
    points2d = np.matmul(P, np.vstack((points3d, np.ones([1, np.shape(points3d)[1]]))))

    # scale projected points
    points2d[0][:] = points2d[0][:] / points2d[2][:]
    points2d[1][:] = points2d[1][:] / points2d[2][:]

    points2d = points2d[0:2]
    return points2d.transpose()


def project_bbox3d_to_2d(bbox3d, P, M=None):
    corners = build_bbox3d_from_params(bbox3d, M)
    return project_points_to_2d(corners, P)


def project_points_to_2d(points3d, P):
    points2d = np.dot(P[:3, :3], points3d.T).T + P[:3, 3]
    points2d = points2d[:, :2] / points2d[:, 2][:, np.newaxis]
    points2d = points2d.astype(np.int32)
    return points2d


def build_bbox3d_from_params(bbox3d, zero_to_camera=None):
    if zero_to_camera is None:
        # Use homogeneous coordinates to apply 3D transformation
        zero_to_camera = np.eye(4)

    bc = np.array([bbox3d['posx'], bbox3d['posy'], bbox3d['posz']])
    od = np.array([bbox3d['length'], bbox3d['width'], bbox3d['height']])

    invert_rotation = np.linalg.inv(zero_to_camera[0:3, 0:3])
    invert_translation = -zero_to_camera[0:3, 3]

    camera_to_zero = np.zeros_like(zero_to_camera)
    camera_to_zero[0:3, 0:3] = invert_rotation
    camera_to_zero[0:3, 3] = invert_translation

    qM = np.matmul(rotx_matrix(-bbox3d['rotx']),np.matmul(roty_matrix(-bbox3d['roty']), rotz_matrix(-bbox3d['rotz'])))

    # Create initial bounding box in base frame
    box_base = np.array([[-od[0] / 2, 0, -od[2] / 2],
                         [od[0] / 2, 0, -od[2] / 2],
                         [od[0] / 2, 0, od[2] / 2],
                         [-od[0] / 2, 0, od[2] / 2]])

    # Rotate Box around origin
    box_base = np.concatenate((box_base+np.array([0, od[1], 0])/2, box_base - np.array([0, od[1], 0])/2))
    box_base = np.matmul(qM[:3, :3], box_base.T).T

    # Add height offset
    box_base = box_base + np.array([0, 0, od[2]])/2
    box_base = np.matmul(camera_to_zero[0:3, 0:3], box_base.T).T
    corners = bc + box_base[:, :3]
    return corners


def draw_bbox3d(img, box3d):
    # Different colors for 3D bounding box: ground (lengthwise and crosswise), height, and top
    color_bbox3d = [(255, 20, 20),
                    (20, 20, 255),
                    (255, 20, 20),
                    (20, 20, 255)]

    for index in range(4):
        img = cv2.line(img, tuple(box3d[index]), tuple(box3d[(index + 1) % 4]), color_bbox3d[index], 1)
        img = cv2.line(img, tuple(box3d[index + 4]), tuple(box3d[(index + 1) % 4 + 4]), (20, 20, 255), 1)
        img = cv2.line(img, tuple(box3d[index]), tuple(box3d[index + 4]), (20, 255, 20), 1)

    # Draw the 3 axes
    img = cv2.line(img, tuple(box3d[0]), tuple(box3d[1]), (255, 0, 0), 2)
    img = cv2.line(img, tuple(box3d[0]), tuple(box3d[3]), (0, 0, 255), 2)
    img = cv2.line(img, tuple(box3d[0]), tuple(box3d[4]), (0, 255, 0), 2)
    return img


def draw_bbox2d_from_kitti(image, label, color=(255, 0, 0)):
    if not label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'No label found!', (200, 200), font, 4, color, 2, cv2.LINE_AA)
        return image
    #x = tuple((label['xleft'], label['ytop']))
    #y = tuple((label['xright'], label['ybottom']))
    x = tuple((label['xleft'], label['ytop']))
    y = tuple((label['xright'], label['ybottom']))
    cv2.rectangle(image, x, y, color, 2)

    return image
