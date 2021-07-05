import numpy as np
import os


def export_as_kitti_calib(P_camera1l, P_camera1r, P_camera2l, P_camera2r, R0_rect, Tr_velo_to_cam, Tr_radar_to_cam, export_path):
    dict = {}
    dict['P0']=P_camera1l.flatten().tolist()
    dict['P1']=P_camera1r.flatten().tolist()
    dict['P2']=P_camera2l.flatten().tolist()
    dict['P3']=P_camera2r.flatten().tolist()
    dict['R0_rect']=R0_rect.flatten().tolist()
    dict['Tr_velo_to_cam']=Tr_velo_to_cam.flatten().tolist()
    dict['Tr_radar_to_cam']=Tr_radar_to_cam.flatten().tolist()

    with open(export_path, 'w') as f:
        for key in ['P0','P1','P2','P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_radar_to_cam']:
            matrix =  ' '.join(map(str, dict[key]))
            total = key + ': ' + matrix + '\n'
            f.writelines(total)

    print('exported dense calib in kitti format')




def load_kitti_calib_data(calib_dir, file):
    # Load 3x4 projection matrix

    # file_name = file.split('_')[1].split('.png')[0]
    # file_name = file_name[3:]
    calib_dir = os.path.join(calib_dir, file + '.txt')
    calib_file = open(calib_dir, "r")

    P1 = []
    for calibration_line in calib_file.readlines():
        splitted_line = calibration_line.split(" ")[1:13]
        # print splitted_line
        P1.append(splitted_line)

    calib_file.close()

    # Read velodyne to camera projection
    P = P1[2]  # reads camera_id 2 Projection Matrix
    P = np.reshape(P, [3, 4])
    P = np.array(P.astype(dtype=np.float32))

    # Read velodyne rectification
    R0_rect = P1[4][0:9]
    R0_rect = np.reshape(R0_rect, [3, 3])
    R0_rect = np.hstack((R0_rect, np.zeros([3, 1])))
    R0_rect = np.vstack((R0_rect, np.array([0, 0, 0, 1])))
    R0_rect = np.array(R0_rect.astype(dtype=np.float32))

    # Read velodyne to Camera
    velodyne_to_camera = P1[5][:]
    velodyne_to_camera = np.reshape(velodyne_to_camera, [3, 4])
    velodyne_to_camera = np.vstack((velodyne_to_camera, np.array([0, 0, 0, 1])))
    velodyne_to_camera = np.array(velodyne_to_camera.astype(dtype=np.float32))

    # Calculate Projection Matrix
    vtc = np.matmul(np.matmul(P, R0_rect), velodyne_to_camera)


    return velodyne_to_camera, P, R0_rect, vtc