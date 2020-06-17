import argparse
import os
import time

import numpy as np
import pyqtgraph.opengl as gl
from beta_modification import BetaRadomization
from pyqtgraph.Qt import QtGui



#fog density

def load_velo_scan(file):
    """Load and parse a velodyne binary file. According to Kitti Dataset"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 5))[:,0:4]


def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar Fog Simulation Filename')
    parser.add_argument('--root', '-r', help='Enter the root folder', default='./example_data/')
    parser.add_argument('--velodyne_folder', '-v', help='Data folder Velodyne', default='LidarData')
    parser.add_argument('--beta', '-b', type=float, help='Enter the fogdensity beta here', default=0.05)
    parser.add_argument('--fraction_random', type=float, default=0.05, help ='Enter fraction of random scattered points')
    parser.add_argument('--sensor_type', type=str, default='VelodyneHDLS3D', help='chose sensor type either "VelodyneHDLS3D" or VelodyneHDLS2')

    args = parser.parse_args()
    args.destination_folder = 'velodyne_points_beta%.5f'%args.beta
    global hazed

    return args


def haze_point_cloud(pts_3D, Radomized_beta, args):
    #print 'minmax_values', max(pts_3D[:, 0]), max(pts_3D[:, 1]), min(pts_3D[:, 1]), max(pts_3D[:, 2]), min(pts_3D[:, 2])
    n = []
    # foggyfication should be applied to sequences to ensure time correlation inbetween frames
    # vectorze calculation
    # print pts_3D.shape
    if args.sensor_type=='VelodyneHDLS3D':
        # Velodyne HDLS643D
        n = 0.04
        g = 0.45
        dmin = 2 # Minimal detectable distance
    elif args.sensor_type=='VelodyneHDLS2':
        #Velodyne HDL64S2
        n = 0.05
        g = 0.35
        dmin = 2
    d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
    detectable_points = np.where(d>dmin)
    d = d[detectable_points]
    pts_3D = pts_3D[detectable_points]

    beta_usefull = Radomized_beta.get_beta(pts_3D[:,0], pts_3D[:, 1], pts_3D[:, 2])
    dmax = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta_usefull))
    dnew = -np.log(1 - 0.5) / (beta_usefull)

    probability_lost = 1 - np.exp(-beta_usefull*dmax)
    lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost

    if Radomized_beta.beta == 0.0:
        dist_pts_3d = np.zeros((pts_3D.shape[0], 5))
        dist_pts_3d[:, 0:4] = pts_3D
        dist_pts_3d[:, 4] = np.zeros(np.shape(pts_3D[:, 3]))
        return dist_pts_3d,  []

    cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))
    idx_stable = np.where(d<dmax)[0]
    old_points = np.zeros((len(idx_stable), 5))
    old_points[:,0:4] = pts_3D[idx_stable,:]
    old_points[:,3] = old_points[:,3]*np.exp(-beta_usefull[idx_stable]*d[idx_stable])
    old_points[:, 4] = np.zeros(np.shape(old_points[:,3]))

    cloud_scatter_idx = np.where(np.logical_and(dmax<d, cloud_scatter))[0]
    cloud_scatter = np.zeros((len(cloud_scatter_idx), 5))
    cloud_scatter[:,0:4] =  pts_3D[cloud_scatter_idx,:]
    cloud_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(cloud_scatter[:,0:3]), np.transpose(np.divide(dnew[cloud_scatter_idx],d[cloud_scatter_idx]))))
    cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta_usefull[cloud_scatter_idx]*dnew[cloud_scatter_idx])
    cloud_scatter[:, 4] = np.ones(np.shape(cloud_scatter[:, 3]))


    # Subsample random scatter abhaengig vom noise im Lidar
    random_scatter_idx = np.where(random_scatter)[0]
    scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
    drand = np.random.uniform(high=scatter_max[random_scatter_idx])
    # scatter outside min detection range and do some subsampling. Not all points are randomly scattered.
    # Fraction of 0.05 is found empirically.
    drand_idx = np.where(drand>dmin)
    drand = drand[drand_idx]
    random_scatter_idx = random_scatter_idx[drand_idx]
    # Subsample random scattered points to 0.05%
    print(len(random_scatter_idx), args.fraction_random)
    subsampled_idx = np.random.choice(len(random_scatter_idx), int(args.fraction_random*len(random_scatter_idx)), replace=False)
    drand = drand[subsampled_idx]
    random_scatter_idx = random_scatter_idx[subsampled_idx]


    random_scatter = np.zeros((len(random_scatter_idx), 5))
    random_scatter[:,0:4] = pts_3D[random_scatter_idx,:]
    random_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(random_scatter[:,0:3]), np.transpose(drand/d[random_scatter_idx])))
    random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta_usefull[random_scatter_idx]*drand)
    random_scatter[:, 4] = 2*np.ones(np.shape(random_scatter[:, 3]))



    dist_pts_3d = np.concatenate((old_points, cloud_scatter,random_scatter), axis=0)

    color = []
    return dist_pts_3d, color


def initialize_window():
    w= None
    return w

def add_random_noise(velodyne_scan):
    random_noise = np.random.normal(0.0, 5, np.shape(velodyne_scan))
    velodyne_scan = velodyne_scan + random_noise
    return  velodyne_scan


def set_color(dist_pts_3d):
    color = []
    for pts in dist_pts_3d:
        if pts[4] == 0:
            color.append([0, 255, 255, 1])
        else:
            color.append([pts[3],0, 0, 1])

    return np.asarray(color)



def main(walk_path, dest_path, beta, args, DEBUG = True):
    #os.path.join(root_in, folder)
    print(walk_path)
    files_all = []
    for root, dirs, files in os.walk(walk_path, followlinks=True):
        print(root)
        assert(root==walk_path)
        files_all = sorted(files)
    print(files_all)
    if DEBUG:
        w = initialize_window()

        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.show()
        w.setWindowTitle('Velodyne Pointlcloud')
        #     86.8051380062 273 - 59
        w.setCameraPosition(pos=[0,0,0], distance=86.8051380062, azimuth=180, elevation=40)

    #g = gl.GLGridItem()
    #w.addItem(g)
    boxes = []

    for i in range(0, len(files_all)):
        B = BetaRadomization(beta)
        B.propagate_in_time(10)
        file1 = files_all[i]
        print(os.path.join(walk_path,  file1))
        velodyne_scan = load_velo_scan(os.path.join(walk_path, file1))
        velodyne_scan[:,3] = velodyne_scan[:,3]/255
        start = time.time()
        dist_pts_3d, color = haze_point_cloud(velodyne_scan, B, args)
        end = time.time()
        print('elapsed_time', end - start)

        if DEBUG:
            pass
            color = set_color(dist_pts_3d)
            if color is None:
                plot = gl.GLScatterPlotItem(pos=dist_pts_3d[:,0:3], size=3)
            else:
                plot = gl.GLScatterPlotItem(pos=dist_pts_3d[:,0:3], size=3, color=color)
            w.addItem(plot)
            w.update()
            w.show()
            app.exec_()
            #save_processed_image
            #w.grabFrameBuffer().save('test%02d.png' % i)
            w.removeItem(plot)
            print('viewing coordinates', w.opts['distance'], w.opts['azimuth'], w.opts['elevation'])

        #Update position in time
        B.propagate_in_time(5)
        save_path_velo = os.path.join(dest_path, file1)
        dist_pts_3d.astype(np.float32).tofile(save_path_velo)


if __name__ == '__main__':

    args = parsArgs()
    args.destination_folder ='hazing/velodyne_points_beta%.5f'%args.beta
    walk_path = os.path.join(args.root, args.velodyne_folder)
    dest_folder = os.path.join(args.root, args.destination_folder)
    print(walk_path, args.beta)
    print(dest_folder, args.beta)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    print('started')
    main(walk_path, dest_folder, beta=args.beta, args=args, DEBUG=True)



