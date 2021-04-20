import cv2
import glob

from tqdm import tqdm
from pathlib import Path

directories = ['/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/KITTI/3D/testing/image_2',
               '/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/KITTI/3D/testing/image_3']

WIDTH = 1224
HEIGHT = 368

for directory in directories:

    files = sorted(glob.glob(f'{directory}/*.png'))

    save_directory = Path(f'{directory}_{WIDTH}x{HEIGHT}')
    save_directory.mkdir(parents=True, exist_ok=True)

    print(f'Starting to create {save_directory}')

    for file in tqdm(files):

        file_path = Path(file)
        file_name = file_path.name

        img = cv2.imread(file)

        height, width, channels = img.shape

        scaled_img = cv2.resize(img, (WIDTH, HEIGHT))

        cv2.imwrite(f'{save_directory}/{file_name}', scaled_img)


print('Done!')