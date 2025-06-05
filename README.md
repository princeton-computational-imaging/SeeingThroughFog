SeeingThroughFog
============================

[Paper Link](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/figures/AdverseWeatherFusion.pdf)     &nbsp; &nbsp;    [Dataset Link](https://light.princeton.edu/datasets/automated_driving_dataset/)

We introduce a object detection benchmark in challenging adverse weather conditions covering 12000 samples in real world driving scenes and 1500 samples in controlled weather conditions within a fog chamber. The dataset covers different weather conditions as Fog, Snow and Rain and was acquired by over 10,000 km of driving in northern Europe. The driven route with cities along the route are shown below. In total 100k Objekts where labeled with accurate 2D and 3D bounding boxes. Main contributions of this dataset are:

- We provide a proving ground for a broad range of algorithms covering signal enhancement, domain adaptation, object detection or multi-modal sensor fusion focusing on the learning of robust redundancies in between sensor especially if they fail asymmetrically in different weather conditions.
- The dataset was created with the initial intention to showcase methods learning robust redundancies in between sensor enabling raw data sensor fusion in case of asymmetric sensor failure induced through adverse weather effects.
  In our case we departed from proposal level fusion and applied a adaptive fusion driven by measurement entropy enabling the detection also in case of unknown adverse weather effects. This method outperforms other reference fusion methods, which even drop in below single image methods.
- Furthermore, we compared the method to variety of related methods as domain adaptation, signal enhancement and adverse weather simulation. Please see [paper](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/figures/AdverseWeatherFusion.pdf) for more information.



<img src="./SeeingThroughFog/DatasetPicture.png"  width="500">



### Videos


##### Real-world recording in dense fog.

<img src="./SeeingThroughFog/dense_fog.gif" width="500">

##### Real-world recording in heavy snowfall.

<img src="./SeeingThroughFog/heavy_snowfall.gif" width="500">

##### Real world recording in snow dust.

<img src="./SeeingThroughFog/snow_dust.gif" width="500">


## Getting Started

Clone the benchmark code.
```
git clone https://github.com/princeton-computational-imaging/SeeingThroughFog
cd SeeingThroughFog
```

For running the evaluation and visualization code you need Python with the following packages:
- tensorflow
- numpy
- cv2
- matplotlib
- qt
- pyqtgraph
- pyquaternion

We provide a conda environment to run our code.
```
conda env create -f environment.yaml
```

Activate the conda environment.
```
conda activate LabelTool
```

Download the SeeingThroughFog dataset from the [Dataset Website](https://light.princeton.edu/datasets/automated_driving_dataset/). Please download all split zip files named SeeingThroughFogCompressed.zXX.
After downloading you can first check the dataset integrity by comparing the sha256sum. The file SeeingThroughFog_sha256sum.txt is part of this repository. 
````
sha256sum -c SeeingThroughFog_sha256sum.txt
````
Secondly you will have to unzip the dataset twice. The first unzipping can be performed using 7z retrieving each compressed split sensor zip. 
You can install 7z applying:
```
sudo apt install p7zip-full
```
The extraction command is as follows,
```
7z x SeeingThroughFogCompressed.zip 
```
The extraction will return you a directory structure containing each sensor in a subdirectory split into compressed zip archives.  
To extract those sensors you can run
```
bash extract.sh
```
Take care to comment the sensors you don't need to save storage and change the source "path_root" and destination path "dest_root".
The labels are stored in the gt_labels folder and labeltool_labels zip and not handeled by the extract.sh script. The gt_labels folder contains all object annotations per sample.
The labeltool_labels.zip contains all meta data about environment conditions including weather, road state and illumination conditions. 

Currently, the dataset is experiencing more demand than expected. This leads to unexpected downtimes we want to excuse. 
Accessing the download page through firefox is recommended and maintained with priority. In case of unavailability, please 
reach out to the corresponding authors with a detailed error message/screenshot (including operating system and browser) via mail. 
Usually, the availability is restored within a week. Afterwards, you will have to register again to retrieve a new download link. 

After unzipping the files, your directory should look like this:
```
.
|-- SeeingThroughFogData
    |-- cam_stereo_left
        |-- *.tiff
        |-- ...
    |-- cam_stereo_left_lut
        |-- *.png
        |-- ...
    |-- lidar_hdl64_last
        |-- *.bin
        |-- ...
    |-- lidar_hdl64_strongest
        |-- *.bin
        |-- ...
    |-- ...
```

Each folder has following meaning:

| Folder                        | Extrinsic Coordinate System  | Necessary for DataViewer   | Description                                                                                                 |
|-------------------------------|--------------------------|---|-------------------------------------------------------------------------------------------------------------|
| cam_stereo_left               | cam_stereo_left_optical  | -  | 12-bit RGB images captured from the left stereo camera.                                                     |
| cam_stereo_left_lut           | cam_stereo_left_optical  | x  | 8-bit RGB left stereo camera images used for annotating the dataset, with custom tone mapping presented in tools/Raw2LUTImages/main.py. |
| cam_stereo_right              | cam_stereo_right_optical | -  | 12-bit RGB images captured from the right stereo camera.                                                    |
| cam_stereo_right_lut          | cam_stereo_right_optical | x  | 8-bit RGB right stereo camera images with custom tone mapping.                                              |
| cam_stereo_sgm                | -                        | -  | Stereo disparity calculated using Stereo Global Matching (SGM).                                        |
| cam_stereo_left_raw_history_\<d>  | cam_stereo_left_optical  | -  | Temporal History data for \<d> in [-5,3] which represents the relativ offset to the annotated samples in cam_stereo_left. |
| fir_axis                      | fir_axis_roof_optical    | -  | 8-bit FIR reference camera. |
| gated\<d>_raw                 | bwv_cam_optical          | -  | 10-bit NIR gated camera image for slice index \<d>. |
| gated\<d>_rect                | bwv_cam_optical          | x  | 10-bit rectified NIR gated camera image for slice index \<d>. |
| gated\<d>_8bit                | bwv_cam_optical          | -  | 8-bit rectified bitshifted NIR gated camera image for slice index \<d> used for dataset annotation.|
| gated_full_rect               | bwv_cam_optical          | x  | 10-bit NIR gated camera image with joint slices from a single capture.|
| gated_full_rect8              | bwv_cam_optical          | -  | 8-bit rectified bitshifted NIR gated camera image with joint slices from a single capture. |
| gated_full_acc_rect           | bwv_cam_optical          | -  | 10-bit NIR gated camera image with overlayed gated slices from gated\<d>_raw. |
| gated_full_acc_rect8          | bwv_cam_optical          | x  | 8-bit rectified bitshifted NIR gated camera image with overlayed gated slices from gated\<d>_8bit. |
| radar_targets                 | radar                    | -  | Radar target pointcloud. |
| lidar_hdl64_last              | lidar_hdl64_s3_roof      | x  | Lidar pointcloud acquired from the Velodyne HDL64-S3D containing the last echo. |
| lidar_hdl64_last_gated        | bwv_cam_optical          | x  | Lidar pointcloud acquired from the Velodyne HDL64-S3D containing the last echo projected into gated camera. |
| lidar_hdl64_last_stereo_left  | cam_stereo_left_optical  | x  | Lidar pointcloud acquired from the Velodyne HDL64-S3D containing the last echo projected into left stereo image. |
| lidar_hdl64_strongest              | lidar_hdl64_s3_roof     | x  | Lidar pointcloud acquired from the Velodyne HDL64-S3D containing the strongest echo. |
| lidar_hdl64_strongest_gated        | bwv_cam_optical         | x  | Lidar pointcloud acquired from the Velodyne HDL64-S3D containing the strongest echo projected into gated camera. |
| lidar_hdl64_strongest_stereo_left  | cam_stereo_left_optical | x  | Lidar pointcloud acquired from the Velodyne HDL64-S3D containing the strongest echo projected into left stereo image. |
| lidar_vlp32_last              | lidar_vlp32_roof         | -  | Lidar pointcloud acquired from the Velodyne VLP32 containing the last echo. |
| lidar_vlp32_strongest         | lidar_vlp32_roof         | -  | Lidar pointcloud acquired from the Velodyne VLP32 containing the strongest echo. |
| velodyne_planes               | -                        | -  | Lidar groundplanes estimated from the lidar_hdl64_strongest measurements. |
| road_friction                 | -                        | x  | Road friction measurements. |
| weather_station               | -                        | x  | Ambient temperature, humidity, dew point, ... |
| labeltool_labels              | -                        | x  | Meta labels on ambient illumination, drivable patch, scene setting and weather. |
| filtered_relevant_can_data    | -                        | x  | Vehicle data on wiper state, speed, ... |
| gt_labels                     | -                        | x  | 3D annotation labels following the Kitti annotation layout. cam_left_labels_TMP contains the labels inside the stereo camera left coordinate system and gated_labels_TMP in the gated camera coordinate system. |

The extrinsic coordinate transormations in the third column are given in tools/DatasetViewer/calibs/calib_tf_tree_full.json. It follows the general tansformation logic from ROS.

For an initial reasoning about the dataset the DataViwer in tools/DatasetViewer/DataViewer_V2.py can be used. The needed data is marked in the table above.


### Sensor Setup

To total we have equipped a test vehicle with sensors covering the visible, mm-wave, NIR and FIR band, see Figure below. We measure intensity, depth and weather condition. For more information please refer to our dataset paper.

<img src="./SeeingThroughFog/car_setup_daimler.png" width="500">

### Labeling
The final dataset provides about 12000 samples from an extensive data collection campaing with in total 1.4 mio samples. 
The subsampling process is described in depth within the supplemental material. Which can be found [here](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/figures/AdverseWeatherFusion_Supplement.pdf).
We provide 2d and 3d annotations for 4 main classes PassengerCars, Pedestrians, RidableVehicles, LargeVehicles. 
If PassengerCars, RidableVehicles and LargeVehicles can not be differentiated the Objects are Labeled within the fallback class 
Vehicle. In case Vehicle and Pedestrians can not be differentiated Objects are labeled as Obstacles. 
Below you can find the illustrated LabelDefinition:

<img src="./SeeingThroughFog/LabelDefinition.png" width="500">

The Annotation format follows the well known [Kitti Format](http://www.cvlibs.net/datasets/kitti/)
with added coloums for 3d Box Rotation angles. You can find the loading functionalities in 

```
./tools/DatasetViewer/lib/read.py
```

Recomended data splits can be found in

```
./splits
```


### Tools
Tools can be found in 
```
"./SeeingThroughFog/tools/".
```
The tools help to visualize the dataset (DatasetViewer), create a TFRecords dataset (CreateTFRecords), 
create simple fog simulations for lidar and rgb data
(DatasetFoggification) and to quickly reason about the dataset statistics (DatasetStatisticsTool).  


### Reference

If you find our work on object detection in adverse weather useful in your research, please consider citing our [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bijelic_Seeing_Through_Fog_Without_Seeing_Fog_Deep_Multimodal_Sensor_Fusion_CVPR_2020_paper.html):

```
@InProceedings{Bijelic_2020_STF,
    author={Bijelic, Mario and Gruber, Tobias and Mannan, Fahim and Kraus, Florian and Ritter, Werner and Dietmayer, Klaus and Heide, Felix},
    title={Seeing Through Fog Without Seeing Fog:
    Deep Multimodal Sensor Fusion in Unseen Adverse Weather},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

## Acknowledgements
These works have received funding from the European Union under the H2020 ECSEL Programme as part of the DENSE project, contract number 692449.


## Feedback/Questions/Error reporting
Feedback? Questions? Any problems/errors? Do not hesitate to contact us!
