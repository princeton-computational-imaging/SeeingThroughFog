#!/bin/bash

#Uncomment sensors you need for your research

declare -a folder=(
"calib_cam_stereo_left.json"
"calib_gated_bwv.json"
"calib_tf_tree_full.json"
#"radar_targets"
#"cam_stereo_left"
#"cam_stereo_left_lut"
#"cam_stereo_left_raw_history_1"
#"cam_stereo_left_raw_history_-1"
#"cam_stereo_left_raw_history_2"
#"cam_stereo_left_raw_history_-2"
#"cam_stereo_left_raw_history_3"
#"cam_stereo_left_raw_history_-3"
#"cam_stereo_left_raw_history_4"
#"cam_stereo_left_raw_history_-4"
#"cam_stereo_left_raw_history_-5"
#"cam_stereo_left_raw_history_-6"
#"fir_axis"
#"gated0_raw"
#"gated0_rect"
#"gated0_rect8"
#"gated1_raw"
#"gated1_rect"
#"gated1_rect8"
#"gated2_raw"
#"gated2_rect"
#"gated2_rect8"
#"gated_full_acc_rect"
#"gated_full_acc_rect8"
#"gated_full_rect"
#"gated_full_rect8"
"lidar_hdl64_last"
#"lidar_hdl64_last_gated"
#"lidar_hdl64_last_stereo_left"
#"lidar_hdl64_strongest"
#"lidar_hdl64_strongest_gated"
#"lidar_hdl64_strongest_stereo_left"
#"lidar_vlp32_last"
#"lidar_vlp32_strongest"
#"road_friction"
#"weather_station"
)

# change path to your zipped dataset and destination path

path_root="SeeingThroughFogCompressed"
dest_root="SeeingThroughFogCompressedExtracted"

c="7z x"

mkdir $dest_root
function extract_and_move() {
    $c $path_root/$1/$1.zip -o$dest_root/ && echo "ended" $1
}
i=0
for j in ${folder[@]}
do
  if [[ "$j" == *json ]]
  then
    echo "cp $path_root/$j $dest_root"
    cp $path_root/$j $dest_root # sshpass -p "175K7_B.F"
  else
    echo "$c $path_root/$j $dest_root"
    extract_and_move $j &
    pids[$i]=$!
    i=$i+1
  fi
done

for pid in ${pids[*]}; do
    wait $pid
done