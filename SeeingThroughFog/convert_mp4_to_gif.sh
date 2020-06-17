#!/bin/bash

in=$1

ffmpeg -i $in -filter:v fps=fps=1 "${in%.*}"FPSLow.mp4 
ffmpeg -i "${in%.*}"FPSLow.mp4 -vf scale=1920:-1 "${in%.*}".gif
