import numpy as np
import os

def loadKneepoints(kneepoints_list, bitdepth=16):
    line_cnt = 1
    x1 = -1
    y1 = -1
    x2 = -1
    y2 = -1
    compression = 0
    kneepoints = dict()
    for x2, y2 in kneepoints_list:
        if (x2 > 2**bitdepth or (y2 > x2 > 2**bitdepth )):
            print("ERROR while parsing decompandig LUT. - A Kneepoint has to have a higher x-value than its precessor!")
            exit(-1)
        if ((x2-x1) <= 0):
            print("ERROR while parsing decompandig LUT. - A Kneepoint has to have a higher x-value than its precessor!")
            exit(-1)
        compression = (y2 - y1) / (x2 - x1)
        kneepoints[x2] = (y2, compression)
        x1 = x2
        y1 = y2
        line_cnt += 1

    return kneepoints

def create_decompand_lut(kneepoints, DEBUG=False):

    decompanded = 0
    src_min = 0
    dst_min = 0
    decompandedLUT = []
    i = 0
    for src_max in sorted(kneepoints.keys()):
        dst_max, compression = kneepoints[src_max]
        for src in range(src_min, src_max+1):
            decompanded = (src - src_min)*compression + dst_min
            if decompanded>dst_max:
                decompanded = dst_max
            decompandedLUT.append(decompanded)
        if DEBUG:
            print("Decompanding section %d : SRC  %d  to %d  ---> DST: %d  to %d"%(i, src_min, src_max, dst_min, decompanded))
        src_min = src_max+1
        dst_min = dst_max+1
        i = i+1
    return np.asarray(decompandedLUT,  dtype=np.uint16).reshape((len(decompandedLUT), ))
