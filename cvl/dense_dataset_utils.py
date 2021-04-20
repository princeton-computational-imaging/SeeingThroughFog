import numpy as np

from typing import List, Tuple


def compare_points(path_last: str, path_strongest: str, min_dist: float=3.) -> \
        Tuple[np.ndarray, List[bool], float, float, float]:

    pc_l = np.fromfile(path_last, dtype=np.float32)
    pc_l = pc_l.reshape((-1, 5))

    pc_s = np.fromfile(path_strongest, dtype=np.float32)
    pc_s = pc_s.reshape((-1, 5))

    num_last = len(pc_l)
    num_strongest = len(pc_s)

    if num_strongest > num_last:
        pc_master = pc_s
        pc_slave = pc_l
    else:
        pc_master = pc_l
        pc_slave = pc_s

    mask = []
    diff = abs(num_strongest - num_last)

    for i in range(len(pc_master)):

        try:

            match_found = False

            for j in range(0, diff + 1):

                if (pc_master[i, :3] == pc_slave[i - j, :3]).all():
                    match_found = True
                    break

            mask.append(match_found)

        except IndexError:
            mask.append(False)

    dist = np.linalg.norm(pc_master[:, 0:3], axis=1)
    dist_mask = dist > min_dist

    mask = np.logical_and(mask, dist_mask)

    return pc_master, mask, num_last, num_strongest, diff