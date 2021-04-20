import os
import glob

from pathlib import Path

########################################################

log_name = '2nd_run.log'    # specify your log file here

########################################################

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

log_dir = dir_path.parent.parent.parent / 'data' / 'dense'
log_path = log_dir / log_name

split_dir = dir_path.parent / 'splits'
split_paths = sorted(glob.glob(f'{split_dir}/*.txt'))

with open(log_path, 'r') as log_file:
    lines = log_file.readlines()

debug_lines = [line for line in lines if 'DEBUG' in line]
warning_lines = [line for line in lines if 'WARNING' in line]

empty_scenes = []
scenes_with_empty_objects = []

for debug_line in debug_lines:

    mentioned_scene = debug_line.split(' ')[5]
    scenes_with_empty_objects.append(mentioned_scene)

for warning_line in warning_lines:

    mentioned_scene = warning_line.split(' ')[5] + '\n'
    empty_scenes.append(mentioned_scene)

empty_scenes = set(empty_scenes)
scenes_with_empty_objects = set(scenes_with_empty_objects)

for split_path in split_paths:

    with open(split_path, 'r') as split_file:
        lines = split_file.readlines()

    for scene in empty_scenes:

        try:
            lines.remove(scene)
        except ValueError:
            pass

    with open(split_path, 'w') as split_file:
        split_file.writelines(lines)

print(f'There are still {len(scenes_with_empty_objects)} scenes with empty objects.')