import json
import os

import matplotlib.pyplot as plt
import numpy as np

from os import walk
from typing import Set, Dict


def extract_shapes(path: str) -> Set:
    """
    Crawls dataset to find set of all shape values
    Example Usage: `extract_shapes('/mcs-cora/cora-derenderer/data')`

    Args:
        path: Path to dataset
        verbose: Whether to extensively print paths

    Returns:

    """
    shape_set = set()
    for (dir_path, dir_names, file_names) in walk(path):
        for file_name in file_names:
            if str(file_name).endswith('.json'):
                # print(f'Parsing {dir_path}/{file_name}...')
                with open(f'{dir_path}/{file_name}', 'r') as file:
                    json_file = json.load(file)
                    objects = json_file.get('objects', [])
                    for obj in objects:
                        shape_set.add(obj.get('shape', {}).get('shape_type', None))
    return shape_set


def scan_objects_for_cylinder(objects):
    """
    Scans objects for cylinder, returns number of objects if only boxes, returns None if cylinders present
    :param objects:
    :return:
    """
    for obj in objects:
        shape = obj.get('shape', {}).get('shape_type', None)
        if shape == 'cylinder':
            return None
    return len(objects)


def scan_frames_for_cylinder(base_path, frames):
    max_number = -1

    for frame in frames:
        frame_scene_graph_path = os.path.join(base_path, f'{frame}.json')
        with open(frame_scene_graph_path, 'r') as file:
            scenegraph = json.load(file)
            objects = scenegraph.get('objects', [])
            scan_result = scan_objects_for_cylinder(objects)
            if scan_result is None:  # If frame contains cylinder, skip entire video
                return None
            else:
                max_number = max(max_number, scan_result)

    return None if max_number == -1 else max_number


def find_box_exclusive_videos(path: str, count_max: int = 25) -> Dict:
    if not path.endswith('/'):
        path = f'{path}/'

    # Find all datasets with only cubes and count number of objects in them
    # Box is 0, cylinder is 1
    count = 0
    map_box_exclusive_video_to_object_number = dict()
    videos = sorted(set([video for video in os.listdir(path)]))
    print(f'Frames for video {path}:\n{videos}')

    for video in videos:
        frames = sorted(set([frame.replace('.json', '').replace('.seg.npy', '').replace('.depth.npy', '')
                             for frame in os.listdir(f'{path}{video}/')]))
        base_path = f'{path}{video}/'
        scan_result = scan_frames_for_cylinder(base_path, frames)
        if scan_result is not None:
            map_box_exclusive_video_to_object_number[video] = scan_result

    return map_box_exclusive_video_to_object_number


def plot_arrays(file_name_depth: str, file_name_segmentation: str):
    depth_array = np.load(file_name_depth)
    segmentation_array = np.load(file_name_segmentation)
    plt.imshow(depth_array)
    plt.show()
    plt.imshow(segmentation_array)
    plt.show()


def determine_depth_map_max(file_name_depth: str):
    depth_array = np.load(file_name_depth)
    print(depth_array.max())


def segment_depth_map(file_name_depth: str, file_name_segmentation: str):
    depth_array = np.load(file_name_depth)
    segmentation_array = np.load(file_name_segmentation)
    unique_segmentation_values = np.unique(segmentation_array)
    for segmentation_value in unique_segmentation_values:
        print(f'Segmentation value: {segmentation_value}')
        depth_array_masked = np.where((segmentation_array == segmentation_value), depth_array, 0)
        plt.imshow(depth_array_masked)
        plt.show()


print(find_box_exclusive_videos('/mcs-cora/cora-derenderer/data'))
# determine_depth_map_max('/Users/fpk/PycharmProjects/cora-derenderer/test_data/00002.depth.npy')
# plot_arrays('/Users/fpk/PycharmProjects/cora-derenderer/test_data/00002.depth.npy',
#             '/Users/fpk/PycharmProjects/cora-derenderer/test_data/00002.seg.npy')
# segment_depth_map('/Users/fpk/PycharmProjects/cora-derenderer/test_data/00002.depth.npy',
#                   '/Users/fpk/PycharmProjects/cora-derenderer/test_data/00002.seg.npy')
# This is how it would work remotely:
# plot_arrays('/mcs-cora/cora-derenderer/data/video_02758/00002.depth.npy',
#             '/mcs-cora/cora-derenderer/data/video_02758/00002.seg.npy')
# print(extract_shapes('/mcs-cora/cora-derenderer/data'))
