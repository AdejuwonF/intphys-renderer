

from typing import Dict

from derender_helper import DerenderHelper
from torch.utils.data import Dataset

import json
import numpy as np
import os
import torch

import logging as log
log.basicConfig(level=log.DEBUG)


class ScenegraphDataset(Dataset):
    def __init__(self, data_path: str):
        # Note: These values can be determined with helper functions for a particular dataset, but are given here
        # TODO Put these values into net_defaults configuration file
        self.max_depth = float(20)
        self.max_no_objects = float(8)
        self.shape_map = {'box': 0, 'cylinder': 1}

        # Prepare global map of objects with globally unique identifiers
        self.obj_next_guid = 0
        self.obj_guid_map = {}  # lookup table of GUID -> object

        self.data_path = data_path
        self._init_dataset()

    def _parse_scenegraph(self, scene_graph_path: str) -> Dict:
        result_dictionary = {}  # Contains mask_id: continuous_value_vector, categorical_value
        log.info(f'Scene graph path: {scene_graph_path}')
        with open(scene_graph_path, 'r') as scene_graph_file:
            scene = json.load(scene_graph_file)
        objects = scene.get('objects', [])
        log.info(f'Number of objects: {len(objects)}')

        for obj in objects:
            name = obj.get('name')
            log.debug(f'Parsing object {name}.')

            # Intermediary values
            temp_shape = obj.get('shape', {})
            temp_shape_params = temp_shape.get('shape_params', {})
            temp_pose6d = obj.get('pose6d', {})
            temp_debug = obj.get('debug', {})

            # Values of interest for result dictionary
            shape_type = self.shape_map[temp_shape.get('shape_type')]  # shape_map maps categorical string to number
            x = temp_pose6d.get('x')
            y = temp_pose6d.get('y')
            z = temp_pose6d.get('z')
            yaw_radians = temp_pose6d.get('yaw_radians')
            pitch_radians = temp_pose6d.get('pitch_radians')
            roll_radians = temp_pose6d.get('roll_radians')
            mask_id = temp_debug.get('ground_truth_mask_id')

            # FIXME Why does PyCharm mark the following line yellow?
            # TODO Add 3 values to this vector so that box and ctylinder values have unique places:
            #

            if shape_type == 0:  # box
                scale_x = temp_shape_params.get('scale_x')
                scale_y = temp_shape_params.get('scale_y')
                scale_z = temp_shape_params.get('scale_z')
                radius = -1
                length = -1
            elif shape_type == 1:  # cylinder
                scale_x = -1
                scale_y = -1
                scale_z = -1
                radius = temp_shape_params.get('radius')
                length = temp_shape_params.get('length')

            vector_continuous = torch.FloatTensor(
                [scale_x, scale_y, scale_z, x, y, z, yaw_radians, pitch_radians, roll_radians, radius, length])
            result_dictionary[mask_id] = vector_continuous, shape_type

        return result_dictionary

    def _init_dataset(self):
        # For all folders get all frames
        # For each frame parse JSON, get all objects and put them into consecutively indexed structure
        for video in os.listdir(self.data_path):
            video_folder = os.path.join(self.data_path, video)

            # If we do not know number of frames (seems to be 00001-00050 for current dataset):
            frames = sorted(set([frame.replace('.json', '').replace('.seg.npy', '').replace('.depth.npy', '')
                                for frame in os.listdir(video_folder)]))
            log.debug(f'Frames for video {video}:\n{frames}')

            for frame in frames:
                # Determine paths of scene graph JSON, depth map file and segmentation map file
                frame_scene_graph_path = os.path.join(video_folder, f'{frame}.json')
                frame_depth_map_path = os.path.join(video_folder, f'{frame}.depth.npy')
                frame_segmentation_map_path = os.path.join(video_folder, f'{frame}.seg.npy')
                log.info(f'Scenegraph path: {frame_scene_graph_path}')

                # Extract scene graph attributes, normalized depth map and normalized segmentation map
                scenegraph_attributes = self._parse_scenegraph(frame_scene_graph_path)
                depth_array = np.load(frame_depth_map_path)
                depth_array_normalized = torch.from_numpy(depth_array / self.max_depth)
                segmentation_array = np.load(frame_segmentation_map_path)
                segmentation_array_normalized = torch.from_numpy((segmentation_array + 1) / self.max_no_objects)

                # For each object return vector of continuous values, categorical value (shape) and 3 maps
                for segmentation_id in scenegraph_attributes.keys():
                    attributes = scenegraph_attributes[segmentation_id]  # 0: continuous, 1: categorical

                    # Cut out depth map for particular object ID
                    depth_array_masked = np.where((segmentation_array == segmentation_id), depth_array, 0)
                    depth_array_masked_normalized = torch.from_numpy(depth_array_masked / self.max_depth)
                    # Stack three maps together
                    img_tuple = torch.stack(
                        (depth_array_normalized, segmentation_array_normalized, depth_array_masked_normalized), dim=0)
                    log.info(f'Image tuple size: {img_tuple.size()}')
                    # Could add batch dimension in front in other context: img_tuple = img_tuple.unsqueeze(0)

                    # Form data dictionary
                    # TODO Add global frame_id -> Tuple(video_id,frame_id)
                    data = {'img_tuple': img_tuple, 'basename': 'na', 'index': self.obj_next_guid,
                            'attributes': attributes[0], 'cat_attributes': attributes[1]}

                    self.obj_guid_map[self.obj_next_guid] = data
                    self.obj_next_guid += 1

    def __len__(self):
        return len(self.obj_guid_map)

    def __getitem__(self, idx):
        return self.obj_guid_map[idx]


if __name__ == '__main__':
    dataset = ScenegraphDataset('../test_data/mcs/')
    print(dataset[50])
    helper = DerenderHelper()
    helper.analyze_data_dictionary(dataset[50])

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    next_loaded_item = next(iter(dataloader))
    print(type(next_loaded_item))
    print(next_loaded_item)
    print(len(next_loaded_item))

