import os
from collections import defaultdict
from copy import deepcopy
import random
from typing import Dict
import json
import torch
import numpy as np
# from easydict import EasyDict - marked for deletion
from torch.utils.data import Dataset
import logging as log
from structure.attributes.shapes_world_attributes import compute_mask
from datasets.utils import find_bounding_box


class ScenegraphDataset(Dataset):
    def __init__(self, data_path: str, cfg, validation=False):
        # Note: These values can be determined with helper functions for a particular dataset, but are given here
        # TODO Put these values into net_defaults configuration file
        self.cfg = cfg
        self.validation = validation
        self.max_depth = float(20)
        self.max_no_objects = float(8)
        self.shape_map = {'box': cfg.MODEL.ATTRIBUTES.SHAPE_MAP.BOX,
                          'cylinder': cfg.MODEL.ATTRIBUTES.SHAPE_MAP.CYLINDER}

        self.maskable_terms = {'box': cfg.MODEL.ATTRIBUTES.MASKABLE_TERMS.BOX,
                               'cylinder': cfg.MODEL.ATTRIBUTES.MASKABLE_TERMS.CYLINDER}

        self.video_max = cfg.DATASETS.MAX_VIDEOS  # There should be 10,000 videos in current dataset, but we can only fit ~2000 in RAM

        # Prepare global map of objects with globally unique identifiers
        self.obj_next_guid = 0
        self.obj_guid_map = {}  # lookup table of GUID -> object

        self.data_path = data_path
        self._init_dataset()

        # data_map_size = sys.getsizeof(self.obj_guid_map)
        # data_map_size_pympler = asizeof.asizeof(self.obj_guid_map)
        # log.debug(f'Size of data map is: {data_map_size} (acc. to sys) and {data_map_size_pympler} (acc. to Pympler).')

        #add dictionaries for means and variances

    def _parse_scenegraph(self, scene_graph_path: str, continuous_items = None, shapes=None) -> Dict:
        result_dictionary = {}  # Contains mask_id: continuous_value_vector, categorical_value
        log.debug(f'Scene graph path: {scene_graph_path}')
        with open(scene_graph_path, 'r') as scene_graph_file:
            scene = json.load(scene_graph_file)
        objects = scene.get('objects', [])
        log.debug(f'Number of objects: {len(objects)}')

        for obj in objects:
            name = obj.get('name')
            if name == 'floor':
                continue
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

            if continuous_items is not None:
                locs = locals()
                [continuous_items[term].append(eval(term, locs)) for term in self.cfg.MODEL.ATTRIBUTES.CONTINUOUS_TERMS]

            if shapes is not None:
                shapes.append(shape_type)

            # vector_continuous = torch.FloatTensor(
            #     [scale_x, scale_y, scale_z, x, y, z, yaw_radians, pitch_radians, roll_radians, radius, length])
            object_attributes = {'scale_x': torch.FloatTensor([scale_x]),
                                     'scale_y': torch.FloatTensor([scale_y]),
                                     'scale_z': torch.FloatTensor([scale_z]),
                                     'x': torch.FloatTensor([x]),
                                     'y': torch.FloatTensor([y]),
                                     'z': torch.FloatTensor([z]),
                                     'yaw_radians': torch.FloatTensor([yaw_radians]),
                                     'pitch_radians': torch.FloatTensor([pitch_radians]),
                                     'roll_radians': torch.FloatTensor([roll_radians]),
                                     'radius': torch.FloatTensor([radius]),
                                     'length': torch.FloatTensor([length]),
                                      'shape': shape_type}
            # TODO At some point might want to build vector like this to speed things up:
            # vector_continuous = torch.FloatTensor(
            #     [scale_x, scale_y, scale_z, x, y, z, yaw_radians, pitch_radians, roll_radians, radius, length])
            result_dictionary[mask_id] = object_attributes

        return result_dictionary

    def _init_dataset(self):
        # For all folders get all frames
        # For each frame parse JSON, get all objects and put them into consecutively indexed structure
        video_count = 0
        self.frames2arrays = {}
        continouos_items = defaultdict(list)
        shapes = []

        video_list = sorted(os.listdir(self.data_path))
        if self.cfg.DATASETS.VAL_VIDEOS > 0:
            if self.validation:
                video_list = video_list[:self.cfg.DATASETS.VAL_VIDEOS]
            else:
                video_list = video_list[self.cfg.DATASETS.VAL_VIDEOS:]

        if self.cfg.DEBUG and len(self.cfg.DEBUG_VIDEOS) >  0:
            video_list = self.cfg.DEBUG_VIDEOS #chosen videos
        elif not self.validation:
            random.shuffle(video_list)

        for video in video_list:
            if video_count >= self.video_max:
                break
            video_count += 1

            video_folder = os.path.join(self.data_path, video)
            if video_count % 100 == 0:
                log.info(f'Added {video_count} videos.')

            # If we do not know number of frames (seems to be 00001-00050 for current dataset):
            frames = sorted(set([frame.replace('.json', '').replace('.seg.npy', '').replace('.depth.npy', '')
                                for frame in os.listdir(video_folder)]))
            log.debug(f'Frames for video {video}:\n{frames}')

            for frame in frames:
                # Determine paths of scene graph JSON, depth map file and segmentation map file
                frame_scene_graph_path = os.path.join(video_folder, f'{frame}.json')
                frame_depth_map_path = os.path.join(video_folder, f'{frame}.depth.npy')
                frame_segmentation_map_path = os.path.join(video_folder, f'{frame}.seg.npy')
                log.debug(f'Scenegraph path: {frame_scene_graph_path}')

                # Extract scene graph attributes, normalized depth map and normalized segmentation map
                scenegraph_attributes = self._parse_scenegraph(frame_scene_graph_path, continouos_items, shapes)
                # segmentation_array_normalized = torch.from_numpy((segmentation_array + 1) / self.max_no_objects)
                self.frames2arrays[(video,frame)] = {'depth': frame_depth_map_path, 'masks': frame_segmentation_map_path}
                segmentation_map = np.load(frame_segmentation_map_path)
                # For each object return vector of continuous values, categorical value (shape) and 3 maps
                for segmentation_id in scenegraph_attributes.keys():
                    attributes = scenegraph_attributes[segmentation_id]  # 0: continuous, 1: categorical
                    area = (segmentation_map == segmentation_id).sum()
                    if area < self.cfg.DATASETS.MIN_OBJ_AREA:
                        continue

                    data = {'obj_id': segmentation_id ,'basename': 'na', 'index': self.obj_next_guid,
                            'attributes': attributes, 'frame_guid': (video, frame)}

                    self.obj_guid_map[self.obj_next_guid] = data
                    self.obj_next_guid += 1

        #compute means and std deviations for continuous terms
        self.means = defaultdict(lambda: float(0))
        self.std_deviations = defaultdict(lambda: float(1))
        shapes = torch.LongTensor(shapes)
        for term in self.cfg.MODEL.ATTRIBUTES.CONTINUOUS_TERMS:
            mask = compute_mask(term, shapes, self.maskable_terms, self.shape_map)
            data = torch.FloatTensor(continouos_items[term]).view(-1,1)
            data = data[mask!=0]
            self.means[term] = data.mean().item()
            self.std_deviations[term] = data.std().item()

    def __len__(self):
        return len(self.obj_guid_map)

    def __getitem__(self, idx):
        data = deepcopy(self.obj_guid_map[idx])
        arrays = self.frames2arrays[data["frame_guid"]]
        depth_array = np.load(arrays["depth"])
        if (depth_array==0).any():
            raise Exception("the depth array has a 0 so 1/d will not be valid")
        segmentation_array = np.load(arrays["masks"])
        box = find_bounding_box(segmentation_array, data["obj_id"])
        depth_array_masked_normalized = np.zeros_like(depth_array)
        depth_array_masked_normalized[box.min_y:box.max_y+1, box.min_x:box.max_x+1] = \
            1.0/(1.0+depth_array[box.min_y:box.max_y+1, box.min_x:box.max_x+1])

        # img_tuple = torch.stack([depth_array_normalized, depth_array_masked_normalized], dim=0)
        data["img_tuple"] = torch.FloatTensor(depth_array_masked_normalized).unsqueeze(0)
        data["depth_array"] = depth_array
        data["index"] = idx
        return data