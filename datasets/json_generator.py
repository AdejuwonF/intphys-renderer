import json
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import numpy as np
from detectron2.data import DatasetCatalog

from datasets.adept import adept_to_shapes_world_object, adept_to_shapes_world_camera, adept_to_adept_object, \
    adept_to_adept_camera
from datasets.intphys import intphys_to_shapes_world_object, intphys_to_shapes_world_camera, \
    intphys_to_adept_object, intphys_to_adept_camera, match_current_to_prev_anns

from datasets.utils import get_dataset_name_and_json, frames2videos
from utils.misc import filter_dataset, CodeTimer, l2_distance

_DUMMY_CAMERA =  {"camera_eye_pose": {"x": 0,
                                      "y": 0,
                                      "z": 0,
                                      "yaw_radians": 0,
                                      "pitch_radians": 0,
                                      "roll_radians": 0 },
                  "fov_degrees": 90, #or should it be 60?
                  "aspect": 1,
                  "near_val": 0.01,
                  "far_val": 100.0 }

_DUMMY_FRAME = {"objects": [],
                "contact_edges": [],
                "suggested_view": {
                    "camera": _DUMMY_CAMERA,
                    "lighting": {
                        "light_source_pos": [1, 2, 3],
                        "ambient_coeff": 0.9 } } }

PROCESS_OBJECT_MAP = {("intphys","shapesworld"): intphys_to_shapes_world_object,
                      ("intphys","adept"): intphys_to_adept_object,
                      ("adept", "shapesworld"): adept_to_shapes_world_object,
                      ("adept", "adept"): adept_to_adept_object}

PROCESS_CAMERA_MAP = {("intphys","shapesworld"): intphys_to_shapes_world_camera,
                      ("intphys","adept"): intphys_to_adept_camera,
                      ("adept","shapesworld"):adept_to_shapes_world_camera,
                      ("adept", "adept"):adept_to_adept_camera}

MATCH_COST_MAP = {"intphys": l2_distance,
                  "adept": None}

def get_jsons_directory(data_cfg, target_physics, attributes_key, dataset_name):
    return os.path.join(data_cfg.BASE_DIRECTORY,
                        target_physics + "_jsons",
                        attributes_key,
                        dataset_name)

class JsonGenerator:
    def __init__(self, data_cfg, split, target_physics, attributes_key, vel_data_assoc):
        self.timer = CodeTimer("building jsons for {}_{} with target  {}".format(data_cfg.BASE_NAME,
                                                                                 split,
                                                                                 target_physics))
        self.process_object = PROCESS_OBJECT_MAP[(data_cfg.BASE_NAME,
                                                  target_physics)]
        self.process_camera = PROCESS_CAMERA_MAP[(data_cfg.BASE_NAME,
                                                  target_physics)]
        self.get_match_cost = MATCH_COST_MAP[data_cfg.BASE_NAME]

        self.requires_vel = vel_data_assoc != "None"
        self.vel_data_assoc = vel_data_assoc
        self.target_physics = target_physics
        self.attributes_key = attributes_key

        self.build_physics_jsons(data_cfg, split)


    def process_frame(self,data_cfg, prev_video_dict,video_dict):
        # process_object = PROCESS_OBJECT_MAP[data_cfg.BASE_NAME]
        prev_video_anns = self.match_annotations(prev_video_dict, video_dict)
        objects = [self.process_object(attributes=an[self.attributes_key],
                                       prev_attributes=prev_an[self.attributes_key] if prev_an is not None
                                                       else None,
                                       object_id=an["object_id"],
                                       segmentation=an["segmentation"])
                   for prev_an, an in zip(prev_video_anns,video_dict["annotations"])]
        objects = [obj for obj in objects if obj is not None]

        camera = self.process_camera(video_dict["camera"])

        state_scene = deepcopy(_DUMMY_FRAME)
        state_scene["objects"] = objects
        state_scene["suggested_view"]["camera"] = camera
        return state_scene


    def video2json(self,data_cfg, video_dict, out_dir, vid_num):
        # init,end = data_cfg.SHAPESWORLD_JSON.FRAMES_RANGE_PER_VIDEO
        frame_range = sorted(list(video_dict.keys()))
        assert np.unique(frame_range).size == np.arange(frame_range[0],
                                                        frame_range[-1]+1).size
        if not self.requires_vel:
            video_dict[frame_range[0]-1] = None
        else:
            frame_range = frame_range[1:]

        scene_states = [self.process_frame(data_cfg,video_dict[f-1],video_dict[f])
                        for f in frame_range]

        # video_name = 'video_' + str(vid_num).zfill(5) + ".json"
        video_name = os.path.basename(video_dict[frame_range[0]]["original_video"]) + ".json"
        out_path = os.path.join(out_dir, video_name)

        with open(out_path, "w") as f:
            json.dump({"scene_states": scene_states,
                       "debug":{"is_possible": video_dict[frame_range[0]]["is_possible"],
                                "original_video": video_dict[frame_range[0]]["original_video"]}},
                      f, indent=4)

    def build_physics_jsons(self,data_cfg, split):
        dataset_name, standard_format_json_file = get_dataset_name_and_json(data_cfg, split)
        dataset = DatasetCatalog.get(dataset_name)
        required_fields_values = {self.attributes_key:{"visible":1}}

        _, dataset = filter_dataset(dataset, required_fields_values=required_fields_values)
        videos_dicts = frames2videos(dataset)
        out_dir = get_jsons_directory(data_cfg,
                                      self.target_physics,
                                      self.attributes_key,
                                      dataset_name)
        os.makedirs(out_dir,exist_ok=True)

        worker_args = [(data_cfg, vid_dict, out_dir, vid_num)
                       for vid_num, vid_dict in videos_dicts.items()]

        if data_cfg.DEBUG:
            [self.video2json(*w) for w in worker_args]
        else:
            with Pool(int(cpu_count())) as p:
                p.starmap(self.video2json, worker_args)
        self.timer.done()

    def match_annotations(self, prev_video_dict, video_dict):
        if not self.requires_vel:
            return [None] * len(video_dict["annotations"])
        prev_anns_matched = []
        if self.vel_data_assoc == "ground_truth":
            #ground truth should have proper object ids for  data association
            prev_map = {prev_an["object_id"]: prev_an
                        for prev_an in prev_video_dict["annotations"]}
        elif self.vel_data_assoc == "heuristic":
            prev_map = match_current_to_prev_anns(video_dict["annotations"],
                                                  prev_video_dict["annotations"],
                                                  self.attributes_key)
        else:
            raise NotImplementedError

        for an in video_dict["annotations"]:
            if an["object_id"] in prev_map:
                prev_anns_matched.append(prev_map[an["object_id"]])
            else:
                prev_anns_matched.append(None)
        return prev_anns_matched








