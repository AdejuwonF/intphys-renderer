import glob
import os
import random
from collections import defaultdict
from copy import deepcopy
from itertools import repeat, chain

import pycocotools
from detectron2.structures import BoxMode
from easydict import EasyDict

from datasets.intphys import deg2rad
from datasets.utils import find_bounding_box, find_bbox_area
from utils.io import write_serialized, read_serialized
from utils.misc import CodeTimer, l2_distance
from multiprocessing import Pool, cpu_count

COLOR_MAP = {k: i for i, k in enumerate(["red", "blue", "green", "brown",
                                         "purple", "cyan", "yellow"])}
COLOR_MAP_INV = {v:k for k,v in COLOR_MAP.items()}
TYPE_MAP = {"Occluder": 0,
            "Sphere": 1}
TYPE_MAP_INV = {v: k for k, v in TYPE_MAP.items()}

CATEGORICAL_TERMS = ["existance", "visible", "type", "color"]
CONTINUOUS_TERMS = list(chain.from_iterable([[pref + "_" + suf for suf in "xyz"]
                                             for pref in ["location", "scale", "velocity"]]))
CONTINUOUS_TERMS += ["rotation_" + el for el in ["roll", "pitch", "yaw"]]
QUANTIZED_TERMS = []
TERMS = CATEGORICAL_TERMS + CONTINUOUS_TERMS

ROTATION_TERMS = []

POSITIVE_TERMS = ["scale_" + n for n in "xyz"]
OBJECT_ID_MAP = {k: k for k in range(10)}

VALID_MAP = {k: {"type": ("Occluder",)}
             for k in ["rotation_roll", "rotation_pitch", "rotation_yaw"]}

#object ID shouldn't be learnable
VALID_MAP.update({"object_id": {"type": ("invalid",)}})

_DUMMY_ATTRIBUTES = {k: 0 for k in TERMS}
_DUMMY_ATTRIBUTES["existance"] = 1
_DUMMY_OBJECT = {k: 0 for k in ["object_id", "is_crowd", "category_id"]}
_DUMMY_OBJECT["attributes"] = _DUMMY_ATTRIBUTES

CASE_GROUPS = {
    "create": {"key": "disappear",
               "impossible": ('2',),
               "possible": ('0', '3', '4')
               },
    "vanish": {"key": "disappear_fixed",
               "impossible": ('1',),
               "possible": ('0', '3', '4')
               },
    "short-overturn": {"key": "overturn",
                       "impossible": ('0',),
                       "possible": ('1',)
                       },
    "long-overturn": {"key": "overturn",
                      "impossible": ('3',),
                      "possible": ('2',)
                      },
    "visible-discontinuous": {"key": "discontinuous",
                              "impossible": ('2',),
                              "possible": ('3', '5', '4')
                              },
    "invisible-discontinuous": {"key": "discontinuous",
                                "impossible": ('1',),
                                "possible": ('0', '5', '4')
                                },
    "delay": {"key": "delay",
              "impossible": ('1',),
              "possible": ('0',)
              },
    "block": {"key": "block",
              "impossible": ('1',),
              "possible": ('0',)
              },
}

def get_video_folders(cfg, split):
    base_folder = split.split("_")[1]
    base_folder = os.path.join(cfg.DATA_LOCATION, base_folder)
    video_folders = sorted(os.listdir(base_folder))
    if split in ["_val", "_train"]:
        return [{"video_folder": os.path.join(base_folder, v),
                 "is_possible": True}
                for v in video_folders]
    elif "human" in split:
        annomaly_type = split.split("_")[2]
        parser = CASE_GROUPS[annomaly_type]
        dir_key = parser["key"]
        case_numbers = list(parser["impossible"]) + list(parser["possible"])
        if dir_key != "disappear":
            video_folders = [os.path.join(base_folder, v)
                             for v in video_folders if dir_key in v
                             and v[-1] in case_numbers]
        elif dir_key == "disappear":
            video_folders = [os.path.join(base_folder, v)
                             for v in video_folders if dir_key in v
                             and v[-1] in case_numbers
                             and "fixed" not in v]

        return [{"video_folder": v, "is_possible": v[-1] not in parser["impossible"]}
                for v in video_folders]

def correct_img_path(incorrect_path, video_folder):
    img_path = incorrect_path.split("/")[-2:]
    img_path = os.path.join(video_folder, *img_path)
    return img_path

def calc_cost(obj,target):
    if not all([obj["adept_shape"] == target["adept_shape"],
                obj["attributes"]["color"] == target["attributes"]["color"]]):
        return  float('Inf')

    return l2_distance(obj["attributes"],target["attributes"])

def find_obj_id(obj, prev_objects):
    min_cost = float('Inf')
    match = None
    for k,o in prev_objects["map"].items():
        cost = calc_cost(obj, o)
        #only  objects that are real close can be matched
        if cost < 1 and cost < min_cost:
            min_cost = cost
            match = k

    return match

def update_prev_objects(prev_objects, annotations):
    anns_dict = {an["object_id"]: an for an in annotations}
    for k in list(prev_objects["map"].keys()):
        if k not in anns_dict:
            del prev_objects["map"][k]

def process_object(raw_object, min_area, prev_objects):
    obj = deepcopy(_DUMMY_OBJECT)
    attributes = obj["attributes"]
    for pref in ["location", "scale", "velocity"]:
        [attributes.__setitem__(pref + "_" + n, v) for n, v in zip("xyz", raw_object[pref])]
    if raw_object["type"] == "Occluder":
        attributes["type"] = TYPE_MAP[raw_object["type"]]
        [attributes.__setitem__("rotation_" + n, v) for n, v in zip(["roll", "pitch", "yaw"], raw_object["rotation"])]
    else:
        attributes["type"] = TYPE_MAP["Sphere"]
    attributes["color"] = COLOR_MAP[raw_object["color"]]

    mask = deepcopy(raw_object["mask"])
    if mask["counts"] != "PPf4":
        mask["counts"] = mask["counts"].encode("ascii")
        mask = pycocotools.mask.decode(mask)
        bbox = find_bounding_box(mask, 1)
        bounding_box = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]
        bounding_box = [int(el) for el in bounding_box]
        if find_bbox_area(bounding_box) > min_area:
            attributes["visible"] = 1
            bbox = find_bounding_box(mask, 1)
            bounding_box = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]
            bounding_box = [int(el) for el in bounding_box]
            obj.update({"bbox": bounding_box,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": deepcopy(raw_object["mask"])})

    obj["adept_shape"] = raw_object["name"].split("_")[0]

    obj["object_id"] = prev_objects["map"][raw_object["name"]]

    prev_objects["map"][obj["object_id"]] = obj
    return obj


def process_frame(scene_states, camera, frame_num, video_number, min_area, video_folder, prev_objects, is_possible):
    img_path, img_2, img_4 = [correct_img_path(scene_states[f]["image_path"], video_folder)
                              for f in [frame_num, frame_num - 2, frame_num - 4]]

    annotations = [process_object(o, min_area, prev_objects)
                   for o in scene_states[frame_num]["objects"]]
    annotations = [an for an in  annotations if an is not None]

    # prev_objects["initialized"] = True
    # update_prev_objects(prev_objects, annotations)

    object_ids = [an["object_id"] for an in annotations]
    assert len(set(object_ids)) == len(object_ids)

    out_frame = {"file_name": img_path,
                 "prev_images": (img_4, img_2),
                 "image_id": video_number * 500 + frame_num,
                 "height": 320,
                 "width": 480,
                 "camera": camera,
                 "original_video": "adept_data" + video_folder.split("adept_data")[1],
                 "annotations": annotations,
                 "is_possible": is_possible}

    return out_frame


def process_video(video, cfg, video_number):
    video_folder = video["video_folder"]
    scene_file = glob.glob(os.path.join(video_folder, "*.yaml"))[0]

    meta_data = read_serialized(scene_file)
    camera = meta_data["camera"]
    camera["camera_theta"] *= -1
    camera["camera_phi"] += 90

    min_area = cfg.MIN_AREA

    prev_objects = {"initialized": False,
                    "map":defaultdict(lambda: len(prev_objects["map"]))}
    frames_dicts = [process_frame(meta_data["scene"], camera, f, video_number,
                                  min_area, video_folder, prev_objects, video["is_possible"])
                    for f in range(4, len(meta_data["scene"]))]

    #leave only objects that never lost continuity
    # for fr in frames_dicts:
    #     fr["annotations"] = [an for an in fr["annotations"] if an["object_id"] in prev_objects["map"]]

    if len(prev_objects["map"])>10:
        print("this video  has more than 10  objects: " + video["video_folder"])
    return frames_dicts


def adept_to_detectron(cfg, split, outfile):
    timer = CodeTimer("started processing videos for {}".format(split))
    videos = get_video_folders(cfg, split)
    # if "_val" in split:
    #     # videos = random.choices(videos, k=cfg.VAL_VIDEOS)
    #     videos = videos[:cfg.VAL_VIDEOS]

    if cfg.DEBUG:
        if len(cfg.DEBUG_VIDEOS) > 0:
            videos = [v for v in videos if v["video_folder"] in cfg.DEBUG_VIDEOS]
        dicts = [process_video(v, cfg, i) for i, v in enumerate(videos)]
    else:
        with Pool(int(cpu_count())) as p:
            dicts = p.starmap(process_video, zip(videos, repeat(cfg), range(len(videos))))

    dicts = list(chain.from_iterable(dicts))

    if "_val" in split:
        dicts = random.choices(dicts, k=cfg.VAL_FRAMES)

    write_serialized(dicts, outfile)
    timer.done()


def adept_to_shapes_world_object(**kwargs):
    attributes = kwargs["attributes"]
    obj_type = TYPE_MAP_INV[attributes["type"]].lower()

    x, y, z, roll, pitch, yaw, scale_x, scale_y, scale_z = \
        [attributes[k]
         for k in ["location_x", "location_y", "location_z",
                   "rotation_roll", "rotation_pitch", "rotation_yaw",
                   "scale_x", "scale_y", "scale_z"]]

    if obj_type == "sphere":
        roll, pitch, yaw = [0.0] * 3

    return {"name": "block{}".format(kwargs['object_id']),
            "shape": {
                "shape_type": "box",
                "shape_params": {
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "scale_z": scale_z
                }
            },
            "pose6d": {
                "x": x,
                "y": y,
                "z": z,
                "yaw_radians": yaw,
                "pitch_radians": pitch,
                "roll_radians": roll
            }
            }

def adept_to_shapes_world_camera(camera_terms):
    camera = {"camera_eye_pose": {"x": 0,
                                         "y": 0,
                                         "z": 0,
                                         "yaw_radians": 0,
                                         "pitch_radians": 0,
                                         "roll_radians": 0},
                     "fov_degrees": 32,
                     "aspect": 480.0 / 320.0,
                     "near_val": 0.01,
                     "far_val": 100.0}

    x, y, z = camera_terms["camera_look_at"]
    pitch_radians = deg2rad(camera_terms["camera_theta"])
    yaw_radians = deg2rad(camera_terms["camera_phi"])
    roll_radians = 0.0
    distance = camera_terms["camera_rho"]
    l = locals()
    camera_eye = {k:l[k] for k in ["x","y", "z",
                                   "pitch_radians",
                                   "yaw_radians",
                                   "roll_radians",
                                   "distance"]}
    camera["camera_eye_pose"] = camera_eye
    return camera

def adept_to_adept_object(**kwargs):
    o = EasyDict(kwargs["attributes"])
    obj_type = TYPE_MAP_INV[o["type"]].lower()

    obj = EasyDict({"type": obj_type,
                    "location": [o["location_"+el] for el in "xyz"],
                    "rotation": [o["rotation_"+el] for el in ['roll', 'pitch', 'yaw']],
                    "scale": [o.scale_x, o.scale_y, o.scale_z],
                    "angular_velocity": [0, 0, 0],
                    "color": COLOR_MAP_INV[o.color],
                    "mask": kwargs["segmentation"],
                    "velocity": [o["velocity_"+el] for el in "xyz"]})

    if obj_type == "sphere":
        obj.rotation = [0, 0, 0]
        obj.type = "object"

    obj.mask["counts"] = obj.mask["counts"].decode("ascii")

    return obj

def adept_to_adept_camera(camera_terms):
    camera = {"camera_eye_pose": {"camera_look_at": camera_terms["camera_look_at"],
                  "camera_phi": camera_terms["camera_phi"],
                  "camera_theta": camera_terms["camera_theta"],
                  "camera_rho": camera_terms["camera_rho"]
                  },
                 "fov_degrees": 32,  # or should it be 60?
                 "aspect": 480.0 / 320.0,
                 "near_val": 0.01,
                 "far_val": 100.0,
                 "width": 480,
                 "height": 320, }
    return camera

# def group_key(d):
#     return tuple(d["original_video"].split("_")[:-1])

def parse_case_name(original_path):
    name = original_path.split('/')[-1]
    split_name = name.split("_")
    key = split_name[1] if "fixed" not in split_name else "disappear_fixed"
    shape = "_".join(name.split(key)[1].split("_")[1:-1])
    index = split_name[-1]
    return key, shape, index
    # split_name = name.split("_")
    # index = split_name[-1][0]
    # anomaly_key = split_name[1] if "fixed" not in split_name else "disappear_fixed"
    # anomaly_key_index = anomaly_key + f'_{index}'
    # if anomaly_key_index not in KEYS_TO_ANNOMALIES:
    #     print(name, " is not in matched cases")
    #     return None, None, None, False
    # anomaly_type = KEYS_TO_ANNOMALIES[anomaly_key_index]
    # case_group_id = "_".join(name.split(anomaly_key)[1].split("_")[1:-1])
    # case_key = (anomaly_type,anomaly_key,case_group_id)
    # possible = index in CASE_GROUPS[anomaly_type]["possible"]
    # # return {"anomaly_type": anomaly_type,
    # #         "case_key":case_key,
    # #         "possible": possible}
    # return anomaly_type, case_key, possible, True

def adept_group_by_control_surprise(dataset_score_dicts):
    for d in dataset_score_dicts:
        key, shape, index = parse_case_name(d['original_video'])
        d.update({'key':key, 'shape':shape, 'index':index})
    grouped_dataset = {}
    for g in CASE_GROUPS:
        grouped_dataset[g] = defaultdict(list)
        key = CASE_GROUPS[g]['key']
        indices = CASE_GROUPS[g]['impossible'] + CASE_GROUPS[g]['possible']
        for d in dataset_score_dicts:
            if d['key'] == key and d['index'] in indices:
                grouped_dataset[g][d['shape']].append(d)

    return grouped_dataset
    # control_surprise_groups = defaultdict(list)
    # [control_surprise_groups[group_key(d)].append(d)
    #  for d in dataset_score_dicts]
    # return control_surprise_groups
