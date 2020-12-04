import itertools
import os
from collections import defaultdict
from copy import deepcopy
from multiprocessing import cpu_count, Pool
import pycocotools
from detectron2.structures import BoxMode
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment

from datasets.utils import find_bounding_box, find_bbox_area
from utils.io import read_serialized, write_serialized
from utils.misc import CodeTimer, quantized_float2idx, quantized_idx2float, l2_distance
from PIL import Image
import numpy as np

TYPE_MAP = {"object": 0,
            "occluder": 1,
            "floor": 2,
            "walls": 3,
            "sky": 4,
            "invalid": 5}  # invalid is used to not learn (via invalidation) some attributes

TYPE_MAP_INV = {v: k for k, v in TYPE_MAP.items()}

SHAPE_MAP = {'Cube': 0,
             'Cone': 1,
             'Sphere': 2}
SHAPE_MAP_INV = {v: k for k, v in SHAPE_MAP.items()}

OBJECT_ID_MAP = {'object_1': 0,
                 'sky': 1,
                 'walls': 2,
                 'floor': 3,
                 'occluder_2': 4,
                 'occluder_1': 5,
                 'object_2': 6,
                 'object_3': 7,
                 'occluder_3': 8,
                 'occluder_4': 9, }
# 'object_4':10,
# 'object_5':11,
# 'object_6':12,
# 'object_7':13,
# 'object_8':14,
# 'object_9':15,
# 'object_10':16}
INV_OBJECT_ID_MAP = {v: k for k, v in OBJECT_ID_MAP.items()}

ROTATION_YAW_ARRAY = np.linspace(-np.pi, np.pi, num=50)

_TERMS = ["type", "location_x", "location_y", "location_z",
          "rotation_roll", "rotation_pitch", "rotation_yaw",
          "scale_x", "scale_y", "scale_z", "shape", "mass",
          "visible"]

CAMERA_TERMS = ["cam_location_x", "cam_location_y", "cam_location_z",
                "cam_rotation_roll", "cam_rotation_pitch", "cam_rotation_yaw"]

_TERMS = _TERMS  # + CAMERA_TERMS

######## for specific types of losses ###########
CATEGORICAL_TERMS = ["existance", "visible", "type", "shape", "object_id"]
CONTINUOUS_TERMS = ["location_x", "location_y", "location_z",
                    "rotation_roll", "rotation_pitch",
                    "scale_x", "scale_y", "scale_z", "mass"]
QUANTIZED_TERMS = ["rotation_yaw"]

ROTATION_TERMS = ["rotation_roll", "rotation_pitch"]

POSITIVE_TERMS = {"scale_x", "scale_y", "scale_z"}

VALID_MAP = {k: {"type": ("object", "occluder", "floor")}
             for k in ["location_x", "location_y", "location_z",
                       "scale_x", "scale_y", "scale_z"]}

VALID_MAP.update({k: {"type": ("occluder",)}
                  for k in ["rotation_roll", "rotation_pitch", "rotation_yaw"]})

VALID_MAP.update({k: {"type": ("object",)} for k in ["shape", "mass"]})
VALID_MAP.update({"object_id": {"type": ("invalid",)}})

_DUMMY_ATTRIBUTES = {k: 0 for k in _TERMS}
_DUMMY_ATTRIBUTES["existance"] = 1

_DUMMY_CAMERA = {'location': {'x': 0.0, 'y': 0.0, 'z': 2.0},
                 'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                 'field_of_view': 90.0,
                 'aspect_ratio': 1.0,
                 'projection_mode': 0}


def get_type(obj_name):
    return TYPE_MAP[obj_name.split("_")[0]]


def deg2rad(x):
    return float(x * np.pi / 180)


def rad2deg(x):
    return float(x * 180 / np.pi)


def normalize_dimensions(el):
    el = deepcopy(el)
    el["location"] = {k: v / 100.0 for k, v in el["location"].items()}
    el["rotation"] = {k: deg2rad(v) for k, v in el["rotation"].items()}
    return el


def relative_to_camera(attributes, camera):
    yaw = camera["rotation"]["yaw"]
    pitch = camera["rotation"]["pitch"]
    p_sin, y_sin = map(np.sin, [yaw, pitch])
    p_cos, y_cos = map(np.cos, [yaw, pitch])

    c_loc = camera["location"]
    a_loc = attributes["location"]

    x, y, z = [a_loc[k] - c_loc[k] for k in ["x", "y", "z"]]
    attributes["location"]["x"] = y_cos * p_cos * x + y_sin * p_cos * y + p_sin * z
    attributes["location"]["y"] = -y_sin * x + y_cos * y
    attributes["location"]["z"] = -y_cos * p_sin * x - y_sin * p_sin * y + p_cos * z

    return attributes


def get_closest_discrete_value(value, discrete_array, map):
    closest_idx = np.abs(discrete_array - value).argmin()
    return map[discrete_array[closest_idx]]


def process_attributes(intphys_attr):
    attributes = deepcopy(_DUMMY_ATTRIBUTES)

    intphys_attr = normalize_dimensions(intphys_attr)
    # intphys_attr = relative_to_camera(intphys_attr, camera)

    ###### set locations ###########
    [attributes.__setitem__("location_" + k, v) for k, v in intphys_attr["location"].items()]

    ####### set rotations #########
    [attributes.__setitem__("rotation_" + k, v) for k, v in intphys_attr["rotation"].items()]
    # attributes["rotation_yaw"] += np.pi if attributes["rotation_yaw"] < 0 else 0.0
    attributes["rotation_yaw"] = quantized_float2idx(attributes["rotation_yaw"], ROTATION_YAW_ARRAY)

    ####### set scales #########
    [attributes.__setitem__("scale_" + k, v) for k, v in intphys_attr["scale"].items()]

    # ####### set camera #########
    # [attributes.__setitem__("cam_location_" + k, v) for k, v in camera["location"].items()]
    # [attributes.__setitem__("cam_rotation_" + k, v) for k, v in camera["rotation"].items()]

    ####### only objects attributes ###########
    if "shape" in intphys_attr:
        attributes["shape"] = SHAPE_MAP[intphys_attr["shape"]]
        attributes["mass"] = intphys_attr["mass"]

    return attributes


def process_object(obj_name, segm, frame_status, min_area, floor_attr, no_status):
    # category_id and iscrowd are necessary for the detector pipeline to run  smoothly
    an = {"category_id": 0,
          "iscrowd": 0}

    if obj_name in frame_status:
        attributes = process_attributes(frame_status[obj_name])
    elif obj_name == "floor":
        attributes = process_attributes(floor_attr)
    else:
        attributes = deepcopy(_DUMMY_ATTRIBUTES)

    ######## On test there are no object names #############
    if not no_status:
        attributes["type"] = TYPE_MAP[obj_name.split("_")[0]]
        ###### set object_id #######val##
        attributes["object_id"] = OBJECT_ID_MAP[obj_name]
        an["object_id"] = OBJECT_ID_MAP[obj_name]
    else:
        an["object_id"] = int(obj_name.split("_")[-1])
        attributes["object_id"] = int(obj_name.split("_")[-1])

    if obj_name in frame_status["masks"]:
        obj_id = frame_status["masks"][obj_name]
    else:
        obj_id = None

    if obj_id in np.unique(segm):
        bbox = find_bounding_box(segm, obj_id)
        bounding_box = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]
        bounding_box = [int(el) for el in bounding_box]

        if find_bbox_area(bounding_box) > min_area:
            mask = segm == obj_id
            encoded_mask = pycocotools.mask.encode(np.asarray(mask, order="F"))
            encoded_mask["counts"] = encoded_mask["counts"].decode('ascii')
            an.update({"bbox": bounding_box,
                       "bbox_mode": BoxMode.XYXY_ABS,
                       "segmentation": encoded_mask})
            attributes["visible"] = 1

    an["attributes"] = attributes

    return an


def process_frame(video_folder, vid_number, frame_num, min_area, status, camera_terms):
    depth_map_file, segmentation_file = \
        map(lambda t: os.path.join(video_folder, t, "{}_{}.png".format(t, str(frame_num + 1).zfill(3))),
            ["depth", "masks"])

    segm = np.asarray(Image.open(segmentation_file))
    height, width = segm.shape

    if "frames" in status:
        no_status = False
        frame_status = status["frames"][frame_num]
        ##### always add invisible objects (including floor) but only add walls and sky when visible #################
        objects = list(set(list(frame_status["masks"].keys()) + ["floor"] +
                           [k for k in frame_status.keys() if k != "masks"]))
        floor = status["header"]["floor"]
    else:
        no_status = True
        frame_status = {"masks": {}}
        floor = None
        objects = []
        for i, obj_id in enumerate(np.unique(segm)):
            obj_name = "object_" + str(i + 1)
            objects.append(obj_name)
            frame_status["masks"][obj_name] = obj_id

    annotations = [process_object(obj_name, segm, frame_status,
                                  min_area, floor, no_status) for obj_name in objects]

    out_frame = {"file_name": depth_map_file,
                 "image_id": vid_number * 500 + frame_num,
                 "height": height,
                 "width": width,
                 "annotations": annotations,
                 "camera": camera_terms,
                 "original_video": "intphys" + video_folder.split("intphys")[1]}

    try:
        out_frame["is_possible"] = status["header"]["is_possible"]
    except KeyError:
        out_frame["is_possible"] = None

    return out_frame


def process_video(video_folder, vid_number, min_area):
    status_file = os.path.join(video_folder, "status.json")
    if os.path.exists(status_file):
        status = read_serialized(status_file)
    else:
        status = {}

    try:
        camera = normalize_dimensions(status["header"]["camera"])
    except KeyError:
        camera = _DUMMY_CAMERA

    ####### set camera #########
    camera_terms = {}
    [camera_terms.__setitem__("cam_location_" + k, v) for k, v in camera["location"].items()]
    [camera_terms.__setitem__("cam_rotation_" + k, v) for k, v in camera["rotation"].items()]

    frames_dicts = [process_frame(video_folder, vid_number, f, min_area, status, camera_terms) for f in range(100)]

    return frames_dicts


def intphys_to_detectron(cfg, split, outfile):
    timer = CodeTimer("started processing videos for {}".format(split))
    base_folder = split.split("_")[1:] if split != "_val" else ["train"]
    base_folder = os.path.join(cfg.DATA_LOCATION, *base_folder)
    video_folders = build_recursive_case_paths(base_folder, [])

    start_vid_num = 0
    if split == "_val":
        video_folders = video_folders[:cfg.VAL_VIDEOS]
    elif split == "_train":
        video_folders = video_folders[cfg.VAL_VIDEOS:]
        start_vid_num = cfg.VAL_VIDEOS

    worker_args = []
    for i, video in enumerate(video_folders, start=start_vid_num):
        worker_args.append((video, i, cfg.MIN_AREA))

    if cfg.DEBUG:
        dicts = [process_video(*w) for w in worker_args]
    else:
        with Pool(int(cpu_count())) as p:
            dicts = p.starmap(process_video, worker_args)

    dicts = list(itertools.chain.from_iterable(dicts))

    # print_mean_std_inv_depth(dicts)

    write_serialized(dicts, outfile)
    timer.done()


def build_recursive_case_paths(input_folder, cases):
    if "scene" not in os.listdir(input_folder):
        to_recurse = sorted(list(os.path.join(input_folder, sub_folder) for sub_folder in os.listdir(input_folder)))
        for new_folder in to_recurse:
            if os.path.isdir(new_folder):
                build_recursive_case_paths(new_folder, cases)
    else:
        cases.append(input_folder)
    return cases


############ utils to transform to shapesworld ################


# def right_handed_to_left_handed(el):
#     el = deepcopy(el)
#     el["location"] = {k: v / 100.0 for k, v in el["location"].items()}
#     # el["location"]["y"] *= -1.0
#     el["rotation"] = {k: deg2rad(v) for k, v in el["rotation"].items()}
#     # el["rotation"]["yaw"] *= -1.0
#     return el

def intphys_to_pybullet(attributes, OBJECT_SCALER, OCCLUDER_SCALER, occluder_center_fn):
    obj_type = TYPE_MAP_INV[attributes["type"]]
    if obj_type == "object":
        scaler = OBJECT_SCALER
    elif obj_type == "occluder":
        scaler = OCCLUDER_SCALER
    else:
        return {}

    shape = SHAPE_MAP_INV[attributes['shape']]
    scales = {k: attributes[k] * scaler[k]
              for k in ["scale_x", "scale_y", "scale_z"]}

    x, y, z, roll, pitch, yaw = [attributes[k] for k in ["location_x", "location_y", "location_z",
                                                         "rotation_roll", "rotation_pitch", "rotation_yaw"]]
    yaw = quantized_idx2float(yaw, ROTATION_YAW_ARRAY)

    # left handed to right handed
    y *= -1
    yaw *= -1

    if obj_type == "occluder":
        x, y, z = occluder_center_fn(x, y, z, roll, yaw, scales)
    vars = locals()
    return EasyDict({k: vars[k] for k in
                     ["obj_type", "x", "y", "z", 'roll', 'pitch', 'yaw', 'scales', 'shape']})


def occluder_center_intphys(x, y, z, roll, yaw, box_params):
    width = box_params["scale_x"] / 2
    height = box_params["scale_z"] / 2
    return [
        x + width * np.cos(yaw) + height * np.sin(roll) * np.sin(yaw),
        # x,
        # y + width * np.sin(yaw) - height * np.sin(roll) * np.cos(yaw),
        y,
        z + height * np.cos(roll)]


def intphys_to_shapes_world_object(**kwargs):
    OCCLUDER_SCALER = {"scale_x": 3.5, "scale_y": .1, "scale_z": 2.0}
    OBJECT_SCALER = {"scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0}
    o = intphys_to_pybullet(kwargs['attributes'], OBJECT_SCALER, OCCLUDER_SCALER, occluder_center_intphys)
    if len(o) == 0:
        return None
    obj = EasyDict({"name": "block1",
                    "shape": {
                        "shape_type": "box",
                        "shape_params": {
                            "scale_x": 1.0,
                            "scale_y": 1.0,
                            "scale_z": 1.0
                        }
                    },
                    "pose6d": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "yaw_radians": 0.0,
                        "pitch_radians": 0.0,
                        "roll_radians": 0.0
                    }
                    })

    obj.name = INV_OBJECT_ID_MAP[kwargs['object_id']]
    shape = "intphys_occluder" if o.obj_type == "occluder" \
        else "intphys_" + o.shape.lower()
    obj.shape["shape_type"] = shape

    obj.shape["shape_params"] = dict(o.scales)

    obj.pose6d.update({k: o[k] for k in "xyz"})

    obj.pose6d.update({k + "_radians": o[k] for k in ["roll", "pitch", "yaw"]})

    return dict(obj)


def intphys_to_shapes_world_camera(camera_terms):
    camera = {"camera_eye_pose": {"x": 0,
                                  "y": 0,
                                  "z": 0,
                                  "yaw_radians": 0,
                                  "pitch_radians": 0,
                                  "roll_radians": 0},
              "fov_degrees": 90,  # or should it be 60?
              "aspect": 1,
              "near_val": 0.01,
              "far_val": 100.0}
    x, y, z, roll, pitch, yaw = [camera_terms[k] for k in CAMERA_TERMS]
    l = locals()
    camera_eye = {k: l[k] for k in "xyz"}
    camera_eye.update({k + "_radians": l[k] for k in ["roll", "pitch", "yaw"]})
    # -theta=pitch, phi+90 =yaw, roll=0
    # theta = -pitch, phi = 180+yaw
    # theta = pitch, phi = yaw-90
    camera_eye["y"] *= -1
    camera_eye["yaw_radians"] *= -1
    camera_eye["yaw_radians"] = float(camera_eye["yaw_radians"] - np.pi / 2)
    camera["camera_eye_pose"] = camera_eye
    return camera


######## utils to transform to ADEPT ###########

def intphys_to_adept_object(**kwargs):
    FPS = 30
    OCCLUDER_SCALER = {"scale_x": 2, "scale_y": .1, "scale_z": 1}
    OBJECT_SCALER = {"scale_x": .5, "scale_y": .5, "scale_z": .5}
    # OCCLUDER_SCALER = {"scale_x": 3.5, "scale_y": .1, "scale_z": 2.0}
    # OBJECT_SCALER = {"scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0}
    attributes = kwargs["attributes"]
    prev_attributes = kwargs["prev_attributes"]
    o = intphys_to_pybullet(attributes, OBJECT_SCALER, OCCLUDER_SCALER, occluder_center_adept)
    if len(o) == 0:
        return None

    if prev_attributes is not None:
        p = intphys_to_pybullet(prev_attributes, OBJECT_SCALER, OCCLUDER_SCALER, occluder_center_adept)
        velocity = [(o[k] - p[k]) * FPS for k in "xyz"]
    else:
        velocity = [0, 0, 0]

    obj = EasyDict({"type": o.obj_type,
                    "location": [o.x, o.y, o.z],
                    "rotation": [o.roll, o.pitch, o.yaw],
                    "scale": [o.scales[k] for k in ["scale_x", "scale_y", "scale_z"]],
                    "angular_velocity": [0, 0, 0],
                    "color": "green",
                    "mask": kwargs["segmentation"],
                    "velocity": velocity})

    obj.mask["counts"] = obj.mask["counts"].decode("ascii")

    return obj


def occluder_center_adept(x, y, z, roll, yaw, box_params):
    width = box_params["scale_x"]
    height = box_params["scale_z"]
    return [
        x + width * np.cos(yaw) + height * np.sin(roll) * np.sin(yaw),
        # x,
        y + width * np.sin(yaw) - height * np.sin(roll) * np.cos(yaw),
        # y,
        z + height * np.cos(roll)]


def intphys_to_adept_camera(camera_terms):
    camera = {"camera_eye_pose": {"x": 0,
                                  "y": 0,
                                  "z": 0,
                                  "yaw_radians": 0,
                                  "pitch_radians": 0,
                                  "roll_radians": 0},
              "fov_degrees": 90,  # or should it be 60?
              "aspect": 1,
              "near_val": 0.01,
              "far_val": 100.0,
              "width":288,
              "height":288,}
    x, y, z, roll, pitch, yaw = [camera_terms[k] for k in CAMERA_TERMS]

    l = locals()
    look_at = [l[k] for k in "xyz"]
    look_at[1] *= -1

    camera_eye = {"camera_look_at": look_at,
                  "camera_phi": float(rad2deg(-yaw - np.pi / 2)),
                  "camera_theta": float(rad2deg(pitch)),
                  "camera_rho": 1.
                  }
    camera["camera_eye_pose"] = camera_eye

    return camera


def only_objects(anns, attributes_key):
    data = [(i, an) for i, an in enumerate(anns)
            if TYPE_MAP_INV[an[attributes_key]["type"]] == "object"]
    if len(data) == 0:
        return None, None
    return zip(*data)


def match_current_to_prev_anns(curr_anns, prev_anns, attributes_key):
    # raise NotImplementedError
    cur_idx, cur_objs = only_objects(curr_anns, attributes_key)
    prev_idx, prev_objs = only_objects(prev_anns, attributes_key)
    if cur_objs is None or prev_objs is None:
        return [None] * len(curr_anns)

    c_ids = np.array([o["object_id"] for o in cur_objs])
    assert np.unique(c_ids).size == c_ids.size
    cost_matrix = np.zeros((len(cur_objs), len(prev_objs)))
    for i, c_obj in enumerate(cur_objs):
        for j, p_obj in enumerate(prev_objs):
            cost_matrix[i, j] = l2_distance(c_obj[attributes_key],
                                            p_obj[attributes_key])
    matched_curr, matched_prev = linear_sum_assignment(cost_matrix)

    ## remove assignments with high distance ###
    costs = cost_matrix[matched_curr, matched_prev]
    matched_curr = matched_curr[costs < 2]
    matched_prev = matched_prev[costs < 2]

    prev_map = {curr_anns[cur_idx[c]]["object_id"]: prev_anns[prev_idx[p]]
                for c, p in zip(matched_curr, matched_prev)}

    return prev_map


def intphys_group_by_control_surprise(dataset_score_dicts):
    control_surprise_groups = defaultdict(lambda: defaultdict(list))
    [control_surprise_groups[d["dataset_split"]][os.path.dirname(d["original_video"])].append(d)
     for d in dataset_score_dicts]
    return control_surprise_groups
