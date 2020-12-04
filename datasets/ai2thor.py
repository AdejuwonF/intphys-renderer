import itertools
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing import cpu_count, Pool
import random

import pycocotools
from detectron2.structures import BoxMode

from datasets.utils import find_bounding_box, find_bbox_area
from utils.io import read_serialized, write_serialized
from utils.misc import CodeTimer
import pathlib
from PIL import Image
import numpy as np

_AXIS = ['x', 'y', 'z']
_POSITIONS = [f'pos_{e}' for e in _AXIS]
_DIMENSIONS = [f'dim_{a}_{i}' for a in _AXIS for i in range(8)]
_ROTATION = ['rotation']

SHAPE_MAP = {'structural': 0,
             'chair': 1,
             'blank block cube': 2,
             'blank block cylinder': 3,
             'potted plant': 4,
             'table': 5,
             'box': 6,
             'changing table': 7,
             'stool': 8,
             'crib': 9,
             'shelf': 10,
             'cylinder': 11,
             'sofa': 12,
             'cube': 13,
             'sofa chair': 14,
             'ball': 15,
             'drawer': 16,}

SHAPE_MAP_INV = {v: k for k, v in SHAPE_MAP.items()}

CONTINUOUS_TERMS = _POSITIONS + _DIMENSIONS + _ROTATION
CATEGORICAL_TERMS = ["shape", "visible", "existance"]
ROTATION_TERMS = []
POSITIVE_TERMS = []
QUANTIZED_TERMS = []
VALID_MAP = {}
OBJECT_ID_MAP = {k: k for k in range(10)}

_DUMMY_ATTRIBUTES = {k: 0 for k in CONTINUOUS_TERMS + CATEGORICAL_TERMS}
_DUMMY_ATTRIBUTES["existance"] = 1


def process_attributes(obj):
    valid = True
    attributes = deepcopy(_DUMMY_ATTRIBUTES)

    ###### positions #########
    [attributes.__setitem__(f'pos_{a}', obj["position"][a]) for a in _AXIS]

    ###### dimensions ########
    if len(obj['dimensions']) == 0:
        valid = False
    else:
        [attributes.__setitem__(f'dim_{a}_{i}', obj['dimensions'][i][a]) for a in _AXIS for i in range(8)]

    attributes["rotation"] = obj["rotation"]

    ###### shape ########
    attributes["shape"] = SHAPE_MAP[obj["shape"]]

    return attributes, valid

def process_object(obj, segm_array_id, min_area):
    an = {"category_id": 0, "iscrowd": 0, "object_id": obj["uuid"]}

    attributes, valid_attributes = process_attributes(obj)
    try:
        segm_id = obj["color"]['r'] * 256 ** 2 + obj["color"]['g'] * 256 + obj["color"]['b']
    except TypeError: #TODO: sometimes color contains None as the colors
        segm_id = -1

    if segm_id in segm_array_id:
        bbox = find_bounding_box(segm_array_id, segm_id)
        bounding_box = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]
        bounding_box = [int(el) for el in bounding_box]

        if find_bbox_area(bounding_box) > min_area:
            mask = segm_array_id == segm_id
            encoded_mask = pycocotools.mask.encode(np.asarray(mask, order="F"))
            encoded_mask["counts"] = encoded_mask["counts"].decode('ascii')
            an.update({"bbox": bounding_box,
                       "bbox_mode": BoxMode.XYXY_ABS,
                       "segmentation": encoded_mask})
            attributes["visible"] = 1

    if valid_attributes:
        an["attributes"] = attributes

    return an


def process_frame(frame_json, vid_number, min_area):
    frame_data = read_serialized(str(frame_json))
    frame_num = int(frame_json.parts[-1].split("_")[0])

    depth_file = str(frame_json.parent / f'depth/{str(frame_num).zfill(4)}_depth.png')
    segm_file = str(frame_json.parent / f'segmentation/{str(frame_num).zfill(4)}_seg.png')
    depth_array, segm_array = map(lambda x: np.array(Image.open(x), dtype=np.uint64), [depth_file, segm_file])

    segm_array_id = segm_array[:, :, 0] * 256 ** 2 + segm_array[:, :, 1] * 256 + segm_array[:, :, 2]

    height, width = depth_array.shape
    # try:
    structural_anns = [process_object(*w) for w in zip(frame_data["structural"],
                                                   repeat(segm_array_id),
                                                   repeat(min_area))]

    nonstructural_anns = [process_object(*w) for w in zip(frame_data["nonstructural"],
                                                        repeat(segm_array_id),
                                                        repeat(min_area))]
    # except:
    #     traceback.print_exc()
    #     print(frame_json)
    #     assert False

    annotations = structural_anns + nonstructural_anns

    annotations = [an for an in annotations if an is not None]

    out_frame = {"file_name": depth_file,
                 "image_id": vid_number * 500 + frame_num,
                 'height': height,
                 'width': width,
                 'annotations': annotations,
                 'original_video': frame_json.parts[-2]
                 }

    return out_frame


def process_video(video_folder, i, min_area):
    frames_jsons = sorted(list(pathlib.Path(video_folder).glob('*_objects*')))

    frame_dicts = [process_frame(f_json, i, min_area) for f_json in frames_jsons]
    return frame_dicts


def ai2thor_intphys_to_detectron(cfg, split, outfile):
    timer = CodeTimer(f'started processing videos for {split}')
    if split == "_val":
        base_folder = os.path.join(cfg.DATA_LOCATION, 'intphys_scenes_validation_dumped_perception')
    elif split == "_train":
        base_folder = os.path.join(cfg.DATA_LOCATION, 'intphys_scenes_dumped_perception')
    else:
        raise NotImplementedError

    video_folders = list(map(lambda x: os.path.join(base_folder, x), sorted(os.listdir(base_folder))))

    if cfg.DEBUG_VIDEOS is not None and cfg.DEBUG:
        video_folders = cfg.DEBUG_VIDEOS

    worker_args = zip(video_folders, range(len(video_folders)), repeat(cfg.MIN_AREA))

    if cfg.DEBUG:
        dicts = [process_video(*w) for w in worker_args]
    else:
        with Pool(int(cpu_count())) as p:
            dicts = p.starmap(process_video, worker_args)

    dicts = list(itertools.chain.from_iterable(dicts))

    write_serialized(dicts, outfile)
    timer.done()