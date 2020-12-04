import json
import os
import random
from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np

from detectron2.structures import BoxMode
from easydict import EasyDict

from utils.misc import read_image
import os.path as osp


def get_num_frames(data_cfg, split):
    num_frames = -1 if split == "_train" \
    else data_cfg.VAL_FRAMES if split == "_val" \
    else data_cfg.TEST_FRAMES
    return num_frames

def get_data_dicts(output_file, num_frames):
    with open(output_file) as f:
        dicts = json.load(f)

    dicts = sample_from_dataset(dicts, num_frames)

    with Pool(int(cpu_count()/2)) as p:
        # serialization makes the constant an integer and that breaks compatibility
        dicts = p.map(fix_from_serialization, dicts)

    return dicts


def register_dataset(data_cfg, split, getter= None, name=None):
    dataset_name, standard_format_json_file = get_dataset_name_and_json(data_cfg, split)
    dataset_name = name if name is not None else dataset_name
    num_frames = get_num_frames(data_cfg, split)
    if getter == None:
        DatasetCatalog.register(dataset_name, lambda: get_data_dicts(standard_format_json_file, num_frames))
    else:
        DatasetCatalog.register(dataset_name, lambda: getter())
    MetadataCatalog.get(dataset_name).set(thing_classes=["object"])
    MetadataCatalog.get(dataset_name).set(json_file=standard_format_json_file.replace(".json", "_coco_format.json"))
    return dataset_name, standard_format_json_file

def get_dataset_name_and_json(data_cfg,split):
    dataset_name = data_cfg.BASE_NAME + split
    standard_format_json_file = os.path.join(data_cfg.BASE_DIRECTORY, dataset_name + '.json')
    return dataset_name, standard_format_json_file

def box_area(box):
    if box[0]>box[2] or box[1] > box[3]:
        return 0.0
    return (box[2]-box[0]) * (box[3] - box[1])

def iou_boxes(box1,box2):
    for b in [box1,box2]:
        assert b[0] < b[2] and b[1] < b[3],  "mode should be xyxy"

    both_boxes = np.stack([box1,box2])
    #maximum min locations
    min_x, min_y = both_boxes[:,:2].max(axis=0)
    #minimum max locations
    max_x, max_y = both_boxes[:,2:].min(axis=0)

    intersection_area = box_area([min_x,min_y,max_x,max_y])

    union_area = box_area(box1) + box_area(box2) - intersection_area

    return intersection_area/union_area


def fix_from_serialization(dict):
    for an in dict["annotations"]:
        if "bbox_mode" in an:
            an["bbox_mode"] = BoxMode.XYXY_ABS
        if "segmentation" in an:
            an["segmentation"]["counts"] = an["segmentation"]["counts"].encode("ascii")
    return dict


def fix_for_serialization(coco_dict):
    for an in coco_dict["annotations"]:
        if "area" in an:
            an["area"] = int(an["area"])
        if "segmentation" in an:
            an["segmentation"]["counts"] = an["segmentation"]["counts"].decode("ascii")
    return coco_dict


def sample_from_dataset(dicts, num_frames):

    if num_frames == -1:
        return dicts

    random.seed(1234)
    dicts = random.sample(dicts, num_frames)
    return dicts


def find_bounding_box(mask_array, segm_id):
    '''
    :param mask_array: 2d numpy array where different  integers correspond to different object masks
    :param segm_id: object id to compute a bounding box for
    :return: box_coordinates -> EasyDict with the outermost coordinates of the segmentation mask
    '''
    rows,cols = (mask_array == segm_id).nonzero()
    return EasyDict(min_y = rows.min(),
                    max_y = rows.max(),
                    min_x = cols.min(),
                    max_x = cols.max())


def find_bbox_area(bbox):
    min_x, min_y, max_x, max_y  =  bbox
    return (max_x - min_x) * (max_y - min_y)

def print_mean_std_inv_depth(data_dicts):
    all = []
    for d in data_dicts:
        depth = read_image(d["file_name"])
        inv_depth = 1/(1+depth)
        all.append(inv_depth.reshape(-1))
    all = np.concatenate(all)
    print("mean: ", all.mean(), "std: ", all.std())


def frames2videos(dataset):
    videos = defaultdict(dict)
    for d in dataset:
        vid_num = d["image_id"]//500
        frame_num = d["image_id"] % 500

        videos[vid_num][frame_num] = d

    return videos
