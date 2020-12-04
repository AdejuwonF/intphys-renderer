import json
import os
import time
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

import numpy as np
import pycocotools

# from configs.detection_config import _C as cfg

from datasets.utils import register_dataset, get_dataset_name_and_json, get_data_dicts, find_bounding_box, \
    find_bbox_area
from utils.io import read_serialized

from detectron2.structures import BoxMode

from utils.misc import CodeTimer

CONTINUOUS_TERMS = ["scale_x", "scale_y", "scale_z", "x", "y", "z",
                     "yaw_radians", "pitch_radians", "roll_radians", 'radius', 'length']

CATEGORICAL_TERMS = ["shape", "visible"]
# _TERMS = _CONTINUOUS_TERMS + _CATEGORICAL_TERMS

VALID_MAP = {"scale_x": {"shape": ("box",)},
              "scale_y": {"shape": ("box",)},
              "scale_z": {"shape": ("box",)},
              'radius': {"shape": ("cylinder",)},
              'length': {"shape": ("cylinder",)}}

SHAPE_MAP = {"box":0,
             "cylinder":1}

def process_object(segm, an, oid, min_area):
    object_id = an["debug"]["ground_truth_mask_id"]
    mask = segm ==object_id

    attributes = an["pose6d"]
    attributes["shape"] = _SHAPE_MAP[an["shape"]["shape_type"]]
    [attributes.__setitem__(k, an["shape"]["shape_params"].get(k, -1))
     for k in ["scale_x", "scale_y", "scale_z", "radius", "length"]]
    attributes["visible"] = 0

    res = {}

    if mask.sum()> min_area:
        encoded_mask = pycocotools.mask.encode(np.asarray(mask, order="F"))
        encoded_mask["counts"] = encoded_mask["counts"].decode('ascii')

        bbox = find_bounding_box(segm, object_id)
        bounding_box = [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]
        bounding_box = [int(el) for el in bounding_box]

        if find_bbox_area(bounding_box) > min_area:
            res.update({"bbox": bounding_box,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": encoded_mask,
                        "category_id": 0,
                        "iscrowd": 0})
            attributes["visible"] = 1

    res["attributes"] = attributes

    return res

    # for obj in anns["objects"]:
    #     if obj["debug"]["ground_truth_mask_id"] == object_id:
    #         attributes = obj["pose6d"]
    #         # attributes["name"] = obj["name"]
    #         attributes["shape"] = _SHAPE_MAP[obj["shape"]["shape_type"]]
    #         [attributes.__setitem__(k,obj["shape"]["shape_params"].get(k, -1))
    #          for k in ["scale_x", "scale_y", "scale_z", "radius", "length"]]
    #         res["attributes"] = attributes
    #         return res

def process_frame(video_folder, vid_num, frame, frame_num, min_area):
    depth_map_file, segmentation_file,annotation_file = \
        map(lambda ending: os.path.join(video_folder, f'{frame}{ending}'),
            [".depth.npy", ".seg.npy", ".json"])

    segm = np.load(segmentation_file)
    anns = read_serialized(annotation_file)

    height, width = segm.shape
    annotations = [process_object(segm, an, oid, min_area) for oid,an in enumerate(anns["objects"])]
    # annotations = [an for an in annotations if find_bbox_area(an["bbox"]) > min_area]

    return {"file_name": depth_map_file,
             "image_id": vid_num*500 + frame_num,
             "height": height,
             "width": width,
             "annotations": annotations}


def process_video(video_folder,vid_num, min_area):
    frames = sorted(set([frame.replace('.json', '').replace('.seg.npy', '').replace('.depth.npy', '')
                         for frame in os.listdir(video_folder)]))
    frames_dicts = [process_frame(video_folder, vid_num, frame, frame_num, min_area)
                    for frame_num,frame in enumerate(frames)]
    return frames_dicts

def shapes_world_to_detectron(cfg, split, out_file):
    timer = CodeTimer("started processing videos")
    video_list = sorted(os.listdir(cfg.DATAFOLDER))

    if split != "_train":
        video_list = video_list[:cfg.VAL_VIDEOS]
    else:
        video_list = video_list[cfg.VAL_VIDEOS:]

    worker_args = []
    for i, video in enumerate(video_list):
        if i > cfg.MAX_VIDEOS:
            break
        video_folder = os.path.join(cfg.DATAFOLDER, video)
        worker_args.append((video_folder, i, cfg.MIN_AREA))

    if not cfg.DEBUG:
        with Pool(int(cpu_count() / 2)) as p:
            dicts = p.starmap(process_video, worker_args)
    else:
        dicts = [process_video(*w) for w in worker_args]

    dicts = [d for vid in dicts for d in vid]
    # dicts = sample_from_dataset(cfg, dicts, split)

    with open(out_file, 'w') as f:
        json.dump(dicts, f, indent=4)

    timer.done()


# def print_mean_std_of_inv_depth(cfg):
#     video_list = sorted(os.listdir(cfg.DATAFOLDER))
#     random.shuffle(video_list)
#     all_inv_depths = []
#     start =  time.time()
#     for i,video in enumerate(video_list[:500]):
#         video_folder = os.path.join(cfg.DATAFOLDER, video)
#         frames = sorted(set([frame.replace('.json', '').replace('.seg.npy', '').replace('.depth.npy', '')
#                              for frame in os.listdir(video_folder)]))
#         if i%100 ==0:
#             print(time.time()-start, " ", i)
#         for frame in frames:
#             depth_map_file = os.path.join(video_folder, f'{frame}.depth.npy')
#             depth = np.load(depth_map_file)
#             all_inv_depths.append((1/(1+depth)).reshape(-1))
#
#     all_inv_depths = np.concatenate(all_inv_depths, axis=0)
#     print(all_inv_depths.mean())
#     print(all_inv_depths.std())


# def build_shapes_world(cfg):
#     data_cfg = cfg.DATA_CFG
#     splits = ["_val", "_train", "_test"]
#     for split in splits:
#         dataset_name, standard_format_json_file = get_dataset_name_and_json(data_cfg, split)
#         if not os.path.exists(standard_format_json_file) or data_cfg.RECOMPUTE_DATA:
#             dump_data_dicts(data_cfg,split,standard_format_json_file)
#         else:
#             register_dataset(data_cfg, split)
#
#     #########Dataset specific attributes for the detector###############
#     if hasattr(cfg, "MODULE_CFG"):
#         module_cfg = cfg.MODULE_CFG
#         module_cfg.DATASETS.TRAIN = ("shapes_world_train",)
#         module_cfg.DATASETS.TEST = ("shapes_world_val",)
#         module_cfg.MODEL.PIXEL_MEAN = [0.11376177423599798]
#         module_cfg.MODEL.PIXEL_STD = [0.07010949697775344]
#         module_cfg.INPUT.MIN_SIZE_TRAIN = (256,)
#         module_cfg.INPUT.MAX_SIZE_TRAIN = 256
#         module_cfg.INPUT.MIN_SIZE_TEST = 256
#         module_cfg.INPUT.MAX_SIZE_TEST = 256



# if __name__ == "__main__":
#     if cfg.COCO_WRITE.RECOMPUTE:
#         dump_data_dicts(cfg.COCO_WRITE, split="_val")
#         dump_data_dicts(cfg.COCO_WRITE, split="_test")
#         dump_data_dicts(cfg.COCO_WRITE, split="_train")
#     else:
#         print("not rewritting files because of cfg.COCO_WRITE.RECOMPUTE")
#         for split in ['_val', '_train', '_test']:
#             register_dataset(cfg.COCO_WRITE, split)
#
#     if cfg.COCO_WRITE.VISUALIZE:
#         meta_data  = MetadataCatalog.get("shapes_world_train")
#         dataset_dicts = get_data_dicts(cfg.OUT_FILE_BASENAME + "_train" + '.json')
#         for d in random.sample(dataset_dicts, 3):
#             img = np.load(d["file_name"])/20.0  * 255.0
#             img = np.stack([img]*3,axis=2)
#             visualizer = Visualizer(img[:, :, ::-1], metadata=meta_data, scale=1.0)
#             vis = visualizer.draw_dataset_dict(d)
#             plt.imshow(vis.get_image()[:, :, ::-1])
#             plt.show()
#
#     if cfg.COCO_WRITE.PRINT_MEAN_STD_DEPTH:
#         # 0.11376177423599798 -> mean of 1/(1+depth)
#         # 0.07010949697775344 -> std of 1/(1+depth)
#         print_mean_std_of_inv_depth(cfg.COCO_WRITE)
