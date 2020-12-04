import json
import os

# import ipdb
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment

from configs.main import load_cfg_from_file
from datasets.utils import iou_boxes, fix_for_serialization, get_dataset_name_and_json
# from trainers.trainable_detector import DetectorPredictor, inference_detection_loader, detection_mapper
from trainers.trainable_detector import DetectorPredictor, inference_detection_loader, DetectionMapper
from utils.misc import CodeTimer, filter_dataset, infer_score_threshold
from multiprocessing import Pool, cpu_count


def add_box_to_plot(img, box, color):
    box = box.astype(np.int32).clip(0, 255)
    img[[box[1], box[3]], box[0]:box[2]] = [color]
    img[box[1]:box[3], [box[0], box[2]]] = [color]
    return img


def plot_input(input, box_list=[], mask_list=[]):
    COLOR_MAP = [40, 40, 40, 60, 80, 80, 80]
    if input is not None:
        img = np.load(input['file_name']) / 20 * 255
    else:
        img = np.zeros((256, 256))
    # img = np.stack([img]*3,axis=2)
    for i, b in enumerate(box_list):
        img = add_box_to_plot(img, b, COLOR_MAP[i])
    plt.imshow(img)
    plt.show()


def match_gt_to_pred_boxes(gt_boxes, pred_boxes, pred_scores):
    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, inf_box in enumerate(pred_boxes):
            cost_matrix[i, j] = (1.0 - iou_boxes(gt_box, inf_box))  # *(1/pred_scores[j])
            assert cost_matrix[i, j] <= 1 and cost_matrix[
                i, j] >= 0, "intersections over unions are always between 0 and 1"
    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    ## remove assignments with bad iou ###
    costs = cost_matrix[gt_idx, pred_idx]
    gt_idx = gt_idx[costs < 0.5]
    pred_idx = pred_idx[costs < 0.5]

    gt2pred = {gt: pr for gt, pr in zip(gt_idx, pred_idx)}
    return gt2pred


def filter_redundant_boxes(boxes, scores):
    filtered_boxes = []
    filtered_scores = []
    boxes = np.flip(boxes, axis=0)
    scores = np.flip(scores)
    for i, box1 in enumerate(boxes):
        to_add = True
        for box2 in boxes[i + 1:]:
            if iou_boxes(box1, box2) > 0.5:
                to_add = False
                # plot_input(None,[box1,box2])
                print("found redundant boxes")
                break
        if to_add:
            filtered_boxes.append(box1)
            filtered_scores.append(scores[i])
    return np.flip(np.array(filtered_boxes), axis=0), np.flip(np.array(filtered_scores))


def add_box_to_dict(my_dict, box, score, ann_idx=None):
    if "annotations" not in my_dict:
        assert ann_idx == None
        my_dict["annotations"] = []

    to_add = {"pred_box": [int(b) for b in box.clip(0, 255)],
              "pred_score": float(score),
              "category_id": 0}

    if ann_idx is None:
        my_dict["annotations"].append(to_add)
    else:
        my_dict["annotations"][ann_idx].update(to_add)


def add_inferred_boxes(input, output):
    # TODO: empty lists can be empty
    pred_boxes = output["pred_boxes"]
    pred_scores = output["scores"]
    pred_boxes, pred_scores = filter_redundant_boxes(pred_boxes, pred_scores)

    # filter_ids = pred_scores>0.8
    # pred_boxes = pred_boxes[filter_ids]
    # pred_scores = pred_scores[filter_ids]
    visible_idx, visible_input = filter_dataset([input], ["bbox"])
    visible_idx = [v[1] for v in visible_idx]
    visible_input = visible_input[0]

    # TODO: what about not having any boxes as in  the test set?
    gt_boxes = np.array([an["bbox"] for an in visible_input["annotations"]])
    ####find assignments between boxes that minimizes the iou of matching######
    gt2pred = match_gt_to_pred_boxes(gt_boxes, pred_boxes, pred_scores)

    ####add predictions to matched ground truth objects (only visible)########
    for gt, pr in gt2pred.items():
        add_box_to_dict(visible_input, pred_boxes[pr], pred_scores[pr], ann_idx=gt)

    [input["annotations"].__setitem__(v, val)
     for v, val in zip(visible_idx, visible_input["annotations"])]

    ####add non_assigned predicted boxes (used for inference, or when no annotations available)######
    only_pred_boxes_idx = [i for i in range(len(pred_boxes)) if i not in gt2pred.values()]
    for box_id in only_pred_boxes_idx:
        add_box_to_dict(input, pred_boxes[box_id], pred_scores[box_id])

    return fix_for_serialization(input)


def parse_worker_args(input, output):
    new_input = {k: input[k] for k in input.keys() if k not in {"image", "instances"}}
    new_output = {"pred_boxes": output["instances"].pred_boxes.tensor.cpu().numpy(),
                  "scores": output["instances"].scores.cpu().numpy()}
    return new_input, new_output


def filter_predicted_boxes_threshold(dicts, threshold):
    new_dicts = []
    for d in dicts:
        new_dicts.append({k: d[k] for k in d.keys() if k != "annotations"})
        new_dicts[-1]["annotations"] = []
        for an in d["annotations"]:
            new_an = {k:v for k,v in an.items() if k not in {'pred_box', 'pred_score'}}
            if 'pred_box' in an:
                score = an['pred_score']
                if score >= threshold:
                    new_an.update({k:an[k] for k in {'pred_box', 'pred_score'}})
            if len(new_an) >= 2: #new_an might end up empty if it is just an spurious box
                new_dicts[-1]["annotations"].append(new_an)

    return new_dicts



def write_with_inferred_boxes(cfg, split):
    # TODO: now there are invisible objects the detection mapper  ignores that, will have to debug tomorrow
    timer = CodeTimer("adding inferred boxes")
    module_cfg = os.path.join(cfg.TRAINED_DETECTOR.EXP_DIR, "config.yaml")
    module_cfg = load_cfg_from_file(module_cfg)
    module_cfg.MODEL.WEIGHTS = cfg.TRAINED_DETECTOR.WEIGHTS_FILE
    if cfg.DEBUG:
        module_cfg.DATALOADER.NUM_WORKERS = 0

    predictor = DetectorPredictor(module_cfg)

    dataset_name, standard_format_json_file = get_dataset_name_and_json(cfg, split)
    data_loader = inference_detection_loader(module_cfg.clone(), dataset_name, DetectionMapper(module_cfg.clone()))

    worker_args = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = predictor(inputs)
            for i in range(len(outputs)):
                worker_args.append(parse_worker_args(inputs[i], outputs[i]))

    if cfg.DEBUG:
        new_dicts = [add_inferred_boxes(*w) for w in worker_args]
    else:
        with Pool(int(cpu_count() / 4)) as p:
            new_dicts = p.starmap(add_inferred_boxes, worker_args)

    if 'PRED_BOX_SCORE_THRESHOLD' not in cfg:
        assert '_val' in split, "start with validation split to compute detection threshold"
        cfg.PRED_BOX_SCORE_THRESHOLD = infer_score_threshold(new_dicts)

    new_dicts = filter_predicted_boxes_threshold(new_dicts, cfg.PRED_BOX_SCORE_THRESHOLD)

    with open(standard_format_json_file, 'w') as f:
        json.dump(new_dicts, f, indent=4)

    timer.done()
