from collections import defaultdict
import time

import numpy as np
import torch
from PIL import Image
from sklearn import svm

from configs.main import set_output_directories, get_cfg

def to_cuda(x,device=None):
    """
    Converts tensors to CUDA, recursively goes through lists and dictionaries, keeps non-tensors untouched
    Args:
        x: Object to (recursively) convert to CUDA via Torch's to() method

    Returns:

    """
    if torch.is_tensor(x):
        return x.cuda() if device is None else x.to(device)
    elif isinstance(x, list):
        return [to_cuda(_,device) for _ in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v,device) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return torch.tensor(x,device=device)
    else:
        return x


def gather_loss_dict(outputs):
    outputs["loss_dict"] = {k: v.mean() for k, v in outputs["loss_dict"].items()}
    return outputs["loss_dict"]

def average_err_dict(outputs):
    outputs["errors"] = {k: v.mean() for k, v in outputs["errors"].items()}
    return outputs["errors"]

# def filter_dataset(dict_list, required_fields):
#     new_dicts = []
#     for  d in dict_list:
#         new_dicts.append({k: d[k] for k in d.keys() if k != "annotations"})
#         for an in d["annotations"]:
#             new_dicts[-1][]
#             if all([r in an for r in required_fields]):
#                 new_dicts[-1].update(an)
#
# def image_based_to_annotation_based(dict_list, use_predicted_boxes, for_inference=False):
#     #TODO: change this filtering to accomodate dynamics models
#     required = []
#     required += ["attributes"] if not for_inference else []
#     required += ["pred_box"] if use_predicted_boxes else []
#     new_dicts = []
#     filtered_indices = []
#     for d_idx,d in  enumerate(dict_list):
#         for a_idx,an in enumerate(d["annotations"]):
#             if all([r in an for r  in required]):
#                 filtered_indices.append((d_idx,a_idx))
#                 new_dicts.append({k: d[k] for k in d.keys() if k != "annotations"})
#                 new_dicts[-1].update(an)
#     return new_dicts

def recursive_field_eval(ann, key, val):
    #TODO: not sure if this always apply
    if not key in ann:
        return False
    ann_val = ann[key]
    if isinstance(ann_val, dict):
        for k, v in val.items():
            if not recursive_field_eval(ann_val, k, v):
                return False
        return True
    else:
        return ann_val == val

def filter_dataset(dict_list, required_fields=[], required_fields_values=None):
    if required_fields_values is None:
        required_fields_values = {}

    new_dicts = []
    filtered_indices = []
    for d_idx,d in  enumerate(dict_list):
        new_dicts.append({k: d[k] for k in d.keys() if k != "annotations"})
        new_dicts[-1]["annotations"] = []
        for a_idx,an in enumerate(d["annotations"]):
            if all([r in an for r in required_fields]) and \
                    all([recursive_field_eval(an, k,v)
                         for k,v in required_fields_values.items()]):
                filtered_indices.append((d_idx, a_idx))
                new_dicts[-1]["annotations"].append(an)
    return filtered_indices, new_dicts

def image_based_to_annotation_based(dict_list, required_fields):  # use_predicted_boxes, for_inference=False):
    filtered_indices, filtered_dicts = filter_dataset(dict_list, required_fields)
    new_dicts = []
    for d_idx, d in enumerate(filtered_dicts):
        for a_idx, an in enumerate(d["annotations"]):
            new_dicts.append({k: d[k] for k in d.keys() if k != "annotations"})
            new_dicts[-1].update(an)
    return filtered_indices, new_dicts

    # dataset_dicts = [image_based_to_annotation_based(d, use_predicted_boxes, for_inference)
    #                  for d in dict_list]
    # dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    # return dataset_dicts

# def _image_based_to_annotation_based(single_dict, use_predicted_boxes, for_inference=False):
#     '''
#     :param use_predicted_boxes: the derender can predict based on ground truth boxes or predicted boxes from a detector
#     :param for_inference: at inference no labels are required, so all boxes can  be used
#     :param single_dict: dictionary containing a single image file and annotations in coco_format
#     :return:  a list of dictionaries each with a single annotation, it filters annotations depending on mode of the dataset
#     '''
#     required = []
#     required += ["attributes"] if not for_inference else []
#     required += ["pred_box"] if use_predicted_boxes else []
#     new_dicts = []
#     for an in single_dict["annotations"]:
#         if all([r in an for r  in required]):
#             new_dicts.append({k: single_dict[k] for k in single_dict.keys() if k != "annotations"})
#             new_dicts[-1].update(an)
#     return new_dicts


def collate_attributes(dict_list):

    collated_dict = defaultdict(list)

    for d in dict_list:
        for an in d["annotations"]:
            if "attributes" in an:
                for term,val in an["attributes"].items():
                    collated_dict[term].append(val)

    return {k:torch.Tensor(v) for k,v in collated_dict.items()}

def read_image(file_path):
    if file_path.endswith(".npy"):
        return np.load(file_path)
    elif file_path.endswith(".png"): #TODO: do only intphys has this behavior
        if "mcs-data" in file_path:
            img = np.array(Image.open(file_path), dtype=np.float)
            return img
        elif "intphys" in file_path:
            img = np.asarray(Image.open(file_path), dtype=np.float)
            #Intphys  encoding  here: https://www.intphys.com/benchmark/training_set.html
            return (2**16 - 1 - img)/100.0
        elif "adept" in file_path:
            img = np.asarray(Image.open(file_path).convert("RGB"), dtype=np.float)
            img = np.moveaxis(img,2,0)
            return img
        else:
            raise NotImplementedError
    raise NotImplementedError


def setup_cfg(args, distributed):
    cfg = get_cfg(args)
    set_output_directories(cfg, distributed)
    return cfg

def quantized_idx2float(idx, quant_array):
    return float(quant_array[idx])

def quantized_float2idx(value, quant_array):
    return int(np.abs(quant_array - value).argmin())

def quantized_onehot2floats_batch(onehots, quant_array):
    indices = onehots.argmax(axis=1)
    return quant_array[indices]

def quantized_idx2floats_batch(indices, quant_array):
    return quant_array[indices]

class CodeTimer():
    def __init__(self, message):
        print(message)
        self.start = time.time()

    def done(self):
        print("done after {}".format(time.time() - self.start))


def l2_distance(obj1, obj2):
    obj1_loc, obj2_loc = map(lambda x: np.array([x[k] for k in ["location_x", "location_y", "location_z"]]),
                             [obj1, obj2])
    l2_dist = np.sqrt(np.power(obj1_loc - obj2_loc, 2).sum())

    return l2_dist


def infer_score_threshold(dataset):
    print("inferring best threshold for prediction")
    scores = []
    labels = []
    for d in dataset:
        for an in d["annotations"]:
            if "bbox" in an and "pred_box" in an:
                labels.append(1.0)
                scores.append(an["pred_score"])
            elif "pred_box" in an and not "bbox" in an:
                labels.append(0.0)
                scores.append(an["pred_score"])

    scores = np.array(scores).reshape(-1, 1)
    labels = np.array(labels)
    clf = svm.SVC(kernel="linear")
    clf.fit(scores,labels)

    xx = np.linspace(0,1,num=1000).reshape(-1,1)
    yy = clf.predict(xx)

    threshold_idx = np.where(yy[:-1] != yy[1:])[0]
    threshold = xx[threshold_idx].item()

    y_hat = clf.predict(scores)
    err = np.abs(labels-y_hat).sum()/len(labels)

    print("using threshold {}, {:.2f}% of boxes are a false positive/negative".format(threshold, err*100))

    return threshold
