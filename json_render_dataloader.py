import torch
import os
import time
import sys
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import trainers.trainable_derender
import configs.dataset_config as data_cfg
from datasets import utils, intphys

import seaborn as sns
import matplotlib.pyplot as plt
from utils.misc import setup_cfg, image_based_to_annotation_based, read_image
from run_experiment import parse_args
from structure.derender_attributes import DerenderAttributes

class IntphysJsonTensor(Dataset):
    def __init__(self, cfg, split, max_obj=10):
        self.cfg = cfg
        self.split = split
        data_cfg = cfg.DATA_CFG
        num_frames = utils.get_num_frames(data_cfg, split)
        dataset_name, standard_format_json_file = utils.get_dataset_name_and_json(data_cfg,split)
        print("reading datasets {}".format(dataset_name))
        start = time.time()
        self.dataset_dicts = utils.get_data_dicts(standard_format_json_file, num_frames)

        # Redoing derender_dataset from trainable_derenderer without detectron
        required_fields = ["pred_box"] if cfg.MODULE_CFG.DATASETS.USE_PREDICTED_BOXES else ["bbox"]
        required_fields += ["attributes"]
        # _, self.dataset_dicts = image_based_to_annotation_based(self.dataset_dicts,required_fields)

        # I've edited DerenderAttrbutes to basically do nothing.  This allows us
        # to call it without registering the dataset with detectron, but it also
        # loses a lot of the functionality it had.
        self.attributes = DerenderAttributes(cfg.MODULE_CFG)

        self.mapper = trainers.trainable_derender.ImageBasedDerenderMapper(cfg.MODULE_CFG.DATASETS.USE_PREDICTED_BOXES,
                            self.attributes,
                            False, #for_inference,
                            use_depth=cfg.MODULE_CFG.DATASETS.USE_DEPTH)

        non_visibles = []
        for i in range(len(self.dataset_dicts)):
            visibles = 0
            has_wall = False
            d = self.dataset_dicts[i]
            for annotation in d["annotations"]:
                visibles += (annotation["attributes"]["visible"] * (annotation["attributes"]["type"] == 0 or annotation["attributes"]["type"] == 1))
                has_wall = has_wall or (annotation["attributes"]["visible"] and annotation["attributes"]["type"] == 3)
            if (visibles == 0):# or has_wall):
                non_visibles.append(i)
        # non_visibles.reverse()
        for idx in range(len(non_visibles)-1, -1, -1):
            self.dataset_dicts.pop(non_visibles[idx])

        # self.dataset_dicts = list(map(self.mapper, self.dataset_dicts))
        # self.dataset_dicts = [self.mapper(x) for x in self.dataset_dicts]
        print("done after {}".format(time.time()-start))


    def __len__(self):
        return len(self.dataset_dicts)


    def __getitem__(self, idx):
        annotations = torch.zeros(10, 39)
        data_dict = self.mapper(self.dataset_dicts[idx])
        n_obj = len(data_dict["annotations"])
        i = 0
        for obj in data_dict["annotations"]:
            if not obj["attributes"]["visible"]:
                n_obj -= 1
            else:
                annotations[i, :] = attributesToTensor(obj)
                i += 1
        depth = data_dict["img_tuple"]
        return annotations, depth, n_obj
        # return attributesToTensor(self.dataset_dicts[idx]),self.dataset_dicts[idx]["img_tuple"]

def attributesToTensor(attr_dict):
    # Continuos terms (treat quantized as continuous for now)
      cont_terms = torch.tensor([attr_dict["attributes"][term] for term in intphys.CONTINUOUS_TERMS])
      quant_terms = torch.tensor([attr_dict["attributes"][term] for term in intphys.QUANTIZED_TERMS])
      camera_terms = torch.tensor([attr_dict["camera"][term] for term in attr_dict["camera"].keys()])
    # Categoric terms
      exists = torch.zeros(2)
      visible = torch.zeros(2)
      obj_type = torch.zeros(6)
      shape = torch.zeros(3)
      obj_id = torch.zeros(10)

      exists[attr_dict["attributes"]["existance"]] = 1
      visible[attr_dict["attributes"]["visible"]] = 1
      obj_type[attr_dict["attributes"]["type"]] = 1
      shape[attr_dict["attributes"]["shape"]] = 1
      obj_id[attr_dict["attributes"]["object_id"]] = 1
      # continuous terms
      # positions 0-9 33-38
      # categorical terms, 10-11, 12-13, 14-19, 20-22, 23-32
      return torch.cat([cont_terms, quant_terms, exists, visible, obj_type, shape, obj_id,
          camera_terms])

def tensorToAttributes(tensor):
    attributes = {intphys.CONTINUOUS_TERMS[i]:tensor[i] for i in range(len(intphys.CONTINUOUS_TERMS))}
    attributes["rotation_yaw"] = tensor[9]
    attributes["existance"] = torch.argmax(tensor[10:12])
    attributes["visible"] = torch.argmax(tensor[12:14])
    attributes["type"] = torch.argmax(tensor[14:20])
    attributes["shape"] = torch.argmax(tensor[20:23])
    attributes["object_id"] = torch.argmax(tensor[23:33])
    camera = {
    "cam_location_x" : tensor[33],
    "cam_location_y" : tensor[34],
    "cam_location_z" : tensor[35],
    "cam_rotation_roll" : tensor[36],
    "cam_rotation_pitch" : tensor[37],
    "cam_rotation_yaw" : tensor[38]
    }

    return {"attributes": attributes, "camera": camera}

class DatasetUtils(object):
    def __init__(self, val_data, device="cpu"):
        self.val_data = val_data
        self.device = device
        self.set_means_stds_attr()
        return
    def denormalize_attr(self, tensor):
        return tensor*self.attr_stds + self.attr_means
    def normalize_attr(self, tensor):
        return (tensor-self.attr_means)/self.attr_stds
    def denormalize_depth(self, tensor):
        # return tensor*self.depth_stds + self.depth_means
        # return tensor / self.depth_abs_max
        return (1/tensor) - 1
    def normalize_depth(self, tensor):
        # return (tensor-self.depth_means)/self.depth_stds
        # return tensor * self.depth_abs_max
        return 1/(1 + tensor)

    def set_means_stds_attr(self):
        start = time.time()
        all_attrs = []
        all_depths = torch.zeros(288, 288, len(self.val_data))
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                all_depths[:, :, i] = data[1]
                for attr in data[0]:
                    all_attrs.append(attr)
        all_attrs = torch.stack(all_attrs, dim=1)

        self.attr_means = all_attrs.mean(dim=1).to(self.device)
        self.attr_means[10:33] = 0
        self.attr_stds = all_attrs.std(dim=1).to(self.device)
        self.attr_stds[10:35] = 1
        self.depth_means = all_depths.mean(dim=2).to(self.device)
        self.depth_stds = all_depths.std(dim=2).to(self.device)
        self.depth_abs_max = abs(all_depths).max().to(self.device)
        print("Set means and std in {0} seconds".format(time.time()-start))
        return



def main(args):
    cfg = setup_cfg(args, args.distributed)
    #print(cfg)
    dataset =  IntphysJsonTensor(cfg, "_val")
    # omega_dataset = IntphysJsonTensor(cfg, "_train")

    """data = torch.zeros(288*288*10000)
    samples = torch.multinomial(torch.arange(len(omega_dataset), dtype=float), 10000)
    for i in range(10000):
        ann, depth, n_obj = omega_dataset[samples[i]]
        data[i*288*288:(i+1)*288*288] = depth.flatten()
    sns.displot(data=data.numpy(), kind="kde")"""

    util = DatasetUtils(dataset)
    # train_dataset = IntphysJsonTensor(cfg, "_train")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)
    # utils = DatasetUtils(dataset)
    return dataloader

if __name__ == "__main__":
    args = parse_args()
    main(args)

