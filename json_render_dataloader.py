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


from utils.misc import setup_cfg, image_based_to_annotation_based, read_image
from run_experiment import parse_args
from structure.derender_attributes import DerenderAttributes

class IntphysJsonTensor(Dataset):
    def __init__(self, cfg, split):
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
        _, self.dataset_dicts = image_based_to_annotation_based(self.dataset_dicts,required_fields)

        # I've edited DerenderAttrbutes to basically do nothing.  This allows us
        # to call it without registering the dataset with detectron, but it also
        # loses a lot of the functionality it had.
        self.attributes = DerenderAttributes(cfg.MODULE_CFG)

        mapper = trainers.trainable_derender.DerenderMapper(cfg.MODULE_CFG.DATASETS.USE_PREDICTED_BOXES,
                            self.attributes,
                            False, #for_inference,
                            use_depth=cfg.MODULE_CFG.DATASETS.USE_DEPTH)
        self.dataset_dicts = list(map(mapper, self.dataset_dicts))
        print("done after {}".format(time.time()-start))


    def __len__(self):
        return len(self.dataset_dicts)


    def __getitem__(self, idx):
        return attributesToTensor(self.dataset_dicts[idx]),self.dataset_dicts[idx]["img_tuple"]

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
      # positions 0-9
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
        self.set_means_stds_attr()
        self.attr_stds = self.attr_stds.to(device)
        self.attr_means = self.attr_means.to(device)
        return
    def denormalize_attr(self, tensor):
        return tensor*self.attr_stds + self.attr_means
    def normalize_attr(self, tensor):
        return (tensor-self.attr_means)/self.attr_stds
    def set_means_stds_attr(self):
        start = time.time()
        all_attrs = torch.zeros((39, len(self.val_data)))
        with torch.no_grad():
            for i, attrs in enumerate(self.val_data):
                all_attrs[:,i] = attrs[0]

        self.attr_means = all_attrs.mean(dim=1)
        self.attr_means[10:33] = 0
        self.attr_stds = all_attrs.std(dim=1)
        self.attr_stds[10:33] = 1
        print("Set means and std in {0} seconds".format(time.time()-start))
        return



def main(args):
    cfg = setup_cfg(args, args.distributed)
    #print(cfg)
    dataset =  IntphysJsonTensor(cfg, "_val")
    # train_dataset = IntphysJsonTensor(cfg, "_train")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)
    # utils = DatasetUtils(dataset)
    return dataloader

if __name__ == "__main__":
    args = parse_args()
    main(args)

