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
from collections import defaultdict


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
        # loses a lot of the functionaluty it had.
        self.attributes = DerenderAttributes(cfg.MODULE_CFG)

        self.mapper = trainers.trainable_derender.DerenderMapper(cfg.MODULE_CFG.DATASETS.USE_PREDICTED_BOXES,
                            self.attributes,
                            False, #for_inference,
                            use_depth=cfg.MODULE_CFG.DATASETS.USE_DEPTH)
        #self.dataset_dicts = list(map(self.mapper, self.dataset_dicts))
        print("done after {}".format(time.time()-start))


    def __len__(self):
        return len(self.dataset_dicts)


    def __getitem__(self, idx):
        # return attributesToTensor(self.mapper(self.dataset_dicts[idx])["attributes"])
        return attributesToTensor(self.dataset_dicts[idx]["attributes"])


def attributesToTensor(attr_dict):
    # Continuos terms (treat quantized as continuous for now)
      cont_terms = torch.tensor([attr_dict[term] for term in intphys.CONTINUOUS_TERMS])
      quant_terms = torch.tensor([attr_dict[term] for term in intphys.QUANTIZED_TERMS])
    # Categoric terms
      exists = torch.zeros(2)
      visible = torch.zeros(2)
      obj_type = torch.zeros(6)
      shape = torch.zeros(3)
      obj_id = torch.zeros(10)

      exists[attr_dict["existance"]] = 1
      visible[attr_dict["visible"]] = 1
      obj_type[attr_dict["type"]] = 1
      shape[attr_dict["shape"]] = 1
      obj_id[attr_dict["object_id"]] = 1
      # continuous terms
      # positions 0-9
      # categorical terms, 10-11, 12-13, 14-19, 20-22, 23-32
      return torch.cat([cont_terms, quant_terms, exists, visible, obj_type, shape, obj_id])

def tensorToAttributes(tensor):
    attributes = {intphys.CONTINUOUS_TERMS[i]:tensor[i] for i in range(len(intphys.CONTINUOUS_TERMS))}
    attributes["rotation_yaw"] = tensor[9]
    attributes["existance"] = torch.argmax(tensor[10:12])
    attributes["visible"] = torch.argmax(tensor[12:14])
    attributes["type"] = torch.argmax(tensor[14:20])
    attributes["shape"] = torch.argmax(tensor[20:23])
    attributes["object_id"] = torch.argmax(tensor[23:33])
    return attributes

class DatasetUtils(object):
    def __init__(self, val_data):
        self.val_data = val_data
        self.MSE = nn.MSELoss()
        self.CE = nn.CrossEntropyLoss()
        self.set_means_stds()
        self.eval_res = None
        return
    def denormalize(self, tensor):
        return tensor*self.stds + self.means
    def normalize(self, tensor):
        return (tensor-self.means)/self.stds
    def set_means_stds(self):
        start = time.time()
        all_attrs = torch.zeros((33, len(self.val_data)))
        with torch.no_grad():
            for i, attrs in enumerate(self.val_data):
                all_attrs[:,i] = attrs

        self.means = all_attrs.mean(dim=1)
        self.means[10:] = 0
        self.stds = all_attrs.std(dim=1)
        self.stds[10:] = 1
        print("Set means and std in {0} seconds".format(time.time()-start))
        return
    def loss(self, predictions, targets):
        loss = 0
        targets = self.normalize(targets)
        #Continuous Terms
        cont_pred = predictions[:, :10]
        cont_target = targets[:, :10]

        loss += self.MSE(cont_pred, cont_target)

        #Categorical Terms
        exists_pred = predictions[:, 10:12]
        exists_target = torch.argmax(targets[:, 10:12], dim=1)
        loss += self.CE(exists_pred, exists_target)

        visible_pred = predictions[:, 12:14]
        visible_target = torch.argmax(targets[:, 12:14], dim=1)
        loss += self.CE(visible_pred, visible_target)

        obj_type_pred = predictions[:, 14:20]
        obj_type_target = torch.argmax(targets[:, 14:20], dim=1)
        loss += self.CE(obj_type_pred, obj_type_target)

        shape_pred = predictions[:, 20:23]
        shape_target = torch.argmax(targets[:, 20:23], dim=1)
        loss += self.CE(shape_pred, shape_target)

        obj_id_pred = predictions[:, 23:]
        obj_id_target = torch.argmax(targets[:, 23:], dim=1)
        loss += self.CE(obj_id_pred, obj_id_target)

        return loss


    def eval(self, model, dataset):
        start = time.time()
        with torch.no_grad():
            metrics = defaultdict(int)
            for i, attrs in enumerate(dataset):
                attr_dict = tensorToAttributes(self.normalize(attrs))
                pred_dict = tensorToAttributes(model.forward(attrs))
                for term in attr_dict:
                    if term in (intphys.CONTINUOUS_TERMS + intphys.QUANTIZED_TERMS):
                        metrics[term] += abs((attr_dict[term] - pred_dict[term]).item())
                    else:
                        metrics[term] += (attr_dict[term] == pred_dict[term]).item()
            for term in metrics:
                metrics[term] /= len(dataset)
        self.eval_res = metrics
        print("Evaluation completed in {0} seconds".format(time.time()-start))
        return self.eval_res


class AttrLoss(object):
    def __init__(self, val_data=None):
        self.MSE = nn.MSELoss()
        self.CE = nn.CrossEntropyLoss()

    def __call__(self, predictions, targets):
        loss = 0
        #Continuous Terms
        cont_pred = predictions[:, :10]
        cont_target = targets[:, :10]

        cont_pred = cont_pred * torch.tensor([1,1,1,1,1,1,1,1,.01,1])
        cont_target = cont_target * torch.tensor([1,1,1,1,1,1,1,1,.01,1])
        loss += self.MSE(cont_pred, cont_target)

        #Categorical Terms
        exists_pred = predictions[:, 10:12]
        exists_target = torch.argmax(targets[:, 10:12], dim=1)
        loss += self.CE(exists_pred, exists_target)

        visible_pred = predictions[:, 12:14]
        visible_target = torch.argmax(targets[:, 12:14], dim=1)
        loss += self.CE(visible_pred, visible_target)

        obj_type_pred = predictions[:, 14:20]
        obj_type_target = torch.argmax(targets[:, 14:20], dim=1)
        loss += self.CE(obj_type_pred, obj_type_target)

        shape_pred = predictions[:, 20:23]
        shape_target = torch.argmax(targets[:, 20:23], dim=1)
        loss += self.CE(shape_pred, shape_target)

        obj_id_pred = predictions[:, 23:]
        obj_id_target = torch.argmax(targets[:, 23:], dim=1)
        loss += self.CE(obj_id_pred, obj_id_target)

        return loss

def main(args):
    cfg = setup_cfg(args, args.distributed)
    #print(cfg)
    dataset =  IntphysJsonTensor(cfg, "_val")
    # train_dataset = IntphysJsonTensor(cfg, "_train")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)
    utils = DatasetUtils(dataset)
    return dataloader

if __name__ == "__main__":
    args = parse_args()
    main(args)

