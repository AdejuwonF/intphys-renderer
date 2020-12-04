import itertools
import string
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog
from torch import nn
import torch

# from utils.constants import SHAPES
from sklearn.metrics import roc_auc_score

from utils.misc import image_based_to_annotation_based, collate_attributes

CONTINUOUS_TERMS = ["scale_x", "scale_y", "scale_z", "x", "y", "z",
                                 "yaw_radians", "pitch_radians", "roll_radians", 'radius', 'length']
CATEGORICAL_TERMS = ["shape"]

MASKABLE_TERMS = {"box": ("scale_x", "scale_y", "scale_z"),
                  "cylinder": ('radius', 'length')}

SHAPE_MAP = {"box":0,
             "cylinder":1}


def compute_mask(term: string, shape_values: torch.LongTensor) -> torch.FloatTensor:
    '''
    :param term:
    :param shape_values: 
    :param maskable_terms: a dict of tuples, keys in the  list correspond to  shape types
                       and elements in the tuples correspond to maskable terms for such shape
                       indices
    :param shape_map: a  dict of  indices where keys are shape types and values are corresponding shape indices
    :return: a vector with 1s at element i where i would be valid combination of term, shape
            (e.g. radius is valid for cylinders)
    '''
    for k in MASKABLE_TERMS.keys():
        if term in MASKABLE_TERMS[k]:
            valid_shape = SHAPE_MAP[k]
            return (shape_values == valid_shape).float()

    return torch.ones_like(shape_values, dtype=torch.float32)

def mask_out_irrelevant_values(attributes):
    for term in attributes.keys():
        mask = (compute_mask(term, torch.LongTensor(attributes["shape"]))==1.0).cpu().numpy()
        attributes[term] = np.array(attributes[term])[mask]
    return attributes




class ShapesWorldAttributes(object):

    @staticmethod
    def get_means_and_stds_from_dataset(cfg):
        if hasattr(cfg.ATTRIBUTES, "MEANS") and hasattr(cfg.ATTRIBUTES, "STDS"):
            return dict(cfg.ATTRIBUTES.MEANS), dict(cfg.ATTRIBUTES.STDS)

        print("computing means and standard deviations of dataset")
        start = time.time()
        dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in cfg.DATASETS.TEST]
        dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
        attributes = collate_attributes(dataset_dicts)
        attributes = mask_out_irrelevant_values(attributes)
        means = {k: v.mean().item() for k, v in attributes.items()}
        stds = {k: v.std().item() if v.std() > 0.01 else 1.0 for k, v in attributes.items()}
        means_cfg = CfgNode(means)
        stds_cfg = CfgNode(stds)
        cfg.ATTRIBUTES.MEANS = means_cfg
        cfg.ATTRIBUTES.STDS = stds_cfg
        print("done after {}".format(time.time() - start))

        return means, stds

    @staticmethod
    def cum_sum(sequence):
        r, s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    @staticmethod
    def absolute_error_acc(input: torch.FloatTensor, target: torch.FloatTensor, **kwargs):
        normalizing_std = kwargs["normalizing_std"]
        error = (input - target).abs()
        mean = error.mean()

        return torch.FloatTensor([mean]) / normalizing_std

    @staticmethod
    def neg_auc_score(input, target, **kwargs):
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        if len(target.unique()) == 1:
            return torch.FloatTensor([1])
        input = torch.nn.functional.softmax(input, dim=1)
        return torch.FloatTensor([1.0 - roc_auc_score(target.cpu(),input[:,1].cpu())])


    def __init__(self, cfg):
        super(ShapesWorldAttributes, self).__init__()
        self.means, self.std_deviations = self.get_means_and_stds_from_dataset(cfg)

        self.continuous_terms = CONTINUOUS_TERMS
        self.categorical_terms = CATEGORICAL_TERMS
        self.term = self.continuous_terms + self.categorical_terms

        self.maskable_terms = MASKABLE_TERMS
        self.shape_map = SHAPE_MAP

        self.loss_methods = {k: nn.MSELoss(reduction='none') for k in self.continuous_terms}
        self.loss_methods.update({k: nn.CrossEntropyLoss(reduction='none') for k in self.categorical_terms})

        self.err_method = {k: self.absolute_error_acc for k in self.continuous_terms}
        self.err_method.update({k: self.neg_auc_score for k in self.categorical_terms})

        self.value_lengths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, len(self.shape_map)]
        self.value_indices = self.cum_sum(self.value_lengths)

    def __len__(self):
        return self.value_indices[-1]

    def forward(self, input):
        x = torch.zeros((input[self.term[0]].shape[0], self.value_indices[-1])).to(input.device)
        for i, term in enumerate(self.term):
            x[:, self.value_indices[i]:self.value_indices[i + 1]] = input[term]
        return x

    def backward(self, input):
        x = {}
        for i, term in enumerate(self.term):
            x[term] = input[:, self.value_indices[i]:self.value_indices[i + 1]]
            # unstandardize output
            x[term] = x[term] * self.std_deviations[term] + self.means[term]
        return x

    def loss(self, input, target):
        loss_dict = {}
        for term, loss_method in self.loss_methods.items():

            normalizing_constant = self.std_deviations[term] if term in self.continuous_terms else 1

            loss = loss_method(input[term] / normalizing_constant,
                               target[term] / normalizing_constant)  # standardize loss

            # masking non-relevant terms (e.g. radius when shape is box)
            mask = compute_mask(term, target["shape"])
            mask = mask.view(*loss.shape)
            assert loss.shape == mask.shape
            loss = (loss*mask).sum()
            mean_denominator = mask.sum()

            loss /= mean_denominator if mean_denominator > 0 else 1.0

            loss_dict[term] = loss

        loss_dict["loss"] = sum(l for l in loss_dict.values())
        return loss_dict

    def pred_error(self, input, target):
        err_dict = {}
        err_dict['overall_mean'] = 0
        for term, err_method in self.err_method.items():
            mask = compute_mask(term, target["shape"])
            i_input = input[term][mask.view(-1)!=0]
            i_target = target[term][mask.view(-1)!=0]
            err_dict[term] = err_method(i_input, i_target,
                                        normalizing_std=self.std_deviations[term])
            err_dict['overall_mean'] += err_dict[term]
        err_dict["overall_mean"] /= len(self.term)

        return err_dict

    def cat_by_key(self, agg_dict, current):
        for term in self.term:
            current[term] = current[term] if current[term].ndim == 2 else current[term].view(-1, 1)
            agg_dict[term] = torch.cat([agg_dict[term].view(-1, current[term].size()[1]).to(current[term].dtype),
                                        current[term]],
                                       dim=0)
        return agg_dict
