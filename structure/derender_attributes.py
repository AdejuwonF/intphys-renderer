import itertools
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from torch import nn

from utils.misc import CodeTimer, to_cuda, quantized_onehot2floats_batch, quantized_idx2floats_batch, image_based_to_annotation_based
from datasets import import_constants_from_dataset, utils
# from datasets.intphys import _ROTATION_YAW_DISCRETE


class DerenderAttributes:
    @staticmethod
    def absolute_error_acc(input, target, **kwargs):
        normalizing_std = kwargs["normalizing_std"]
        error = np.abs(input - target)
        mean = error.mean().item()

        return mean / normalizing_std

    @staticmethod
    def cum_sum(sequence):
        r, s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    @staticmethod
    def neg_auc_score(input, target, **kwargs):
        unique_targets = np.sort(np.unique(target))
        if len(unique_targets) == 1 or len(input) == 0:
            return 1.0
        # assert ((np.arange(len(unique_targets))) == np.sort(unique_targets)).all()
        target_map = np.zeros(unique_targets.max()+1,dtype=np.int) - 1
        target_map[unique_targets] = np.arange(len(unique_targets))
        target = target_map[target]
        input = softmax(input[:,unique_targets], axis=1)
        if input.shape[1] == 2:
            input = input[:,1]
        return 1.0 - roc_auc_score(target, input, average="macro", multi_class='ovo')

    @staticmethod
    def rotation_loss(input,target):
        diff = (input - target)
        diff_plus_2pi = (diff + 2*np.pi).abs()
        diff_minus_2pi = (diff - 2*np.pi).abs()
        all = torch.stack([diff.abs(),
                           diff_minus_2pi,
                           diff_plus_2pi], dim=0)
        loss, idx = all.min(dim=0,keepdim=False)
        return loss

    def __init__(self, cfg):
        import_constants_from_dataset(self, cfg.ATTRIBUTES.NAME)
        """self.set_means_and_stds(cfg)

        self.loss_methods = {k: nn.MSELoss(reduction='none') for k in self.continuous_terms}
        self.loss_methods.update({k: nn.CrossEntropyLoss(reduction='none')
                                  for k in self.categorical_terms + self.quantized_terms})
        self.loss_methods.update({k: self.rotation_loss for k in self.rotation_terms})

        self.err_method = {k: self.absolute_error_acc for k in self.continuous_terms + self.quantized_terms}
        self.err_method.update({k: self.neg_auc_score for k in self.categorical_terms})

        self.value_indices = self.get_value_lengths()"""

    def backward(self, input):
        x = {}
        for term in self.terms:
            x[term] = input[:, self.value_indices[term]]
            if term in self.positive_terms:
                # make positive if required
                x[term] = torch.exp(x[term])
            elif term not in self.rotation_terms and term in self.continuous_terms:
                # unstandardize output
                x[term] = x[term] * self.std_deviations[term] + self.means[term]
        return x

    def loss(self, input, target):
        loss_dict = {}
        for term, loss_method in self.loss_methods.items():
                                                            # and term not in self.ranged_terms \
            normalize = term in self.continuous_terms \
                        and term not in self.positive_terms \
                        and term not in self.rotation_terms
            # normalizing_constant = self.std_deviations[term] if term in self.continuous_terms \
            #                                                  and term not in self.positive_terms \
            #                                                  and term not in self.rotation_terms \
            #                        else 1
            l_input, l_target = map(lambda z: z/self.std_deviations[term] if normalize
                                              else z,
                                    [input[term],target[term]])

            loss = loss_method(l_input, l_target)

            # masking non-relevant terms (e.g. radius when shape is box)
            mask = self.compute_mask(target, term)
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
            mask = self.compute_mask(target, term)
            i_input = input[term][mask.view(-1).cpu()!=0]
            i_target = target[term][mask.view(-1).cpu()!=0]
            if term in self.quantized_terms:
                i_input = quantized_onehot2floats_batch(i_input, eval("self.{}_array".format(term)))
                i_target = quantized_idx2floats_batch(i_target, eval("self.{}_array".format(term)))

            err_dict[term] = err_method(i_input, i_target,
                                        normalizing_std=self.std_deviations[term])

            err_dict['overall_mean'] += err_dict[term]
        err_dict["overall_mean"] /= len(self.terms)
        return err_dict

    def predict(self,input):
        predictions = {}
        for term, values in input.items():
            if term in self.categorical_terms + self.quantized_terms:
                values = values.argmax(axis=1)
            predictions[term] = values
        return predictions

    def get_value_lengths(self):
        lengths = []
        for term in self.terms:
            if term in self.categorical_terms:
                lengths.append(len(getattr(self,"{}_map".format(term))))
            elif term in self.quantized_terms:
                lengths.append(len(getattr(self, "{}_array".format(term))))
            elif term in self.continuous_terms:
                lengths.append(1)
        # cat_lengths = [len(getattr(self,"{}_map".format(term))) for term in self.categorical_terms]
        # quantized_lenghts = [len(getattr(self,"{}_array".format(term))) for term in self.quantized_terms]
        # cont_lengths = [1] * len(self.continuous_terms)
        all_lengths  = self.cum_sum(lengths)

        self._len = all_lengths[-1]

        value_indices = {}
        for i,term in enumerate(self.terms):
            value_indices[term] = list(range(all_lengths[i],all_lengths[i+1]))

        return value_indices


    def set_means_and_stds(self, cfg):
        ######after computing them they should be saved to cfg so the model get's properly loaded######
        if hasattr(cfg.ATTRIBUTES, "MEANS") and hasattr(cfg.ATTRIBUTES, "STDS"):
            means, stds = dict(cfg.ATTRIBUTES.MEANS), dict(cfg.ATTRIBUTES.STDS)
            mins, maxes = 0,0
        else:
            timer = CodeTimer("computing means and standard deviations of dataset")
            #######compute the means and stds only on the validation  set#############
            dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in cfg.DATASETS.TEST if
                             "_val" in dataset_name]
            #Get dataset dict without detectrono
            """data_cfg = self.data_cfg
            for dataset_name in cfg.DATASETS.TEST:
                if "_val" in dataset_name:
                    num_frames = utils.get_num_frames(data_cfg, "_val")
                    dataset_name, standard_format_json_file = utils.get_dataset_name_and_json(data_cfg, "_val")
                    dataset_dicts = utils.get_data_dicts(standard_format_json_file, num_frames)

            required_fields = ["pred_box"] if cfg.DATASETS.USE_PREDICTED_BOXES else ["bbox"]
            required_fields += ["attributes"]
            _, dataset_dicts = image_based_to_annotation_based(dataset_dicts, required_fields)"""

            dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
            attributes = self.collate_attributes(dataset_dicts)
            attributes = self.mask_out_irrelevant_values(attributes)
            means = {k: v.mean().item() if k in self.continuous_terms else 0.0 for k, v in attributes.items()}
            stds = {k: v.std().item() if v.std() > 0.01 and k in self.continuous_terms else 1.0
                    for k, v in attributes.items()}
            maxes  = {k: v.max().item() if k in self.continuous_terms else 0.0
                     for k, v in attributes.items()}
            mins = {k: v.min().item() if k in self.continuous_terms else 0.0
                     for k, v in attributes.items()}
            means_cfg = CfgNode(means)
            stds_cfg = CfgNode(stds)
            cfg.ATTRIBUTES.MEANS = means_cfg
            cfg.ATTRIBUTES.STDS = stds_cfg
            timer.done()

        self.means = means
        self.std_deviations = stds
        self.maxes =  maxes
        self.mins = mins

    def add_camera(self, backbone_out,inputs):
        camera = torch.cat([inputs["camera"][c] for c in self.camera_terms], dim=1)
        out_with_camera =  torch.cat([backbone_out,camera], dim=1)
        return out_with_camera

    def mask_out_irrelevant_values(self,attributes):
        new_attributes = {}
        for term,vector in attributes.items():
            mask = self.compute_mask(attributes,term)
            mask = mask.view(*vector.shape).cpu().numpy() == 1.0
            new_attributes[term] = np.array(vector)[mask]

        return new_attributes

    def compute_mask(self, attributes, term):
        attributes = to_cuda(attributes)
        if term in self.valid_map:
            all_masks = []
            for val_term, valid_els in self.valid_map[term].items():
                categories_map = eval("self.{}_map".format(val_term))
                valid_categories = torch.LongTensor([categories_map[el] for el in valid_els])

                val_vector = attributes[val_term].view(-1,1).repeat(1,len(valid_categories)).cuda()
                valid_categories = valid_categories.repeat(len(val_vector),1).cuda()

                mask = ((val_vector - valid_categories).abs().min(dim=1).values == 0).float()
                all_masks.append(mask)
            return reduce(lambda x,y: x*y, all_masks)
        else:
            return torch.ones(len(attributes[term]),dtype=torch.float).cuda()

    def collate_attributes(self,data_dicts):
        collated_dict = defaultdict(list)

        for d in data_dicts:
            for an in d["annotations"]:
                if "attributes" in an:
                    for term, val in an["attributes"].items():
                        collated_dict[term].append(val)

        return {k: torch.FloatTensor(v) if k in self.continuous_terms
                else torch.LongTensor(v)
                for  k, v in collated_dict.items()}

    def __len__(self):
        return self._len





