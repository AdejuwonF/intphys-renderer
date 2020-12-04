import itertools
from collections import defaultdict
from functools import reduce
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from detectron2.data import DatasetCatalog
from torch import nn
from torch.utils.data import Dataset

from utils.misc import filter_dataset, CodeTimer, quantized_idx2float
from datasets import import_constants_from_dataset


def get_max_num_objects(dicts):
    max_num_objects = 0
    for d in dicts:
        num_annos = len(d["annotations"])
        if num_annos > max_num_objects:
            max_num_objects = num_annos
    return max_num_objects

class FrameRange:
    def __init__(self):
        self.max_frame = 0
        self.min_frame = 999999
    def update(self,f):
        if f>self.max_frame:
            self.max_frame = f
        if f < self.min_frame:
            self.min_frame = f

    def remove_lead_zeros(self, tensor):
        fmin,fmax,frange = self.get_range()
        out = np.zeros_like(tensor)
        out[:frange+1] = tensor[fmin:fmax+1]
        return  out

    def get_range(self):
        return self.min_frame, self.max_frame, self.max_frame-self.min_frame


class DynamicsDataset(Dataset):
    def __init__(self, cfg, datasets, attributes_key):
        import_constants_from_dataset(self, cfg.DATASETS.BASE_NAME, add_object_id=True)

        self.max_num_objects = len(self.object_id_map)

        #TODO: ADEPT has a variable number of frames
        if cfg.DATASETS.BASE_NAME == "intphys":
            self.max_num_frames = 100
        elif cfg.DATASETS.BASE_NAME == "shapes_world":
            self.max_num_frames = 50
        elif cfg.DATASETS.BASE_NAME == "adept":
            self.max_num_frames = 180 #TODO: heuristic and pretty bad I think
        else:
            raise NotImplementedError
        # required = "pred_attributes" if cfg.DATASETS.USE_PREDICTED_ATTRIBUTES else "attributes"
        required = attributes_key
        data = [self.process_dataset(dname, required, cfg) for dname in datasets]

        inputs, targets, possible_flags, original_videos = map(lambda z: reduce(lambda x,y: x+y, z),
                                                               zip(*data))

        self.inputs = inputs
        self.targets = targets
        self.possible_flags = possible_flags
        self.original_videos = original_videos
        self.value_indices = self.get_value_lengths()
        self.loss_methods = {k: nn.MSELoss(reduction='none') for k in self.continuous_terms}
        self.loss_methods.update({k: nn.CrossEntropyLoss(reduction='none') for k in self.categorical_terms})

    def process_dataset(self, dataset_name, required, cfg):
        timer = CodeTimer("building dynamics for {} with key {}".format(dataset_name,
                                                                        required))
        dataset_dicts = DatasetCatalog.get(dataset_name)
        filtered_idx, filtered_dicts = filter_dataset(dataset_dicts, [required])

        worker_args = defaultdict(lambda: {"vid_dicts": []})
        for d in filtered_dicts:
            vid_num = d["image_id"] // 500
            worker_args[vid_num]["vid_dicts"].append(d)
            worker_args[vid_num]["required"] = required

        if cfg.DEBUG:
            data = [self.process_video(w) for w in worker_args.values()]
        else:
            with Pool(int(cpu_count())) as  p:
                data = p.map(self.process_video, worker_args.values())

        inputs, targets, possible_flags, original_videos = zip(*data)
        inputs = [torch.tensor(el) for el in inputs]
        targets = [{k: torch.tensor(v) for k, v in el.items()} for el in targets]
        timer.done()
        return inputs, targets, possible_flags, original_videos

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, vid_num):
        '''
        :param vid_num:
        :return:
        dictionary with 2 keys:
            - input: dictionary with (predicted or ground truth) attributes of only visible objects, contains len(TERMS) keys:
                - <term> : dictionary with max_num keys: (not required filled with dummies)
                    -  <obj_num> : two dimmensional tensor with LENGTH_VIDS rows and dim(term) columns.
                    e.g. - for term=="location_x" each  object contains tensor with  shape==(LENGTH_VIDS,1)
                         - for term=="shape" tensor with  shape==(LENGTH_VIDS,NUM_SHAPES)
            - target: dictionary with ground  truth  attributes of objects, it has the same format as input
        '''
        return {"input": self.inputs[vid_num],
                "targets": self.targets[vid_num],
                "is_possible": self.possible_flags[vid_num],
                "original_video": self.original_videos[vid_num]}

    @property
    def input_size(self):
        return self.max_num_objects * self.inputs[0].shape[-1]

    @staticmethod
    def cum_sum(sequence):
        r, s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def get_value_lengths(self):
        cat_lengths = [len(getattr(self, "{}_map".format(term))) for term in self.categorical_terms]
        cont_lengths = [1] * (len(self.continuous_terms) + len(self.quantized_terms))
        all_lengths = self.cum_sum(cat_lengths + cont_lengths)

        self._len = all_lengths[-1]

        value_indices = {}
        for i, term in enumerate(self.terms):
            value_indices[term] = list(range(all_lengths[i], all_lengths[i + 1]))

        return value_indices

    def loss(self, predictions, targets):
        loss_dict = {}
        for term, loss_method in self.loss_methods.items():
            target = targets[term].view(-1)
            pred = predictions[term].view(-1, predictions[term].shape[-1]).squeeze()
            loss = loss_method(pred, target)
            loss_dict[term] = loss.mean()
        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict

    def input_2_dict(self, x):
        dict_input = {}
        for term in self.terms:
            dict_input[term] = x[:, :, :, self.value_indices[term]]
        return dict_input

    def dummy_input(self, term):
        if term in self.categorical_terms:
            feat_dim = len(eval("self.{}_map".format(term)))
        else:
            feat_dim = 1

        out = np.zeros((self.max_num_objects,
                        self.max_num_frames,
                        feat_dim), dtype=np.float32)

        if term == "existance":
            out[:, :, 0] = 1.0

        return out

    def dummy_target(self, term):
        if term in self.categorical_terms:
            data_type = np.long
        else:
            data_type = np.float32

        return np.zeros((self.max_num_objects,
                         self.max_num_frames),
                        dtype=data_type)

    def process_video(self, w_args):
        vid_dicts = w_args["vid_dicts"]
        required = w_args["required"]
        inputs = {k: self.dummy_input(k)
                  for k in self.terms}
        targets = {k: self.dummy_target(k)
                   for k in self.terms}
        vr = FrameRange()
        for d in vid_dicts:
            frame_num = d["image_id"] % 500
            vr.update(frame_num)
            for obj_num, an in enumerate(d["annotations"]):
                for term in self.terms:
                    if term == "existance":
                        an[required][term] = 1
                        ##########must clear default non-existance###########
                        inputs[term][obj_num, frame_num, :] = 0

                    if term == "object_id":
                        #TODO: this gets overwritten by the inference on the derender
                        #      object id should've never been in the predictions from the derender
                        an[required][term] = an["object_id"]
                        assert an["object_id"] < self.max_num_objects

                    if term in self.continuous_terms:
                        feat_num = 0
                        val = an[required][term]
                    elif term in self.quantized_terms:
                        feat_num = 0
                        val = quantized_idx2float(an[required][term], eval("self.{}_array".format(term)))
                    elif term in self.categorical_terms:
                        feat_num = an[required][term]
                        val = 1.0
                    else:
                        raise NotImplementedError

                    # categorical inputs have one hots
                    inputs[term][obj_num, frame_num, feat_num] = val

                    # categorical targets have target indices
                    targets[term][obj_num, frame_num] = an[required][term]


        # print(self.terms)
        # print([inputs[k].shape for k in self.terms])

        inputs = np.concatenate([inputs[k] for k in self.terms], axis=-1)
        # inputs = self.sort_per_frame(inputs)
        # targets = self.sort_per_frame(targets)

        # TODO: not the most elegant way but quickly fixing a bug
        inputs = vr.remove_lead_zeros(np.swapaxes(inputs, axis1=1, axis2=0))
        targets = {k: vr.remove_lead_zeros(np.swapaxes(v, axis1=0, axis2=1))
                   for k, v in targets.items()}


        return inputs, targets, vid_dicts[0]["is_possible"], vid_dicts[0]["original_video"]

    def sort_per_frame(self, x):
        if isinstance(x, dict):
            to_sort = np.stack([x[k] for k in self.terms], axis=2)
        else:
            to_sort = x

        for frame in range(self.max_num_frames):
            keys = np.rot90(to_sort[:, frame, :])
            sort_idx = np.lexsort(keys)
            if isinstance(x, dict):
                for term in self.terms:
                    x[term][:, frame] = x[term][sort_idx, frame]
            else:
                x[:, frame, :] = x[sort_idx, frame, :]

        return x
