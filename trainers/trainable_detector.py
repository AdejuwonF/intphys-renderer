import copy
import os

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils, build_detection_train_loader, build_detection_test_loader, \
    get_detection_dataset_dicts, DatasetFromList, MapDataset, samplers, DatasetCatalog
from detectron2.data.build import trivial_batch_collator
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.utils.registry import Registry

from utils.misc import read_image


def filter_non_coco_annotations(annos):
    return [an for an in annos if "bbox" in an] #Assumed that if there is a box there's also segmentation


class DetectionMapper():
    def __init__(self, cfg):
        self.use_dept = cfg.USE_DEPTH
    def __call__(self, dataset_dict):
        # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        img = read_image(dataset_dict["file_name"])
        if self.use_dept:
            img = torch.FloatTensor(1 / (1 + img))
            dataset_dict["image"] = img.unsqueeze(dim=0)
        else:
            img = torch.FloatTensor(img)/255.0
            dataset_dict["image"] = img
        # img = torch.FloatTensor(1 / (1 + img)) if self.use_dept else torch.FloatTensor(img)
        # dataset_dict["image"] = img.unsqueeze(dim=0)
        image_shape = (dataset_dict["height"], dataset_dict["width"])

        annos = dataset_dict["annotations"]
        annos = filter_non_coco_annotations(annos)
        instances = detection_utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        dataset_dict["instances"] = instances
        return dataset_dict

class CustomDetectTrainer(DefaultTrainer):
    def __init__(self,cfg):
        super().__init__(cfg)
        if cfg.RESUME:
            self.resume_or_load(resume=True)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, DetectionMapper(cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, DetectionMapper(cfg))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

class DetectorPredictor:
    def __init__(self,cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg).eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def __call__(self, input_batch):
        return self.model(input_batch)

def inference_detection_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_test_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)

    dataset = DatasetFromList(dataset_dicts)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator
    )
    return data_loader