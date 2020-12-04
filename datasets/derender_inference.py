import json
import os

import torch
from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset
from torch.utils.data import DataLoader

from configs.main import load_cfg_from_file
from datasets.utils import get_dataset_name_and_json, fix_for_serialization
from trainers.trainable_derender import DerenderPredictor, DerenderMapper
from utils.misc import CodeTimer, image_based_to_annotation_based, to_cuda


def write_with_inferred_attributes(cfg, split, attributes_key):
    timer = CodeTimer("adding inferred attributes split:{}, attributes_key:{}".format(split,  attributes_key))
    module_cfg = os.path.join(cfg.TRAINED_DERENDER.EXP_DIR, "cfg.yaml")
    module_cfg = load_cfg_from_file(module_cfg)
    module_cfg.MODEL.WEIGHTS = cfg.TRAINED_DERENDER.ATTRIBUTES_WEIGHTS_MAP[attributes_key]

    module_cfg.DATALOADER.OBJECTS_PER_BATCH = 1000 if cfg.BASE_NAME == "intphys" else 450
    module_cfg.DATALOADER.NUM_WORKERS = 8 if cfg.BASE_NAME == "adept" else module_cfg.DATALOADER.NUM_WORKERS

    if cfg.DEBUG:
        module_cfg.DATALOADER.NUM_WORKERS = 0
        module_cfg.DEBUG = True
        module_cfg.DATALOADER.OBJECTS_PER_BATCH = 50

    predictor = DerenderPredictor(module_cfg)

    # if not cfg.DEBUG:
    #     gpu_ids = [_ for _ in range(torch.cuda.device_count())]
    #     predictor.derenderer = torch.nn.parallel.DataParallel(predictor.derenderer, gpu_ids)

    dataset_name, standard_format_json_file = get_dataset_name_and_json(cfg, split)
    dataset = DatasetCatalog.get(dataset_name)
    required_fields = ["pred_box"] if cfg.TRAINED_DERENDER.USE_INFERRED_BOXES else ["bbox"]
    filtered_idx, \
    mapped_dataset = image_based_to_annotation_based(dataset, required_fields)
    mapped_dataset = DatasetFromList(mapped_dataset, copy=False)
    mapper = DerenderMapper(cfg.TRAINED_DERENDER.USE_INFERRED_BOXES,
                            predictor.attributes,
                            for_inference=True,
                            use_depth=cfg.TRAINED_DERENDER.USE_DEPTH)
    mapped_dataset = MapDataset(mapped_dataset, mapper)

    data_loader = DataLoader(dataset=mapped_dataset, batch_size=module_cfg.DATALOADER.OBJECTS_PER_BATCH,
                              num_workers=module_cfg.DATALOADER.NUM_WORKERS, shuffle=False)

    fil_pointer = 0
    with torch.no_grad():
        for inputs in data_loader:
            inputs = to_cuda(inputs)
            outputs = predictor(inputs)
            batch_size = list(outputs.values())[0].shape[0]
            for oix,(img_idx, an_idx) in  zip(range(batch_size),
                                              filtered_idx[fil_pointer:
                                                           fil_pointer+batch_size]):

                dataset[img_idx]["annotations"][an_idx][attributes_key] = \
                    {k: v[oix].item() for k, v in outputs.items()}
                    # {k: v[oix].item() if v[oix].size == 1
                    #                   else [float(el) for el in v[oix]]
                    # for k,v in outputs.items()}

            fil_pointer = fil_pointer+batch_size

    dataset = [fix_for_serialization(d) for d in dataset]

    with open(standard_format_json_file, "w") as f:
        json.dump(dataset, f, indent=4)

    timer.done()