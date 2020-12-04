import json

from detectron2.data import DatasetCatalog
from detectron2.data.datasets.coco import convert_to_coco_dict

from datasets.adept import adept_to_detectron
from datasets.ai2thor import ai2thor_intphys_to_detectron
from datasets.json_generator import JsonGenerator
from datasets.derender_inference import write_with_inferred_attributes
from datasets.detector_inference import write_with_inferred_boxes
from datasets.intphys import intphys_to_detectron
# from datasets.shapes_world import shapes_world_to_detectron
from datasets.utils import register_dataset, get_dataset_name_and_json
from utils.misc import CodeTimer, filter_dataset

RAW_PROCESSORS_MAP = {"intphys": intphys_to_detectron,
                      "adept": adept_to_detectron,
                      "ai2thor-intphys": ai2thor_intphys_to_detectron}


def build_dataset(cfg):
    data_cfg = cfg.DATA_CFG
    splits = data_cfg.SPLITS
    for split in splits:
        dataset_name, standard_format_json_file = get_dataset_name_and_json(data_cfg, split)
        # if not os.path.exists(standard_format_json_file) or data_cfg.RECOMPUTE_DATA:
        register_dataset(data_cfg, split)
        if data_cfg.REPROCESS_RAW_VIDEOS:
            raw_to_detectron = RAW_PROCESSORS_MAP[data_cfg.BASE_NAME]
            raw_to_detectron(data_cfg, split, standard_format_json_file)
            write_coco_format_json(data_cfg, split)

        if data_cfg.TRAINED_DETECTOR.DO_INFERENCE:
            write_with_inferred_boxes(data_cfg, split)

        if data_cfg.TRAINED_DERENDER.DO_INFERENCE:
            for attributes_key in data_cfg.TRAINED_DERENDER.ATTRIBUTES_WEIGHTS_MAP:
                write_with_inferred_attributes(data_cfg, split, attributes_key)

        if data_cfg.SHAPESWORLD_JSON.REPROCESS:
            for attributes_to_use in data_cfg.ATTRIBUTES_KEYS:
                JsonGenerator(data_cfg, split, "shapesworld", attributes_to_use,
                              vel_data_assoc="None")

        if data_cfg.ADEPT_JSON.REPROCESS:
            for attributes_to_use in data_cfg.ADEPT_JSON.ATTRIBUTES_KEYS:
                JsonGenerator(data_cfg, split, "adept", attributes_to_use,
                              vel_data_assoc=data_cfg.ADEPT_JSON.VEL_DATA_ASSOC)


def write_coco_format_json(cfg, split):
    timer = CodeTimer("writting to coco")
    dataset_name, standard_format_json_file = get_dataset_name_and_json(cfg, split)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    _,filtered_dicts = filter_dataset(dataset_dicts, required_fields=["bbox", "bbox_mode", "segmentation"])
    register_dataset(cfg, split, getter= lambda: filtered_dicts, name=dataset_name+"_for_coco")

    coco_dict = convert_to_coco_dict(dataset_name+"_for_coco")

    json_format_file = standard_format_json_file.replace(".json", "_coco_format.json")
    with open(json_format_file, "w") as f:
        json.dump(coco_dict, f)
    timer.done()


