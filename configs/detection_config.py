from detectron2 import model_zoo
from detectron2.config import CfgNode
from detectron2.config import get_cfg as detectron_get_cfg

def get_detection_cfg(dataset_name):
    _C = detectron_get_cfg()
    _C.DEBUG = False #TODO: should be false
    _C.TYPE = "detector"

    ###### if True the detector will  be started from whatever is in MODEL.WEIGHTS, if no checkpoint will load from imagenet
    _C.RESUME = True #TODO: should be true
    ###### if != this directory's "last_checkpoint" will be used to resume the training
    # _C.RESUME_DIR = 'output/adept/detector/exp_00011'
    # _C.RESUME_DIR = "output/intphys/detector/distributed_exp"
    _C.RESUME_DIR = ''

    _C.USE_DEPTH = True
    _C.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # _C.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training cd ..initialize from model zoo
    _C.MODEL.WEIGHTS = "output/intphys/detector/bk2_distributed_exp/model_0014999.pth"

    _C.TEST.EVAL_PERIOD = 2000
    _C.DATALOADER.NUM_WORKERS = 0 if _C.DEBUG else 4

    _C.SOLVER.WARMUP_FACTOR = 1.0 / 100
    _C.SOLVER.WARMUP_ITERS = 100

    _C.SOLVER.IMS_PER_BATCH = 16 if not _C.DEBUG else 8
    _C.SOLVER.BASE_LR = 0.0005  # pick a good LR
    _C.SOLVER.MAX_ITER = 80000
    _C.SOLVER.STEPS = (10000, 20000, 30000, 40000)
    _C.SOLVER.GAMMA = 0.3

    # _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (some_object)

    _C.INPUT.MASK_FORMAT = "bitmask"

    if dataset_name == "adept":
        _C.INPUT.FORMAT = "DDD"
        _C.DATASETS.TRAIN = ("adept_train",)
        # _C.DATASETS.TRAIN = ("adept_val",)
        _C.DATASETS.TEST = ("adept_val",)
    elif dataset_name == "intphys":
        _C.INPUT.FORMAT = "D"
        _C.DATASETS.TRAIN = ("intphys_train",) #TODO: train on train not _val
        # _C.DATASETS.TRAIN = ("intphys_val",)
        _C.DATASETS.TEST = ("intphys_val",)
    else:
        raise NotImplementedError
    return _C
