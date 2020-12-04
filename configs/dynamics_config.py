from detectron2.config import CfgNode

def get_dynamics_cfg(dataset_base_name):
    _C  = CfgNode()
    _C.DEBUG = False
    _C.TYPE = "dynamics"

    _C.DATASETS  = CfgNode()
    # _C.DATASETS.USE_PREDICTED_ATTRIBUTES = True
    _C.DATASETS.BASE_NAME = dataset_base_name

    _C.DATALOADER = CfgNode()
    _C.DATALOADER.NUM_WORKERS = 4
    _C.DATALOADER.BATCH_SIZE = 15
    _C.MODEL = CfgNode()
    _C.MODEL.ARCHITECTURE = "interaction"
    _C.MODEL.RNN_NUM_LAYERS = 6
    _C.MODEL.DROP_OUT = 0.2
    _C.MODEL.HIDDEN_SIZE= 300

    _C.SOLVER = CfgNode()
    _C.SOLVER.BASE_LR = 0.001
    _C.SOLVER.BIAS_LR_FACTOR = 2
    _C.SOLVER.OPT_TYPE = "Adam"
    _C.SOLVER.MOMENTUM  = 0.996
    _C.SOLVER.ADAM_BETA = 0.9999427846237621
    _C.SOLVER.WEIGHT_DECAY = 0.0005
    _C.SOLVER.WEIGHT_DECAY_BIAS = 0

    _C.SOLVER.GAMMA = 0.5
    _C.SOLVER.STEPS = (999999, 999999)

    _C.SOLVER.WARMUP_FACTOR = 1.0 / 3
    _C.SOLVER.WARMUP_ITERS = 500
    _C.SOLVER.WARMUP_METHOD = "linear"

    _C.SOLVER.MAX_TIME_SECS = 999999999
    # Print metrics every _ seconds
    _C.SOLVER.PRINT_METRICS_TIME = 180 #TODO: set back to 180
    # write to tensorboard summary every_ secconds
    _C.SOLVER.TENSORBOARD_SECS = 1
    # Checkpoint every _ seconds
    _C.SOLVER.CHECKPOINT_SECS = 1200
    # Run validation every _ seconds
    _C.SOLVER.VALIDATION_SECS = 60 #TODO: change back to 300
    _C.SOLVER.VALIDATION_MAX_SECS = 9999

    if _C.DATASETS.BASE_NAME == "intphys":
        _C.DATASETS.TRAIN = ("intphys_val",)
        _C.DATASETS.TEST = ("intphys_dev_O1",
                            "intphys_dev_O2",
                            "intphys_dev_O3",
                            # "intphys_dev-meta_O1",
                            # "intphys_dev-meta_O2",
                            # "intphys_dev-meta_O3"
                            )
        _C.ATTRIBUTES_KEYS = ("pred_attr_401469",
                              "pred_attr_003227")
    elif _C.DATASETS.BASE_NAME == "adept":
        _C.ATTRIBUTES_KEYS = ("attributes",
                              # "pred_attr_00650",
                              "pred_attr_43044",
                              # 'pred_attr_10580',
                              # "pred_attr_18377",
                              # "pred_attr_34216"
                              )
        _C.DATASETS.TRAIN = ("adept_train",)
        _C.DATASETS.TEST = (
            "adept_val",
            # "adept_train",
            # "adept_human_create",
            # "adept_human_vanish",
            # "adept_human_short-overturn",
            # "adept_human_long-overturn",
            # "adept_human_visible-discontinuous",
            # "adept_human_invisible-discontinuous",
            # "adept_human_delay",
            # "adept_human_block",
        )
    return _C
