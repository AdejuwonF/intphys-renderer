from detectron2.config import CfgNode


def get_derender_config(dataset_name):
    _C = CfgNode()
    _C.DEBUG = True
    _C.TYPE = "derender"
    _C.ATTRIBUTES = CfgNode()

    _C.RESUME = False
    _C.RESUME_DIR = ""
    _C.SEED = -1
    _C.CUDNN_BENCHMARK = False
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    _C.MODEL = CfgNode()
    # Number of derender visual feature channels
    _C.MODEL.FEATURE_CHANNELS = 512
    # number of hidden layers after backbone
    _C.MODEL.NUM_MID_LAYERS = 2
    # Number of intermediate layer channels
    _C.MODEL.MID_CHANNELS = 256

    _C.INPUT = CfgNode()

    _C.DATASETS = CfgNode()
    _C.DATASETS.USE_PREDICTED_BOXES = False
    _C.DATASETS.BASE_NAME = dataset_name

    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    _C.DATALOADER = CfgNode()
    # Number of data loading threads
    if _C.DEBUG:
        _C.DATALOADER.NUM_WORKERS = 0
    else:
        _C.DATALOADER.NUM_WORKERS = 6

    if _C.DEBUG:
        _C.DATALOADER.VAL_BATCH_SIZE = 80
    else:
        _C.DATALOADER.VAL_BATCH_SIZE = 160

    # ---------------------------------------------------------------------------- #
    # Solver
    # ---------------------------------------------------------------------------- #
    _C.SOLVER = CfgNode()

    #maximum number of seconds for training
    _C.SOLVER.MAX_TIME_SECS = 43200
    # Print metrics every _ seconds
    _C.SOLVER.PRINT_METRICS_TIME = 180 #TODO: set back to 180
    # write to tensorboard summary every_ secconds
    _C.SOLVER.TENSORBOARD_SECS = 1
    # Checkpoint every _ seconds
    _C.SOLVER.CHECKPOINT_SECS = 600
    # Run validation every _ seconds
    _C.SOLVER.VALIDATION_SECS = 600 #TODO: change back to 300
    _C.SOLVER.VALIDATION_MAX_SECS = 60


    _C.SOLVER.BASE_LR = 6.658777172739463e-5
    _C.SOLVER.BIAS_LR_FACTOR = 2


    _C.SOLVER.OPT_TYPE = "Adam"  #options "Adam" "SGD"
    _C.SOLVER.MOMENTUM = 0.9960477666835778 #found via Bayesian Optimization
    _C.SOLVER.ADAM_BETA = 0.9999427846237621

    _C.SOLVER.WEIGHT_DECAY = 0.0005
    _C.SOLVER.WEIGHT_DECAY_BIAS = 0

    #Factor of reduction at  iteration == el for el in SOLVER.STEPS
    _C.SOLVER.GAMMA = 0.3
    _C.SOLVER.STEPS = (80000, 100000)
    # _C.SOLVER.STEPS = (3000,5000)

    _C.SOLVER.WARMUP_FACTOR = 1.0 / 3
    _C.SOLVER.WARMUP_ITERS = 500
    _C.SOLVER.WARMUP_METHOD = "linear"

    ######### INTPHYS ###############

    if _C.DATASETS.BASE_NAME == "intphys":

        _C.DATASETS.TRAIN = ("intphys_train",)
        # _C.DATASETS.TRAIN = ("intphys_val",)
        _C.DATASETS.TEST = ("intphys_val",)
        _C.DATASETS.USE_DEPTH = True

        _C.DATALOADER.OBJECTS_PER_BATCH = 160  # TODO: get back to 160

        _C.ATTRIBUTES.NAME = "intphys"
        _C.MODEL.ADD_CAMERA = True

        # Input channels for the model (segmented depth map, depth map)
        _C.MODEL.IN_CHANNELS = 2
        # The size of pooling kernel in the last layer of the resnet34
        _C.MODEL.POOLING_KERNEL_SIZE = (8, 8)


    ####### ADEPT #################
    elif _C.DATASETS.BASE_NAME == "adept":
        _C.DATASETS.TRAIN = ("adept_train",)
        # _C.DATASETS.TRAIN = ("adept_val",)
        _C.DATASETS.TEST = ("adept_val",)
        _C.DATASETS.USE_DEPTH = False
        _C.ATTRIBUTES.NAME = "adept"
        _C.MODEL.ADD_CAMERA = False

        if _C.DEBUG:
            _C.DATALOADER.OBJECTS_PER_BATCH = 20  # TODO: get back to 40
        else:
            _C.DATALOADER.OBJECTS_PER_BATCH = 120  # TODO: get back to 40
        # Input channels for the model (segmented depth map, depth map)
        _C.MODEL.IN_CHANNELS = 12
        # The size of pooling kernel in the last layer of the resnet34
        _C.MODEL.POOLING_KERNEL_SIZE = (10, 15)

    elif _C.DATASETS.BASE_NAME == "ai2thor-intphys":
        _C.DATASETS.TRAIN = ("ai2thor-intphys_train",)
        # _C.DATASETS.TRAIN = ("ai2thor-intphys_val",)
        _C.DATASETS.TEST = ("ai2thor-intphys_val",)
        _C.DATASETS.USE_DEPTH = True
        _C.ATTRIBUTES.NAME = "ai2thor"
        _C.MODEL.ADD_CAMERA = False

        if _C.DEBUG:
            _C.DATALOADER.OBJECTS_PER_BATCH = 20
        else:
            _C.DATALOADER.OBJECTS_PER_BATCH = 120

        # Input channels for the model (segmented depth map, depth map)
        _C.MODEL.IN_CHANNELS = 2
        # The size of pooling kernel in the last layer of the resnet34
        _C.MODEL.POOLING_KERNEL_SIZE = (8, 16)
    else:
        raise NotImplementedError

    return _C


