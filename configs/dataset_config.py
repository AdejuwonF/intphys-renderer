from detectron2.config import CfgNode

def data_get_cfg(dataset_base_name):
    _C = CfgNode()
    _C.DEBUG = False
    _C.DEBUG_VIDEOS = []
        # ['/disk1/mcs-data/intphys_scenes_dumped_perception/gravity_goal-0247',
        #                '/disk1/mcs-data/intphys_scenes_dumped_perception/gravity_goal-0489']
    _C.MAX_VIDEOS = 10000
    _C.VAL_VIDEOS = 130
    _C.VAL_FRAMES = -1
    _C.TEST_FRAMES = -1

    _C.BASE_NAME = dataset_base_name #options intphys, adept, ai2thor-intphys
    _C.REPROCESS_RAW_VIDEOS = False

    _C.TRAINED_DETECTOR = CfgNode()
    _C.TRAINED_DETECTOR.DO_INFERENCE = False

    _C.TRAINED_DERENDER = CfgNode()
    _C.TRAINED_DERENDER.DO_INFERENCE = False
    _C.TRAINED_DERENDER.USE_INFERRED_BOXES = False

    _C.SHAPESWORLD_JSON = CfgNode()
    _C.SHAPESWORLD_JSON.REPROCESS = False

    _C.ADEPT_JSON = CfgNode()
    _C.ADEPT_JSON.REPROCESS = False

    if _C.BASE_NAME == "intphys":
        # _C.DATA_LOCATION = "/all/home/aldo/data/intphys_data/"
        # _C.DATA_LOCATION = "/nobackup/users/aldopa/data/intphys.zip/"
        _C.DATA_LOCATION = "/disk1/intphys_data"
        _C.VAL_FRAMES = 1000
        # _C.SPLITS = ("_val","_train")
        # _C.SPLITS = ("_dev-meta_O1", "_dev-meta_O2", "_dev-meta_O3", "_val", "_train")
        # _C.SPLITS = ("_dev-meta_O1", "_dev-meta_O2", "_dev-meta_O3", "_val")
        _C.SPLITS = (
                    "_val",
                    # "_dev_O1", "_dev_O2", "_dev_O3",
                    # "_dev-meta_O1", "_dev-meta_O2", "_dev-meta_O3",
                   # "_test_O1", "_test_O2", "_test_O3",
                    "_train",
                   )
        _C.SHAPESWORLD_JSON.FRAMES_RANGE_PER_VIDEO = (0, 100)
        # _C.ATTRIBUTES_KEYS = ("attributes",
        #                       "pred_attr_401469",
        #                       "pred_attr_003227")

        _C.ADEPT_JSON.VEL_DATA_ASSOC = "heuristic"  # options: ground_truth, heuristic, None
        _C.ADEPT_JSON.ATTRIBUTES_KEYS = (
                                         # "attributes",
                                         "pred_attr_82649",
                                         "pred_attr_03227",
                                         "pred_attr_06472",
                                         "pred_attr_12664",
                                         "pred_attr_22318",
                                         "pred_attr_41337",
                                         )

        _C.MIN_AREA = 25

        _C.TRAINED_DERENDER.EXP_DIR = "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/"
        _C.TRAINED_DERENDER.ATTRIBUTES_WEIGHTS_MAP = CfgNode({
                                                              "pred_attr_82649": "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/model_0082649.pth",
                                                              "pred_attr_03227": "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/model_0003227.pth",
                                                              "pred_attr_06472": "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/model_0006472.pth",
                                                              "pred_attr_12664": "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/model_0012664.pth",
                                                              "pred_attr_22318": "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/model_0022318.pth",
                                                              "pred_attr_41337": "/all/home/aldo/cora-derenderer/output/intphys/derender/exp_00000/model_0041337.pth",
                                                              })
        _C.TRAINED_DERENDER.USE_DEPTH = True

    elif _C.BASE_NAME == "adept":
        _C.DATA_LOCATION = "/all/home/aldo/data/adept_data"
        # _C.DATA_LOCATION = "/nobackup/users/aldopa/data/adept.zip"
        _C.SPLITS = (
                     # "_val",
                     # "_train",
                     "_human_create",
                     "_human_vanish",
                     "_human_short-overturn",
                     "_human_long-overturn",
                     "_human_visible-discontinuous",
                     "_human_invisible-discontinuous",
                     "_human_delay",
                     "_human_block",
        )
        _C.VAL_FRAMES = 1000

        _C.MIN_AREA = 100

        _C.TRAINED_DETECTOR.EXP_DIR = "output/adept/detector/exp_00011"
        _C.TRAINED_DETECTOR.WEIGHTS_FILE = "output/adept/detector/exp_00011/model_0209999.pth"


        _C.TRAINED_DERENDER.EXP_DIR = "/all/home/aldo/cora-derenderer/output/adept/derender/exp_00007"
        _C.TRAINED_DERENDER.ATTRIBUTES_WEIGHTS_MAP = CfgNode({
            # "pred_attr_00650": "/all/home/aldo/cora-derenderer/output/adept/derender/exp_00007/model_0000650.pth",
            "pred_attr_43044": "/all/home/aldo/cora-derenderer/output/adept/derender/exp_00007/model_0043044.pth",
            # "pred_attr_10580": "/all/home/aldo/cora-derenderer/output/adept/derender/exp_00007/model_0010580.pth",
            # "pred_attr_18377": "/all/home/aldo/cora-derenderer/output/adept/derender/exp_00007/model_0018377.pth",
            # "pred_attr_34216": "/all/home/aldo/cora-derenderer/output/adept/derender/exp_00007/model_0034216.pth",
        })
        _C.TRAINED_DERENDER.USE_DEPTH = False

        _C.SHAPESWORLD_JSON.FRAMES_RANGE_PER_VIDEO = (4, 124)
        _C.ADEPT_JSON.VEL_DATA_ASSOC = "None"  # options: ground_truth, heuristic, None
        _C.ADEPT_JSON.ATTRIBUTES_KEYS = (
            "attributes",
            "pred_attr_43044",
        )
    elif _C.BASE_NAME == "ai2thor-intphys":
        _C.DATA_LOCATION = "/disk1/mcs-data"
        _C.SPLITS = (
            "_train",
            "_val",
        )
        _C.MIN_AREA = 50
    else:
        raise NotImplementedError

    # _C.DATA_MODE = "zip" if _C.DATA_LOCATION.endswith(".zip") else "folder"

    return _C
