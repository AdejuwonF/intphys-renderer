from detectron2.config import CfgNode


def get_adept_cfg(dataset_basename):
    _C = CfgNode()
    _C.TYPE = "adept"
    _C.DEBUG = False

    _C.VERSION = "paper-adept" #options: "paper-adept", "intphys-adept"

    _C.RESUME = False
    _C.RESUME_DIR = ""
    _C.SEED = -1
    _C.CUDNN_BENCHMARK = False
    _C.DEBUG_VIDEOS = []  #["output/intphys/.data_tmp/adept_jsons/pred_attr_82649/intphys_dev_O1/video_00054.json"]
    _C.ANALYZE_RESULTS_FOLDER =  "None"
        #"output/intphys/adept/bk_distributed_exp"
        #"/home/aldo/cora-derender/output/adept/adept/bk_distributed_exp"
        #"/all/home/aldo/cora-derenderer/output/adept/adept/distributed_exp/"
    #'/all/home/aldo/cora-derenderer/output/intphys/adept/distributed_exp_complete_bk' #"None" #"output/intphys/adept/exp_00025"

    # -----------------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------------
    _C.DATASETS = CfgNode()
    _C.DATASETS.BASE_NAME = dataset_basename

    if _C.DATASETS.BASE_NAME == "intphys":
        ####### INTPHYS ###########
        _C.ATTRIBUTES_KEYS = (
                             "pred_attr_82649",
                             # "pred_attr_03227",
                             # "pred_attr_06472",
                             # "pred_attr_12664",
                             # "pred_attr_22318",
                             # "pred_attr_41337",
                             )

        _C.DATASETS.TEST = (
                            "intphys_dev_O1",
                            "intphys_dev_O2",
                            "intphys_dev_O3",
                            # "intphys_dev-meta_O1",
                            # "intphys_dev-meta_O2",
                            # "intphys_dev-meta_O3"
                            )
    elif _C.DATASETS.BASE_NAME == "adept":
        _C.ATTRIBUTES_KEYS = (
                              # "attributes",
                              "pred_attr_43044",
                              # "pred_attr_00650",
                              # 'pred_attr_10580',
                              # "pred_attr_18377",
                              # "pred_attr_34216"
                              )
        _C.DATASETS.TEST = (
            # "adept_val",
            # "adept_train", #TODO get back to only test
            "adept_human_create",
            "adept_human_vanish",
            "adept_human_short-overturn",
            "adept_human_long-overturn",
            "adept_human_visible-discontinuous",
            "adept_human_invisible-discontinuous",
            "adept_human_delay",
            "adept_human_block",
        )



    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    _C.MODEL = CfgNode()
    _C.MODEL.META_ARCHITECTURE = "PARTICLE_FILTER"
    # Particles to be used in the particle filter
    _C.MODEL.N_PARTICLES = 128
    # Threshold for minimal area for an objects to be considered visible
    _C.MODEL.AREA_THRESHOLD = 200.


    # -----------------------------------------------------------------------------
    # Dynamics Model
    # -----------------------------------------------------------------------------
    _C.MODEL.STEP = CfgNode()
    _C.MODEL.STEP.PERTURBATION = CfgNode()
    # Whether to perturb the objects
    _C.MODEL.STEP.PERTURBATION.TO_PERTURB = True
    # Sigma in the velocity term
    _C.MODEL.STEP.PERTURBATION.VELOCITY_SIGMA = [.01, .06]
    _C.MODEL.STEP.PERTURBATION.SCALE_SIGMA = .0005
    # Sigma in the location term
    _C.MODEL.STEP.PERTURBATION.LOCATION_SIGMA = [.01, .06]
    # Sigma in the velocity term, multiplicative
    _C.MODEL.STEP.PERTURBATION.VELOCITY_LAMBDA = [.01, .06]

    # -----------------------------------------------------------------------------
    # Magic in the dynamics model
    # -----------------------------------------------------------------------------
    _C.MODEL.STEP.MAGIC = CfgNode()
    # Whether to use magic
    _C.MODEL.STEP.MAGIC.USE_MAGIC = True
    # The probability to disappear
    _C.MODEL.STEP.MAGIC.DISAPPEAR_PROBABILITY = .02
    # The penalty for magically disappearing
    _C.MODEL.STEP.MAGIC.DISAPPEAR_PENALTY = 10.
    # The probability for magically stopping
    _C.MODEL.STEP.MAGIC.STOP_PROBABILITY = .02
    # The penalty for magically stopping
    _C.MODEL.STEP.MAGIC.STOP_PENALTY = 1.
    # The probability for magically accelerating
    _C.MODEL.STEP.MAGIC.ACCELERATE_PROBABILITY = .04
    # The penalty for magically accelerating
    _C.MODEL.STEP.MAGIC.ACCELERATE_PENALTY = 1.
    # The magnitude for magically accelerating
    _C.MODEL.STEP.MAGIC.ACCELERATE_LAMBDA = 1.5

    # -----------------------------------------------------------------------------
    # Particle filter
    # -----------------------------------------------------------------------------
    # The period for particle filter to resample
    _C.MODEL.RESAMPLE = CfgNode()
    # Resample every period
    _C.MODEL.RESAMPLE.PERIOD = 1
    # Scaling on nll
    _C.MODEL.RESAMPLE.FACTOR = 1.


    # -----------------------------------------------------------------------------
    # Mass sampler
    # -----------------------------------------------------------------------------
    _C.MODEL.MASS = CfgNode()
    # Whether to sample mass
    _C.MODEL.MASS.TO_SAMPLE_MASS = False
    # The log mean of mass
    _C.MODEL.MASS.LOG_MASS_MU = 0
    # The log stdev of mass
    _C.MODEL.MASS.LOG_MASS_SIGMA = 1

    # -----------------------------------------------------------------------------
    # Observation Model
    # -----------------------------------------------------------------------------
    _C.MODEL.UPDATING = CfgNode()
    _C.MODEL.UPDATING.MATCHED = CfgNode()
    # Loss for matched object updating
    _C.MODEL.UPDATING.MATCHED.LOSS = "Smoothed_L_Half"
    # Sigma in the location term
    _C.MODEL.UPDATING.MATCHED.LOCATION_SIGMA = .2
    # Sigma in the velocity term
    _C.MODEL.UPDATING.MATCHED.VELOCITY_SIGMA = .2
    _C.MODEL.UPDATING.MATCHED.SCALE_SIGMA = .05

    _C.MODEL.UPDATING.UNMATCHED_BELIEF = CfgNode()
    # Base Penalty coefficient for unseen object
    _C.MODEL.UPDATING.UNMATCHED_BELIEF.BASE_PENALTY = 1.
    # Penalty coefficient for unseen object w.r.t. mask area shown
    _C.MODEL.UPDATING.UNMATCHED_BELIEF.MASK_PENALTY = .0001

    _C.MODEL.UPDATING.UNMATCHED_OBSERVATION = CfgNode()
    # Penalty for object appearing
    _C.MODEL.UPDATING.UNMATCHED_OBSERVATION.PENALTY = .02
    _C.MODEL.UPDATING.UNMATCHED_OBSERVATION.MAX_PENALTY = 12.

    _C.MODEL.MATCHER = CfgNode()
    # PENALTY FOR MISMATCHED OBJECT TYPES, ONLY BETWEEN OCCLUDER AND OTHER
    _C.MODEL.MATCHER.TYPE_PENALTY = 10.
    # PENALTY FOR MISMATCHED OBJECT COLOR
    _C.MODEL.MATCHER.COLOR_PENALTY = 12.
    # PENALTY FOR MISMATCHED OBJECT WHEN THEY ARE AFAR
    _C.MODEL.MATCHER.DISTANCE_PENALTY = 14. if _C.VERSION=="intphys-adept" else 20.
    # THE THRESHOLD FOR OBJECT BEING AFAR
    _C.MODEL.MATCHER.DISTANCE_THRESHOLD = 2.
    # THE BASE PENALTY BETWEEN PLACEHOLDER AND OBJECTS
    _C.MODEL.MATCHER.BASE_PENALTY = 8.
    # when more than 5 objects creating more objects should not happen
    _C.MODEL.MATCHER.BASE_PENALTY_HIGH = 16

    return _C
    # -----------------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------------
    # _C.CASE_NAMES = []
    # _C.USE_GT_OBSERVATION = False
    # _C.ANNOTATION_FOLDER = ""
    # _C.OBSERVATION_FOLDER = ""
    # _C.OUTPUT_FOLDER = ""
    # _C.LOG_PREFIX = ""
    # _C.PLOT_SUMMARY = True
    # _C.ID = 0
