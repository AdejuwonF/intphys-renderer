
import os

import yaml
from detectron2.config import CfgNode

from configs.adept_config import get_adept_cfg
from configs.dataset_config import data_get_cfg
from configs.derender_config import get_derender_config
from configs.detection_config import get_detection_cfg
from configs.dynamics_config import get_dynamics_cfg

_MODULE_CONFIG_MAP ={"detection": get_detection_cfg,
                     "derender": get_derender_config,
                     "dynamics": get_dynamics_cfg,
                     "adept": get_adept_cfg}

def set_output_directories(training_cfg, distributed):
    dataset_cfg = training_cfg.DATA_CFG
    base_directory = "output/"
    dataset_directory = os.path.join(base_directory,dataset_cfg.BASE_NAME)
    if dataset_cfg.DEBUG:
        dataset_cfg.BASE_DIRECTORY = os.path.join(dataset_directory, ".data_tmp_debug")
    else:
        dataset_cfg.BASE_DIRECTORY = os.path.join(dataset_directory, ".data_tmp")
    # dataset_cfg.BASE_DIRECTORY = os.path.join(dataset_directory, ".data_tmp")
    # dataset_cfg.BASE_DIRECTORY = os.path.join(dataset_directory, ".data_tmp_debug")
    os.makedirs(dataset_cfg.BASE_DIRECTORY, exist_ok=True)
    if hasattr(training_cfg, "MODULE_CFG"):
        module_cfg = training_cfg.MODULE_CFG
        experiments_directory = os.path.join(dataset_directory,module_cfg.TYPE)
        os.makedirs(experiments_directory,exist_ok=True)
        already_run = [int(r[-5:]) for r in os.listdir(experiments_directory) if r.startswith("exp_")]
        exp_num = max(already_run)+1 if len(already_run) > 0 else 0
        exp_folder = "exp_" + str(exp_num).zfill(5)
        exp_folder = "distributed_exp" if distributed else exp_folder
        exp_folder = exp_folder if not module_cfg.DEBUG else "debug_exp"

        experiment_output = os.path.join(experiments_directory, exp_folder)
        if module_cfg.RESUME_DIR != '':
            experiment_output = module_cfg.RESUME_DIR
            assert 'last_checkpoint' in os.listdir(experiment_output), 'last checkpoint should be present to resume'

        os.makedirs(experiment_output, exist_ok=True)
        module_cfg.OUTPUT_DIR = experiment_output
        # module_cfg.LOG_FILE= os.path.join(experiment_output, "training.log")

def load_cfg_from_file(cfg_file):
    with open(cfg_file,"r") as f:
        cfg = yaml.load(f)
    cfg = CfgNode(cfg)
    return cfg

def get_cfg(args):
    cfg = CfgNode()
    if "module" in args:
        module_cfg = _MODULE_CONFIG_MAP[args.module](args.dataset)
        cfg.MODULE_CFG = module_cfg
    if "dataset" in args:
        cfg.DATA_CFG = data_get_cfg(args.dataset)
    return cfg
