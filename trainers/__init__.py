from .trainable_detector import CustomDetectTrainer
from .trainable_derender import Trainable_Derender
from .trainable_dynamics import TrainableDynamics

_TRAINERS_MAP = {"detector": CustomDetectTrainer,
                 "derender": Trainable_Derender,
                 "dynamics": TrainableDynamics}

def build_trainer(cfg):
    return _TRAINERS_MAP[cfg.MODULE_CFG.TYPE](cfg.MODULE_CFG)