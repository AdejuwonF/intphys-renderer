import pybullet as p
from .lite_step_objects import LiteObjectStepManager


# SIM_TIME_STEP = 1 / 30


def step(objects, sim_time_step, num_steps, forward=True):
    """Step a simulation"""
    om = LiteObjectStepManager(objects, forward=forward)
    p.setTimeStep(sim_time_step)
    for i in range(num_steps):
        p.stepSimulation()
    new_objects = []
    for object in om.object_ids:
        new_objects.append(om.get_object_motion(object))
    return new_objects


def reverse_step(config, sim_time_step, num_steps):
    """Reverse step a simulation"""
    return step(config, sim_time_step, num_steps ,False)
