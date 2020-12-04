import codecs
import json
import pathlib

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, List


class Agent(ABC):
    @abstractmethod
    def select_action(self, scene_step_output) -> Tuple[Union[str, None], Dict]:
        """
        Selects action based on step output from scene.

        Args:
            scene_step_output: Step output from scene

        Returns:
            Tuple of action chosen and additional parameters.
            If the action is `None`, that means the agent has chosen to end the scene.
        """
        return 'Pass', {}

    # Could later contain geometry computations methods like these:
    # def move_to_object(self):
    #     my_position = np.array([0.0, 0.512499593, 0.0])
    #     my_view_direction = np.array([0.0, 0.0, 1.0])
    #     object_position = np.array([-2.5, 0, 3.5])
    #     object_direction = np.array([-0.5779197, -0.106701627, 0.8090881])
    #     object_distance = 8.602308
    #     cosine = np.dot(my_view_direction, object_direction) / (np.linalg.norm(my_view_direction) * np.linalg.norm(object_direction))
    #     print(f'RotateLook, rotation={-math.degrees(math.acos(cosine))}')
    #     rotation, horizon = math.degrees(object_direction[0]), math.degrees(object_direction[1])
    #     print(f'RotateLook, rotation={rotation}, horizon={horizon}')
    #
    #     direction_guess = object_position - my_position
    #     direction_guess_normalized = direction_guess / np.linalg.norm(direction_guess)
    #     print(f'Direction guess: {direction_guess_normalized}, real direction: {object_direction}')
    #
    # def compute_vector(self, my_position, my_view, object_position):
    #     difference = object_position - my_position
    #     cosine = np.dot(my_view, object_position) / (np.linalg.norm(my_view) * np.linalg.norm(object_position))
    #     angle = math.acos(cosine)
    #     print(f'Difference: {difference}, cosine: {cosine}, angle: {angle}')


class AgentLimitedSteps(Agent):
    """
    Agent that takes an action for a limited number of steps after which it returns None.
    Currently only used to pass time, but can be used to run for specific number of frames.
    """
    def __init__(self, max_steps: int = 100) -> None:
        """
        Initialize agent
        Args:
            max_steps: Number of steps to take before returning None as action
        """
        self.count = 0
        self.max_steps = max_steps

    def reset(self) -> None:
        """
        Resets step counter. Useful to do in between scenes.
        Returns:
            None
        """
        self.count = 0

    def select_action(self, scene_step_output) -> Tuple[Union[str, None], Dict]:
        """
        Selects action based on step output from scene. Currently only returns "Pass" until max number of steps is
        reached and then returns None.

        Args:
            scene_step_output: Step output from scene

        Returns:
            Tuple of action chosen and additional parameters (currently empty dictionary)
        """
        if self.count < self.max_steps:
            self.count += 1
            return 'Pass', {}
        else:
            return None, {}


class AgentPredeterminedPlan(Agent):
    """
    Scripted agent that follows a predefined path to a goal. Intended to execute goal actions from MCS IntPhys scenes.
    """
    def __init__(self, path_to_scene_json_with_goal_path):
        """
        Initializes goal path agent.

        Args:
            path_to_scene_json_with_goal_path: Path to JSON scene file that contains actions to execute
        """
        self.count = 0
        self.action_list = self.read_action_list_from_mcs_scene(path_to_scene_json_with_goal_path)

    def reset(self) -> None:
        """
        Resets step counter (i.e. restarts the predetermined plan). Useful to do in between scenes.
        Returns:
            None
        """
        self.count = 0

    def select_action(self, scene_step_output) -> Tuple[Union[str, None], Dict]:
        """
        Selects next action. Returns action from its list until it reaches end.

        Args:
            scene_step_output: Step output from scene

        Returns:
            Selected action and additional parameters (currently empty dict)
        """
        del scene_step_output
        if self.count < len(self.action_list):
            selected_action = self.action_list[self.count]
            self.count += 1
            return selected_action, {}
        else:
            return None, {}

    @staticmethod
    def read_action_list_from_mcs_scene(p: pathlib.Path) -> List[str]:
        """
        Reads action list from an MCS scene JSON file.

        Args:
            p: Path to MCS JSON file to extract actions from

        Returns:
            List of actions
        """
        with open(str(p), 'rb') as file:
            file_content = codecs.decode(file.read(), 'utf-8-sig')
            scene_spec = json.loads(file_content)
            actions = scene_spec.get('goal', {}).get('action_list', [])
            action_list = [action[0] for action in actions if len(action) > 0]
            return action_list
