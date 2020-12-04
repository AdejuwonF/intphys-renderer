import json
import math
import numpy as np
import os
import pathlib
from typing import Dict, List, Tuple

import mcs_implant.ai2thor_inventory

import logging as log

log.basicConfig(level=log.DEBUG)


# TODO List
# - Run on set of videos
# - Render test locations
# - Unit tests

class VariableStatistics:
    def __init__(self, variable_name: str, value: float):
        self.name = variable_name
        self.max = value
        self.min = value
        self.mean = value
        self.s_1 = 1
        self.s_x = value
        # If we should want to add SD, kurtosis and skewness in the future,
        # see https://www.johndcook.com/blog/standard_deviation/ and https://www.johndcook.com/blog/skewness_kurtosis/

    def store_next_value(self, value: float):
        if value > self.max:
            self.max = value
        elif value < self.min:
            self.min = value
        self.s_1 += 1
        self.s_x += value
        self.mean = self.s_x / self.s_1

    def __str__(self):
        abs_max = max(abs(self.max), abs(self.min))
        return f'Variable {self.name} has max {self.max}, min {self.min} ("absmax" {abs_max}) and mean {self.mean}.'


class ScenegraphConverter:

    def __init__(self):
        # Variable position_x has max 42.733736886158795, min -31.142052029371285 ("absmax" 42.733736886158795) and mean 0.6146091242254588.
        # Variable position_y has max 34.573365808674716, min -30.27921600709447 ("absmax" 34.573365808674716) and mean -0.38649033239594555.
        # Variable position_z has max 1.6069136673974862, min -1.2021972778021028 ("absmax" 1.6069136673974862) and mean 0.3851243910365225.
        self.CORA_DATASET_X_MAX = 50
        self.CORA_DATASET_Y_MAX = self.CORA_DATASET_X_MAX
        self.CORA_DATASET_Z_MAX = 2
        self.AI2THOR_X_MAX = 4.8
        self.AI2THOR_Y_MAX = 3.0
        self.AI2THOR_Z_MAX = 4.8
        self.CORA_TO_AI2THOR_COORD_FACTOR = 50.0 / 3.2

        # Variable scale_x has max 3.213827334794973, min -2.4043945556042057 ("absmax" 3.213827334794973) and mean 0.4209375122069409.
        # Variable scale_y has max 3.213827334794973, min -2.4043945556042057 ("absmax" 3.213827334794973) and mean 0.4209375122069409.
        # Variable scale_z has max 3.213827334794973, min -2.4043945556042057 ("absmax" 3.213827334794973) and mean 0.4209375122069409.
        # Variable radius has max 0.6938748732475681, min 0.3002468832444363 ("absmax" 0.6938748732475681) and mean 0.5109614726726474.
        # Variable length has max 1.6873130931093117, min 0.4567667419432115 ("absmax" 1.6873130931093117) and mean 1.1304760291224367.
        self.CORA_TO_AI2THOR_SCALE_FACTOR = 0.13

        # Variable pitch_radians has max 0.0, min 0.0 ("absmax" 0.0) and mean 0.0.
        # Variable yaw_radians has max 0.9861513120641386, min -2.29598167599285 ("absmax" 2.29598167599285) and mean -0.447220927945784.
        # Variable roll_radians has max 0.0, min 0.0 ("absmax" 0.0) and mean 0.0.
        # => Just converting radians to degrees should suffice

        self.variable_dictionary: Dict[VariableStatistics] = dict()

    def convert_coordinates_from_cora_to_ai2thor(self, x: float, y: float, z: float):
        # Note: Cora floor is x,y-plane, whereas AI2Thor floor is x,z-plane
        # x and z should be in [-4.8,4.8] for AI2Thor rather than [-10,10]; y in [0,3] rather than [0,2.5] or so
        # Can move anywhere on floor which is 100x100 centered at origin, so real range is [-50,50]
        # Option a:
        # position_x = x / self.CORA_TO_AI2THOR_COORD_FACTOR
        # position_y = z / self.CORA_TO_AI2THOR_COORD_FACTOR
        # position_z = y / self.CORA_TO_AI2THOR_COORD_FACTOR
        # Option b:
        position_x = x * self.AI2THOR_X_MAX / self.CORA_DATASET_X_MAX
        position_y = 0.0  # currently all objects on floor
        position_z = y * self.AI2THOR_Z_MAX / self.CORA_DATASET_Y_MAX
        return position_x, position_y, position_z

    @staticmethod
    def build_rotation_matrices(x: float, y: float, z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, math.cos(x), -math.sin(x)],
                        [0.0, math.sin(x), math.cos(x)]])
        r_y = np.array([[math.cos(y), 0.0, math.sin(y)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(y), 0.0, math.cos(y)]])
        r_z = np.array([[math.cos(z), -math.sin(z), 0.0],
                        [math.sin(z), math.cos(z), 0.0],
                        [0.0, 0.0, 1.0]])
        return r_x, r_y, r_z

    @staticmethod
    def convert_rotation_from_cora_to_ai2thor(yaw: float, pitch: float, roll: float):
        # Cora uses Euler angle order YPR in radians (just like math.cos/sin), AI2Thor uses RPY (zxy) in degrees
        # cmp. https://docs.unity3d.com/ScriptReference/Transform-eulerAngles.html
        # Yaw is around vertical axis; Pitch is around side-to-side axis; Roll is around front-to-back axis
        # => Multiply Cora angle matrices together, then extract AI2Thor angles following
        #    https://www.geometrictools.com/Documentation/EulerAngles.pdf

        # Cora is in x,y-plane, so pitch is around x (horizontal), roll around y (into scene), yaw around z (vert. axis)
        r_x, r_y, r_z = ScenegraphConverter.build_rotation_matrices(pitch, roll, yaw)
        # => r_x pitch, r_y roll, r_z yaw; YPR => first r_z, then r_x, then r_y (first transformation is right in mult.)
        r_cora_ypr = r_y @ r_x @ r_z  # Yaw first (vert.), then pitch (horizontal), then roll (around axis into scene)
        # print(f'Initial pitch {pitch * 180.0 / math.pi} roll {roll * 180.0 / math.pi} yaw {yaw * 180.0 / math.pi}')

        # AI2Thor MCS does RPY and since it is in x,z-plane => R~z, P~x and Y~y => Need R_y R_x R_z
        # cmp. section 2.3 in David Eberly, Geometric Tools (2014/Mar/10)
        r_12 = r_cora_ypr[1][2]
        if r_12 < 1:
            if r_12 > -1:
                theta_x = math.asin(- r_12)
                theta_y = math.atan2(r_cora_ypr[0][2], r_cora_ypr[2][2])
                theta_z = math.atan2(r_cora_ypr[1][0], r_cora_ypr[1][1])
            else:  # R-12 == -1
                theta_x = math.pi / 2.0
                theta_y = - math.atan2(- r_cora_ypr[0][1], r_cora_ypr[0][0])
                theta_z = 0.0
        else:  # r_12 == 1
            theta_x = - math.pi / 2.0
            theta_y = math.atan2(- r_cora_ypr[0][1], r_cora_ypr[0][0])
            theta_z = 0.0

        # But: Eberly assumes our z to be x, our x to be y and our y to be z => Need R_z R_y R_x (cmp. section 2.6)
        # r_20 = r_cora_ypr[2][0]
        # if r_20 < 1:
        #     if r_20 > -1:
        #         theta_y = math.asin(- r_20)
        #         theta_z = math.atan2(r_cora_ypr[1][0], r_cora_ypr[0][0])
        #         theta_x = math.atan2(r_cora_ypr[2][1], r_cora_ypr[2][2])
        #     else:  # R-12 == -1
        #         theta_y = math.pi / 2.0
        #         theta_z = - math.atan2(- r_cora_ypr[1][2], r_cora_ypr[1][1])
        #         theta_x = 0.0
        # else:  # r_12 == 1
        #     theta_y = - math.pi / 2.0
        #     theta_z = math.atan2(- r_cora_ypr[1][2], r_cora_ypr[1][1])
        #     theta_x = 0.0

        # Factorize as R_x, R_z, R_y (section 2.2)
        # r_01 = r_cora_ypr[0][1]
        # if r_01 < 1:
        #     if r_01 > -1:
        #         theta_z = math.asin(- r_01)
        #         theta_x = math.atan2(r_cora_ypr[2][1], r_cora_ypr[1][1])
        #         theta_y = math.atan2(r_cora_ypr[0][2], r_cora_ypr[0][0])
        #     else:  # R-12 == -1
        #         theta_z = math.pi / 2.0
        #         theta_x = - math.atan2(- r_cora_ypr[2][0], r_cora_ypr[2][2])
        #         theta_y = 0.0
        # else:  # r_12 == 1
        #     theta_z = - math.pi / 2.0
        #     theta_x = math.atan2(- r_cora_ypr[2][0], r_cora_ypr[2][2])
        #     theta_y = 0.0


        # TODO: This sequence is the only one which looks right - why do we have to flip like this?
        # Long shot: Maybe x and y are flipped from what I assumed and then y and z have to be flipped for AI2Thor?
        rotation_x = theta_y * 180.0 / math.pi
        rotation_y = theta_z * 180.0 / math.pi
        rotation_z = theta_x * 180.0 / math.pi
        # Note: This seems exactly identical to just converting pitch, roll and yaw to degrees, but is more principled
        #       (order has to be roll, yaw, pitch)
        # print(f'Resulting rotX {rotation_x} rotY {rotation_y} rotZ {rotation_z}')

        return rotation_x, rotation_y, rotation_z

    def detect_occluder(self, shape: Dict) -> bool:
        # Occluders should all have same shape
        # "shape": {
        #     "shape_type": "box",
        #     "shape_params": {
        #         "scale_x": 1.5,
        #         "scale_y": 0.1,
        #         "scale_z": 8.0
        #     }
        if 'shape' in shape:
            shape = shape.get('shape', {})
        shape_type = shape.get('shape_type', '')
        shape_params = shape.get('shape_params', {})
        scale_x = shape_params.get('scale_x', 0.0)
        scale_y = shape_params.get('scale_y', 0.0)
        scale_z = shape_params.get('scale_z', 0.0)
        return self.detect_occluder_from_params(shape_type, scale_x, scale_y, scale_z)

    @staticmethod
    def detect_occluder_from_params(shape_type: str, scale_x: float, scale_y: float, scale_z: float) -> bool:
        return shape_type == 'box' and scale_x == 1.5 and scale_y == 0.1 and scale_z == 8.0

    def convert_scale_from_cora_to_ai2thor(self, x: float, y: float, z: float):
        scale_x = max(0, x * self.CORA_TO_AI2THOR_SCALE_FACTOR)
        # Note: scale_z and scale_y are flipped due to x,y- vs x,z-plane
        scale_z = max(0, y * self.CORA_TO_AI2THOR_SCALE_FACTOR)
        scale_y = max(0, z * self.CORA_TO_AI2THOR_SCALE_FACTOR)
        return scale_x, scale_y, scale_z

    def convert_cylinder_scale_from_cora_to_ai2thor(self, radius: float, length: float):
        scale_x = max(0, radius * self.CORA_TO_AI2THOR_SCALE_FACTOR)
        scale_y = max(0, length * self.CORA_TO_AI2THOR_SCALE_FACTOR)
        scale_z = max(0, radius * self.CORA_TO_AI2THOR_SCALE_FACTOR)
        return scale_x, scale_y, scale_z

    def extract_data_from_cora_frame(self, cora_object: Dict):
        name = cora_object.get('name', '')
        shape = cora_object.get('shape', {})
        shape_type = shape.get('shape_type', None)
        shape_params = shape.get('shape_params', {})
        scale_x = shape_params.get('scale_x', None)
        scale_y = shape_params.get('scale_y', None)
        scale_z = shape_params.get('scale_z', None)
        radius = shape_params.get('radius', None)
        length = shape_params.get('length', None)
        pose6d = cora_object.get('pose6d', {})
        position_x = pose6d.get('x')
        position_y = pose6d.get('y')
        position_z = pose6d.get('z')
        pitch_radians = pose6d.get('pitch_radians')
        yaw_radians = pose6d.get('yaw_radians')
        roll_radians = pose6d.get('roll_radians')
        return name, shape_type, position_x, position_y, position_z, scale_x, scale_y, scale_z, radius, length, \
            pitch_radians, yaw_radians, roll_radians

    def track_variable(self, variable_name: str, variable_value) -> None:
        if variable_value is None or variable_name is None or variable_name == '':
            return
        if variable_name in self.variable_dictionary:
            self.variable_dictionary[variable_name].store_next_value(variable_value)
        else:
            self.variable_dictionary[variable_name] = VariableStatistics(variable_name, variable_value)

    def statistics_for_dataset(self, input_data_path: pathlib.Path, skip_floor: bool = True,
                               skip_occluders: bool = True, only_analyze_initial_frame: bool = False):
        """
        Computes statistics for variables in dataset including position, scale and rotation.

        Args:
            input_data_path: Path to dataset folder
            skip_floor: Whether to skip floor when computing statistics
            skip_occluders: Whether to skip occluders when computing statistics
            only_analyze_initial_frame: Whether to only analyze first frame in each video (True) or all (False)

        Returns:
            None

        Example:
            converter = ScenegraphConverter()
            converter.statistics_for_dataset(pathlib.Path('/home/fplk/data/subset_pybullet_input'),
                                             skip_floor=True, skip_occluders=True)
        """
        videos = sorted([x for x in input_data_path.iterdir() if x.is_dir()])
        log.debug(f'Videos: {videos}')
        self.variable_dictionary = dict()
        for video in videos:
            scenes = sorted(video.glob('**/*.json'))
            log.debug(f'Scenes for video {video}: {scenes}')

            for scene in scenes:
                log.debug(f'Parsing scene {scene}...')
                with scene.open(mode='r') as file:  # pathlib.Path(str(scene))
                    scenegraph = json.load(file)
                    for object_shapesworld in scenegraph.get('objects', []):
                        name, shape_type, position_x, position_y, position_z, scale_x, scale_y, scale_z, radius, \
                            length, pitch_radians, yaw_radians, roll_radians = \
                            self.extract_data_from_cora_frame(object_shapesworld)
                        if skip_floor and name == 'floor':
                            continue
                        if skip_occluders and self.detect_occluder_from_params(shape_type, scale_x, scale_y, scale_z):
                            continue
                        self.track_variable('position_x', position_x)
                        self.track_variable('position_y', position_y)
                        self.track_variable('position_z', position_z)
                        self.track_variable('scale_x', scale_x)
                        self.track_variable('scale_y', scale_y)
                        self.track_variable('scale_z', scale_z)
                        self.track_variable('radius', radius)
                        self.track_variable('length', length)
                        self.track_variable('pitch_radians', pitch_radians)
                        self.track_variable('yaw_radians', yaw_radians)
                        self.track_variable('roll_radians', roll_radians)
                if only_analyze_initial_frame:
                    log.debug('Skipping other frames in video, since user selected to only analyze first one.')
                    break
        # Print statistics for full dataset
        for variable_key in self.variable_dictionary.keys():
            print(self.variable_dictionary[variable_key])

    def convert_cora_dataset_folder_to_ai2thor(self, input_data_path: str, output_data_path: str):
        """
        Converts entire dataset folder with many videos to AI2Thor format

        Args:
            input_data_path: Path to dataset folder
            output_data_path: Path to output folder

        Returns:
            None
        """
        input_data_path = pathlib.Path(input_data_path)
        log.info(f'Converting dataset at {input_data_path} to AI2Thor format with destination {output_data_path}...')
        videos = sorted(set([video for video in os.listdir(input_data_path)]))
        output_data_path = pathlib.Path(output_data_path)
        for video in videos:
            log.debug(f'Converting video {video} to AI2Thor...')
            self.convert_cora_video_scenegraphs_to_ai2thor(input_data_path / video,
                                                           output_data_path / f'{video}.json')

    def convert_cora_video_scenegraphs_to_ai2thor(self, input_directory: pathlib.Path, target_file_name: pathlib.Path,
                                                  pretty: bool = False):
        """
        Converts individual video folder from Cora syntax to AI2Thor

        Args:
            input_directory: Folder with Cora files
            target_file_name: File to write AI2Thor scenegraph into (note: one AI2Thor scenegraph comprises many frames)
            pretty: Whether to indent resulting JSON files for human consumption

        Returns:
            None

        Example:
            converter = ScenegraphConverter()
            converter.convert_cora_dataset_folder_to_ai2thor('/home/fplk/data/subset_pybullet_input',
                                                             '/home/fplk/data/subset_pybullet_scenegraphs')
        """
        # scenes = sorted([x for x in input_directory.iterdir() if x.is_file() and str(x).endswith('.json')])
        scenes = sorted(list(input_directory.glob('**/*.json')))

        scenegraph_ai2thor = {
            'ceilingMaterial': 'AI2-THOR/Materials/Walls/DrywallOrange',
            'floorMaterial': 'AI2-THOR/Materials/Walls/RedDrywall',
            'wallMaterial': 'AI2-THOR/Materials/Walls/DrywallBeige'
        }
        objects_ai2thor = dict()

        for index, scene in enumerate(scenes):
            # print(f'Processing scene {index}')
            with scene.open(mode='r') as file:
                scenegraph = json.load(file)
                for object_shapesworld in scenegraph.get('objects', []):
                    name = object_shapesworld.get('name', '')
                    if name == 'floor' or name is None or name == '':
                        continue

                    # Extract all information
                    _, shape_type, position_x, position_y, position_z, scale_x, scale_y, scale_z, radius, \
                        length, pitch_radians, yaw_radians, roll_radians = \
                        self.extract_data_from_cora_frame(object_shapesworld)

                    if self.detect_occluder_from_params(shape_type, scale_x, scale_y, scale_z):
                        object_material = 'AI2-THOR/Materials/Plastics/OrangePlastic'
                        salient_materials = ['plastic']
                    else:
                        object_material = 'AI2-THOR/Materials/Wood/LightWoodCounters4'
                        salient_materials = ['wood']

                    x, y, z = self.convert_coordinates_from_cora_to_ai2thor(position_x, position_y, position_z)

                    if shape_type == 'box':
                        scale_x, scale_y, scale_z = self.convert_scale_from_cora_to_ai2thor(scale_x, scale_y, scale_z)
                    elif shape_type == 'cylinder':
                        scale_x, scale_y, scale_z = self.convert_cylinder_scale_from_cora_to_ai2thor(radius, length)
                    else:
                        log.warning(f'Unknown shape type {shape_type}.')

                    rotation_x, rotation_y, rotation_z = \
                        self.convert_rotation_from_cora_to_ai2thor(yaw_radians, pitch_radians, roll_radians)

                    # Create object if it does not exist
                    if name not in objects_ai2thor and index == 0:
                        objects_ai2thor[name] = {
                            'id': name,
                            'type': 'cube' if shape_type == 'box' else 'cylinder',
                            'mass': 1.54,  # Placeholder - feel free to change
                            'salientMaterials': salient_materials,
                            'materialFile': object_material,
                            'pickupable': True,
                            'kinematic': True,
                            # Placeholder - feel free to change
                            'shows': []
                        }

                    # Extend object
                    rotation_y_offset = -90.0  # NOTE: We manually rotate objects - not sure why this is necessary!!
                    if name in objects_ai2thor and index == 0:
                        # Extend list
                        objects_ai2thor[name]['shows'].append({
                            'rotation': {
                                'x': rotation_x,
                                'y': rotation_y + rotation_y_offset,  #
                                'z': rotation_z
                            },
                            'position': {
                                'x': x,
                                'y': y,
                                'z': z
                            },
                            'stepBegin': index,
                            'scale': {
                                'x': scale_x,
                                'y': scale_y,  # zy
                                'z': scale_z
                            }
                        })
                    elif name in objects_ai2thor:
                        # Extend list
                        objects_ai2thor[name]['shows'].append({
                            'rotation': {
                                'x': rotation_x,
                                'y': rotation_y + rotation_y_offset,  # zy
                                'z': rotation_z
                            },
                            'position': {
                                'x': x,
                                'y': y,
                                'z': z
                            },
                            'stepBegin': index
                        })

        # Cora PyBullet Camera is defined in:
        # https://github.com/probcomp/Cora/blob/9d7b16e14a06ae5d52f2ed4e9c7b8f48a59a0b41/ShapesWorld/src/constants.jl#L29-L44
        # CAMERA_TARGET_POSITION = [0.0, 0.0, 1.5] Distance from camera to this target point: 5.0
        # CAMERA_YAW = -45.0 CAMERA_PITCH = -15.0 CAMERA_ROLL = 0.0 CAMERA_FOV_DEGREES = 60.0

        # Option a) Just setting "z": -4 will provide good overview of scene
        # scenegraph_ai2thor['performerStart'] = {
        #     "position": {
        #         "z": -4
        #     }}

        # Option b) Setting x = -1, z = 1 and rotation(0, 135, 1) for closeup of scene in front of original occluders
        # scenegraph_ai2thor['performerStart'] = {
        #     "position": {
        #         "x": -1,
        #         "z": 1
        #     },
        #     'rotation': {
        #         'x': 0,
        #         'y': 135,
        #         'z': 1
        #     }
        # }

        scenegraph_ai2thor['performerStart'] = {
            "position": {
                "x": -1,
                "z": -1
            },
            'rotation': {
                'x': 0,
                'y': 45,
                'z': 0
            }
        }

        scenegraph_ai2thor['goal'] = {'action_list': (len(scenes) - 1) * [['Pass']]}

        # scenegraph_ai2thor['objects'] = objects_ai2thor
        objects_ai2thor_list = []
        for object_ai2thor_key, object_ai2thor_value in objects_ai2thor.items():
            objects_ai2thor_list.append(object_ai2thor_value)
        scenegraph_ai2thor['objects'] = objects_ai2thor_list

        # if not target_file_name.endswith('.json'):
        #     target_file_name += '.json'
        with target_file_name.open(mode='w') as target_file:
            if pretty:
                json.dump(scenegraph_ai2thor, target_file, indent=4)
            else:
                json.dump(scenegraph_ai2thor, target_file)


if __name__ == '__main__':
    converter = ScenegraphConverter()
    converter.convert_cora_dataset_folder_to_ai2thor('/home/fplk/data/my_archive2/ts_1582426874',
                                                     '/home/fplk/data/subset_pybullet_scenegraphs')

    # converter = ScenegraphConverter()
    # converter.statistics_for_dataset(pathlib.Path('/home/fplk/data/subset_pybullet_input'),
    #                                  skip_floor=True, skip_occluders=True)
