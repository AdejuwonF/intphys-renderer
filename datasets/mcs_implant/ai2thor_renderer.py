import argparse

import json
import pathlib
from typing import List, Dict

from machine_common_sense import MCS, MCS_Step_Output, MCS_Object

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import logging as log

from ai2thor_agent import AgentLimitedSteps, Agent, AgentPredeterminedPlan
from ai2thor_scenegraph_generator import ScenegraphGenerator


class AI2ThorRenderer:
    """
    Renders scenegraphs in AI2Thor MCS

    Example Usage:
    1 - Render example scenes with agents)
    renderer = AI2ThorRenderer()
    renderer.render_example_scenes()

    2 - Render minimal scenegraphs)
    2.1 First run scenegraph generator via
    scenegraph_generator = ScenegraphGenerator()
    scenegraph_generator.generate_scenegraphs_in_bulk(10, 'generated_scenegraphs_minimal', True)
    2.2 Then run renderer via
    renderer = AI2ThorRenderer()
    renderer.render_first_frame_in_bulk_for_directory('generated_scenegraphs_minimal/')

    3 - Resume scene generation
    Generate scenes from start index 1234 until 99999 (both inclusive) into a target directory
    To find resume index in an existing directory: "ls -1 | wc -l", add 1
    renderer.create_first_frame_until_n('<some_path>/generated_scenegraphs_renderings/', 99999, 1234)
    """
    def __init__(self, unity_app_file_path: str = '~/mitibm/AI2Thor_MCS/MCS-AI2-THOR-Unity-App-v0.0.3.x86_64'):
        """
        Initialize renderer

        Args:
            unity_app_file_path: Path to Unity app provided by AI2Thor team.
        """
        log.debug(f'AI2Thor Renderer initialized with working directory {os.getcwd()}')
        self.unity_app_file_path = os.path.expanduser(unity_app_file_path)
        self.controller = MCS.create_controller(self.unity_app_file_path)

    def render_via_agent(self, scenes: List[str], output_dir: str, goal_mode: bool) -> None:
        """
        First major type of rendering: Render by instantiating an agent. This method iterates over the provided scenes
        and calls into render_scene() to do the actual rendering based on the agent actions.

        Args:
            scenes: List of scene paths to load (can be absolute or relative)
            output_dir: Directory to render into (with result described in
                        :func:`~mcs_implant.AI2ThorRenderer.render_scene`)
            goal_mode: Whether to instantiate a goal path agent (otherwise, a limited steps agent is used)

        Returns:
            None
        """

        if not goal_mode:
            agent: Agent = AgentLimitedSteps(1)

        for index_video in range(0, len(scenes)):
            print(f'Rendering video {index_video + 1}/{len(scenes)}...')

            input_path = scenes[index_video]  # pathlib.Path(scenes_path) /

            # Option a: Label output videos consecutively irregardless of original indices
            # output_path = pathlib.Path(output_dir) / f'video_{str(index_video + 1).zfill(5)}'  # Cora indexes from 1

            # Option b: Label output videos based on input video index
            match = re.findall(r'video_\d+', str(input_path))
            if len(match) != 1:
                log.warning(f'Unexpected number of video_<index> matches in input path. Skipping video {input_path}')
                continue
            output_path = pathlib.Path(output_dir) / match[0]

            # Create output folder if it should not exist
            output_path.mkdir(parents=True, exist_ok=True)

            # Important: Reset agent for new epoch - otherwise it will return with last response (e.g. None to stop)
            if goal_mode:
                agent = AgentPredeterminedPlan(input_path)
            else:
                agent.reset()

            self.render_scene(agent, input_path, output_path)

    def save_mcs_output_to_files(self, output: MCS_Step_Output, output_path: pathlib.PosixPath, frame_count: int,
                                 create_png: bool = False, debug_render: bool = False):
        file_name = str(frame_count).zfill(5)
        rgb_image = np.asarray((output.image_list[0]))
        segmentation_map = np.asarray((output.object_mask_list[0]))
        depth_map = np.asarray((output.depth_mask_list[0]))

        if debug_render:
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(rgb_image)
            axarr[1].imshow(segmentation_map)
            axarr[2].imshow(depth_map)
            plt.show()

        # If skip_if_invisible_objects, check whether number of objects according to scenegraph equals visible ones
        if create_png:
            matplotlib.image.imsave(output_path / f'{file_name}.png', rgb_image)
        np.save(output_path / f'{file_name}.seg.npy', segmentation_map)
        np.save(output_path / f'{file_name}.depth.npy', depth_map)

    def update_scenegraph_with_segmentation_to_object_associations(self, scenegraph: Dict,
                                                                   object_list: List[MCS_Object], frame_count: int):
        frame_count = frame_count - 1  # Video names are indexed from 1, but internal arrays from 0 => subtract 1 here
        # For each object in list retrieve entry in scenegraph and add color to its shows entry for current frame_count
        for mcs_object in object_list:
            identifier = mcs_object.uuid
            color = mcs_object.color
            if identifier is None or color is None:
                continue  # Insufficient information for update => skip
            # Find object in scenegraph and go to frame_count in shows field
            # target_object = next((item for item in scenegraph.get('objects', [])
            #                 if item.get('id', None) == id), None)
            object_index = next(i for i, val in enumerate(scenegraph.get('objects', []))
                                if val.get('id', None) == identifier)
            # NOTE: Assumption is that our frames are ordered in shows array!! If they are not, we need to perform
            #       a search on field stepBegin like we did for uuid
            # try:
            # print(f'Updating scenegraph for object index {object_index} and frame {frame_count}...')
            scenegraph['objects'][object_index]['shows'][frame_count]['color'] = color
            # except IndexError:
            #     return
        # We could return scenegraph (return scenegraph), but dicts are mutable, so they are modified in-place in Python

    def render_scene(self, agent, input_path: str, output_path: str, create_png: bool = True,
                     debug_render: bool = False) -> None:
        """
        Places the given agent (`agent`) in the given scene (described by `input_path`), and renders in `output_path`
        the scene state at each time step from time zero until the scene ends.  (The scene ends when the agent chooses
        the action `None`.)

            Creates video_<index> folders with:
                * Rendering <frame_index>.png if create_png is set (default)
                * Depth Map <frame_index>.depth.npy
                * Segmentation Map <frame_index>.seg.npy
                * Scenegraph <frame_index>.json

        Args:
            agent: Agent to decide on actions
            input_path: Path to scenegraph JSON that describes scene setup
            output_path: Path to render into
            create_png: Whether to create rendered image (otherwise, only depth and segmentation map will be produced)
            debug_render: Whether to show side-by-side rendering for developer

        Returns:
            None
        """
        print(f'Rendering {input_path}...')
        frame_count = 1
        output_path_posix = pathlib.Path(output_path)  # TODO: Refactor so this function takes pathlib.Path
        # config_data, status = MCS.load_config_json_file(scenes[index_video])
        config_data, status = MCS.load_config_json_file(input_path)
        log.debug(f'Status: {status}')
        # controller = MCS.create_controller(self.unity_app_file_path)
        output = self.controller.start_scene(config_data)

        # Save output to files
        self.save_mcs_output_to_files(output, output_path_posix, frame_count, create_png)
        # Update scenegraph
        self.update_scenegraph_with_segmentation_to_object_associations(config_data, output.object_list, frame_count)

        # Use ML to select next action based on scene output (goal, actions, images, metadata, ...) from previous action
        action, params = agent.select_action(output)

        # Continue to select actions until frame number is reached
        while action is not None:
            frame_count += 1
            # file_name = str(frame_count).zfill(5)

            output = self.controller.step(action)
            if output is None:
                break

            # Save output to files
            self.save_mcs_output_to_files(output, output_path_posix, frame_count, create_png)
            # Update scenegraph
            self.update_scenegraph_with_segmentation_to_object_associations(config_data, output.object_list,
                                                                            frame_count)

            action, params = agent.select_action(output)

            # rgb_image = np.asarray((output.image_list[0]))
            # segmentation_map = np.asarray((output.object_mask_list[0]))
            # depth_map = np.asarray((output.depth_mask_list[0]))
            #
            # # if debug_render:
            # #     f, axarr = plt.subplots(1, 3)
            # #     axarr[0].imshow(rgb_image)
            # #     axarr[1].imshow(segmentation_map)
            # #     axarr[2].imshow(depth_map)
            # #     plt.show()
            #
            # # If skip_if_invisible_objects, check whether number of objects according to scenegraph equals visible ones
            # if create_png:
            #     matplotlib.image.imsave(f'{output_path}/{file_name}.png', rgb_image)
            # np.save(f'{output_path}/{file_name}.seg.npy', segmentation_map)
            # np.save(f'{output_path}/{file_name}.depth.npy', depth_map)

        self.controller.end_scene('classification', 1.0)

        # TODO: Write out modified scenegraph
        with (output_path_posix / 'scenegraph.json').open('w') as file:
            json.dump(config_data, file, indent=4)



        self.controller.end_scene('classification', 'confidence')

    # def associate_objects_with_segmentation(self, object_list):
    #     for scene_object in object_list:
    #         name = scene_object.uuid
    #         color = np.array([scene_object.color.get('r', 0), scene_object.color.get('g', 0),
    #                           scene_object.color.get('b', 0)])
    #         # if debug_render:
    #         #     segmentation_for_object = np.where((segmentation_map == color), segmentation_map, 0)
    #         #     f, axarr = plt.subplots(1, 2)
    #         #     axarr[0].imshow(segmentation_for_object)
    #         #     axarr[1].imshow(segmentation_map)
    #         #     plt.show()

    def render_example_scenes(self) -> None:
        """
        Method to render some predefined example scenes. Shows both IntPhys and static scene rendering.
        Returns:
            None
        """
        scenes_goal_oriented = [
            '../python_api/scenes/intphys_energy_conservation_implausible_sample_1.json',
            '../python_api/scenes/intphys_energy_conservation_plausible_sample_1.json',
            '../python_api/scenes/intphys_gravity_implausible_sample_1.json',
            '../python_api/scenes/intphys_gravity_plausible_sample_1.json',
            '../python_api/scenes/intphys_object_permanence_implausible_sample_1.json',
            '../python_api/scenes/intphys_object_permanence_plausible_sample_1.json',
            '../python_api/scenes/intphys_shape_constancy_implausible_sample_1.json',
            '../python_api/scenes/intphys_shape_constancy_plausible_sample_1.json'
        ]
        scenes_sandbox = [
            '../python_api/scenes/playroom.json',
            './test.json',
        ]
        # self.render_via_agent(scenes_goal_oriented, 'test_output_ai2thor_goal_oriented', True)
        self.render_via_agent(scenes_sandbox, 'test_output_main_ai2thor', False)

    # def render_first_frame_in_bulk_for_directory(self, input_directory: str, output_index: int = 1,
    #                                              output_dir: str = 'generated_scenegraphs_renderings/',
    #                                              skip_if_invisible_objects: bool = True):
    #     """
    #     Iterates over scene files in input_directory and renders them to output directory via
    #     render_first_frame_in_bulk().
    #
    #     Args:
    #         input_directory: Directory with scenegraphs, e.g. those generated by Scenegraph Generator
    #         output_index: Index to resume from
    #         output_dir: Directory to render into
    #         skip_if_invisible_objects: Whether to skip a scene if it contains objects that are invisible in the image
    #     Returns:
    #         None
    #     """
    #     if not input_directory.endswith('/'):
    #         input_directory += '/'
    #     scenes = sorted(set([f'{input_directory}{scenegraph}' for scenegraph in os.listdir(input_directory) if scenegraph.endswith('.json')]))
    #     print(scenes)
    #     self.render_first_frame_in_bulk(scenes, output_dir, skip_if_invisible_objects, output_index)
    #
    # def render_first_frame_in_bulk(self, scenes: List[str], output_dir: str,
    #                                skip_if_invisible_objects: bool = False, output_index: int = 1) -> None:
    #     """
    #     Renders first frame of scenes in list.
    #
    #     Args:
    #         scenes: List of paths to scenegraphs to render
    #         output_dir: Directory to render into
    #         skip_if_invisible_objects: Whether to skip a scene if it contains objects not visible in the image
    #         output_index: Index to resume from
    #
    #     Returns:
    #         None
    #     """
    #
    #     # Index for output folder (video_<output_index>), only incremented if scene is picked
    #     #                   [Scene is not picked if we skip due to invisible objects.]
    #
    #     for scene_index in range(0, len(scenes)):
    #         print(f'Processing scene {scene_index + 1}/{len(scenes)}...')
    #
    #         input_path = scenes[scene_index]  # pathlib.Path(scenes_path) /
    #         output_path = pathlib.Path(output_dir) / f'video_{str(output_index).zfill(5)}'
    #
    #         success = self.render_first_frame_from_file(input_path, output_path, skip_if_invisible_objects)
    #         if success:
    #             output_index += 1
    #
    # def render_first_frame_from_file(self, input_path, output_path, skip_if_invisible_objects: bool = False,
    #                                  create_png: bool = True, debug_render: bool = False) -> bool:
    #     """
    #     Renders the first frame of a scene that is loaded from a file
    #
    #     Example:
    #         renderer.render_first_frame_from_file('<path>/video_00001/00001.json', 'rendered_frames/')
    #
    #     Args:
    #         input_path: Path to scenegraph of scene to render
    #         output_path: Path to directory to render into
    #         skip_if_invisible_objects: Whether to skip a scene if it contains objects not visible in the image
    #         create_png: Whether to create rendered image (otherwise, only depth and segmentation map will be produced)
    #         debug_render: Whether to render images side-by-side for developer
    #
    #     Returns:
    #         True if scene was rendered, False if it was rejected (due to invisible objects)
    #     """
    #     config_data, status = MCS.load_config_json_file(input_path)
    #     log.debug(f'Status: {status}')
    #     return self.render_first_frame_from_dict(output_path, config_data, skip_if_invisible_objects, create_png,
    #                                              debug_render)
    #
    # def render_first_frame_from_dict(self, output_path, config_data, skip_if_invisible_objects: bool = False,
    #                                  create_png: bool = True, debug_render: bool = False) -> bool:
    #     """
    #     Renders the first frame of a scene that is loaded from a file.
    #     Creates video_<index> folders with:
    #         * Rendering <frame_index>.png if create_png is set (default)
    #         * Depth Map <frame_index>.depth.npy
    #         * Segmentation Map <frame_index>.seg.npy
    #         * Scenegraph <frame_index>.json
    #
    #     Args:
    #         input_path: Path to scenegraph of scene to render
    #         config_data: Dictionary with scenegraph (in MCS JSON format)
    #         skip_if_invisible_objects: Whether to skip a scene if it contains objects not visible in the image
    #         create_png: Whether to create rendered image (otherwise, only depth and segmentation map will be produced)
    #         debug_render: Whether to render images side-by-side for developer
    #
    #     Returns:
    #         True if scene was rendered, False if it was rejected (due to invisible objects)
    #     """
    #     output = self.controller.start_scene(config_data)
    #
    #     frame_count = 1
    #     file_name = str(frame_count).zfill(5)
    #
    #     rgb_image = np.asarray((output.image_list[0]))
    #     segmentation_map = np.asarray((output.object_mask_list[0]))
    #     depth_map = np.asarray((output.depth_mask_list[0]))
    #
    #     if debug_render:
    #         f, axarr = plt.subplots(1, 3)
    #         axarr[0].imshow(rgb_image)
    #         axarr[1].imshow(segmentation_map)
    #         axarr[2].imshow(depth_map)
    #         plt.show()
    #
    #     def has_occluded_objects():
    #         return len(config_data.get('objects', [])) != (np.unique(segmentation_map.reshape(-1,3), axis=0).shape[0]-2)
    #
    #     # If skip_if_invisible_objects, check whether number of objects according to scenegraph equals visible ones
    #     success = False  # Default value: Unsuccessful (is set if next line triggers)
    #     if skip_if_invisible_objects and has_occluded_objects():
    #         pass
    #     else:
    #         output_path.mkdir(parents=True, exist_ok=True)
    #         if create_png:
    #             matplotlib.image.imsave(f'{output_path}/{file_name}.png', rgb_image)
    #         np.save(f'{output_path}/{file_name}.seg.npy', segmentation_map)
    #         np.save(f'{output_path}/{file_name}.depth.npy', depth_map)
    #         with open(f'{output_path}/{file_name}.json', 'w') as file:  # _scenegraph_gold not appended for consistency
    #             json.dump(config_data, file, indent=4)
    #         success = True
    #
    #     self.controller.end_scene('classification', 'confidence')
    #     return success
    #
    # def create_first_frame_until_n(self, output_dir: str, n: int, output_index: int = 1, minimal: bool = True,
    #                                skip_if_invisible_objects: bool = True) -> None:
    #     """
    #     Generates scenes and renders their first frame until a maximum number is reached.
    #     Args:
    #         output_dir: Directory to render into
    #         n: Maximum number of scenes
    #         output_index: Index to resume from if set
    #         minimal: Whether to generate simplified scenegraphs (limited shapes, biased to place objects within view)
    #         skip_if_invisible_objects: Whether to skip a scene if it contains objects not visible in the image
    #
    #     Returns:
    #         None
    #     """
    #     scenegraph_generator = ScenegraphGenerator()
    #
    #     while output_index <= n:
    #         scenegraph = scenegraph_generator.generate_scenegraph_dict(minimal)
    #         output_path = pathlib.Path(output_dir) / f'video_{str(output_index).zfill(5)}'
    #         success = self.render_first_frame_from_dict(output_path, scenegraph, skip_if_invisible_objects)
    #         if success:
    #             output_index += 1


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_folder", help="Folder with scenegraphs to render")
    # parser.add_argument("output_folder", help="Output folder to render into")
    # args = parser.parse_args()
    # print(f'Rendering scenegraphs from {args.input_folder} into {args.output_folder}')
    # renderer = AI2ThorRenderer()
    # renderer.render_first_frame_in_bulk_for_directory(args.input_folder,
    #                                                   output_index=1,
    #                                                   output_dir=pathlib.Path(args.output_folder),
    #                                                   skip_if_invisible_objects=False)

    renderer = AI2ThorRenderer()
    input_data_path = pathlib.Path('/home/fplk/data/subset_pybullet_scenegraphs')
    scenes = sorted(input_data_path.glob('**/*.json'))
    renderer.render_via_agent(scenes, '/home/fplk/data/subset_rendered_in_ai2thor_mcs', True)

    # renderer.render_example_scenes()
