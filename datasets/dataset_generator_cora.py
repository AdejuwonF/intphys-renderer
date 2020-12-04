from structure.pipeline_configuration import Backend

import subprocess

import logging as log


class DatasetGenerator:
    def __init__(self, cora_dir: str):
        self.cora_dir = cora_dir

    def generate_videos(self, backend, output_dir: str, number_of_videos: int, frames_per_video: int, create_png: bool):
        """
        Generates video dataset
        Args:
            backend: Which backend to choose (Cora ShapesWorld or AI2Thor MCS), might support TDW in future
            output_dir: Path to output directory
            number_of_videos: Number of videos to generate
            frames_per_video: Frames in each video
            create_png: Whether to generate rendering as well (True) or only segmentation and depth map (False)

        Returns: None

        """
        if backend == Backend.CORA:
            log.info(f'Rendering {number_of_videos} videos via Cora ShapesWorld...')
            render_command_cora = f'{self.cora_dir}ShapesWorld/shapesworld_julia ' \
                                  f'{self.cora_dir}synthetic_videos/generate_scene.jl --num_videos={number_of_videos} '\
                                  f'--output_dir={output_dir} --frames_per_video={frames_per_video}'
            if create_png:
                render_command_cora = render_command_cora + ' --create_png'
            log.debug(f'Executing Cora ShapesWorld render command:\n{render_command_cora}')
            render_command = render_command_cora.split(' ')
            process = subprocess.Popen(render_command, shell=False)
            stdout, stderr = process.communicate()
            log.debug(f'stdout: {stdout}\nstderr: {stderr}')

