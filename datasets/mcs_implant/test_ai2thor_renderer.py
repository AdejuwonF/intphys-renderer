import json
import os
import pathlib
import shutil
from pathlib import Path

from mcs_implant.ai2thor_renderer import AI2ThorRenderer


# Note: This unit test will only work if mcs_implant directory is copied into MCS project main folder, since it relies
#       on rendering components
# def test_rendering():
#     renderer = AI2ThorRenderer()
#     renderer.create_first_frame_until_n('./test_temp_renderings_deleteme/', 1, 1)
#     path = Path('./test_temp_renderings_deleteme/')
#     videos = sorted(set([video for video in os.listdir(path)]))
#     assert 'video_00001' in videos, 'videos_00001 not found in generated folders for test rendering.'
#     frames = sorted(set([video for video in os.listdir(path / 'video_00001')]))
#     assert '00001.depth.npy' in frames, 'Depth map not found in generated rendering files.'
#     assert '00001.seg.npy' in frames, 'Segmentation map not found in generated rendering files.'
#     assert '00001.png' in frames, 'Image not found in generated rendering files.'
#     assert '00001.json' in frames, 'Scenegraph JSON not found in generated rendering files.'
#     try:
#         shutil.rmtree(path)
#     except OSError as e:
#         print(f'Error when trying to remove temporary data from test_rendering(): {e.filename} - {e.strerror}.')

def test_update_scenegraph_with_segmentation_to_object_associations():
    renderer = AI2ThorRenderer()
    ai2thor_mcs_scenegraph_path = pathlib.Path('./test_data') / 'ai2thor_scenegraph_multiframe.json'
    with ai2thor_mcs_scenegraph_path.open('r') as file:
        ai2thor_mcs_scenegraph = json.load(file)
    object_list = [{'uuid': 'block1', 'color': {'r': 157, 'g': 208, 'b': 133}},
                   {'uuid': 'block2', 'color': {'r': 224, 'g': 216, 'b': 157}},
                   {'uuid': 'block4', 'color': {'r': 51, 'g': 222, 'b': 118}},
                   {'uuid': 'block5', 'color': {'r': 178, 'g': 68, 'b': 194}}]
    frame_count = 15
    renderer.update_scenegraph_with_segmentation_to_object_associations(ai2thor_mcs_scenegraph, object_list,
                                                                        frame_count)
    print(json.dumps(ai2thor_mcs_scenegraph, indent=4))
    updated_obect = ai2thor_mcs_scenegraph['objects'][0]['shows'][15].get('color', {})


if __name__ == '__main__':
    test_update_scenegraph_with_segmentation_to_object_associations()