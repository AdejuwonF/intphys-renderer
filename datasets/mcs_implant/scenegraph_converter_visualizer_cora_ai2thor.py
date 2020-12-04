from typing import Union

import json
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import PIL
from PIL import Image
import streamlit as st
import io


def plot_object_positions(folder_cora: pathlib.PosixPath, folder_ai2thor: pathlib.PosixPath, selected_video: str,
                          selected_frame: int):
    cora_scenegraph_path = folder_cora / selected_video / f'{str(selected_frame + 1).zfill(5)}.json'
    mcs_scenegraph_path = folder_ai2thor / selected_video / 'scenegraph.json'

    with cora_scenegraph_path.open('r') as cora_file:
        cora_scenegraph = json.load(cora_file)

    with mcs_scenegraph_path.open('r') as mcs_file:
        mcs_scenegraph = json.load(mcs_file)

    # block1: cora_x, cora_y, mcs_x, mcs_z, [future: visible_cora, visible_mcs?]
    object_catalog = {}
    cora_objects = cora_scenegraph.get('objects', {})
    for cora_object in cora_objects:
        name = cora_object.get('name', None)
        if name == 'floor' or name is None or name == '':
            continue
        pose6d = cora_object.get('pose6d', {})
        x = pose6d.get('x', None)
        y = pose6d.get('y', None)
        if x is not None and y is not None:
            object_catalog[name] = [x, y]
    mcs_objects = mcs_scenegraph.get('objects', {})
    for mcs_object in mcs_objects:
        name = mcs_object.get('id', None)
        shows = mcs_object.get('shows', [])
        if selected_frame >= len(shows):
            print('WARNING: Tried to access frame number bigger than shows list in MCS scenegraph. Skipping.')
            continue
        position = shows[selected_frame].get('position', {})
        x = position.get('x', None)
        z = position.get('z', None)
        if x is not None and z is not None and name in object_catalog:
            object_catalog[name].extend([x, z])

    for key, value in object_catalog.items():
        print(f'Object {key} positions: {value}')

    cora_x = [o[0] for _, o in object_catalog.items()]
    cora_y = [o[1] for _, o in object_catalog.items()]
    mcs_x = [o[2] for _, o in object_catalog.items()]
    mcs_z_for_y_axis = [o[3] for _, o in object_catalog.items()]

    cora_range_max = 50
    ax1 = plt.subplot(121, aspect='equal', xlim=[-cora_range_max, cora_range_max],
                      ylim=[-cora_range_max, cora_range_max])
    ax1.set_title('Object positions in Cora', fontsize=12, loc='left')
    ax1.set_xticks(np.arange(-cora_range_max, cora_range_max + 1, cora_range_max / 2))
    ax1.set_yticks(np.arange(-cora_range_max, cora_range_max + 1, cora_range_max / 2))
    ax1.scatter(cora_x, cora_y)
    ax1.grid(True)

    mcs_range_max = 4
    ax2 = plt.subplot(122, aspect='equal', xlim=[-mcs_range_max, mcs_range_max], ylim=[-mcs_range_max, mcs_range_max])
    ax2.set_title('Object positions in MCS', fontsize=12, loc='left')
    ax2.set_xticks(np.arange(-mcs_range_max, mcs_range_max + 1, mcs_range_max / 4))
    ax2.set_yticks(np.arange(-mcs_range_max, mcs_range_max + 1, mcs_range_max / 4))
    ax2.scatter(mcs_x, mcs_z_for_y_axis)
    ax2.grid(True)
    # Could save to file...
    # img_file = folder_ai2thor / f'{str(selected_frame_index).zfill(5)}_position'
    # plt.savefig(img_file)
    # ...and create video via: ffmpeg -framerate 24 -i "%05d_position.png" -pix_fmt yuv420p -y video_position.mp4

    buffer = io.BytesIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    pil_image.save(buffer, 'PNG')
    plt.close()
    return buffer.getvalue()


def load_maps_as_image(folder_cora: pathlib.PosixPath, folder_ai2thor: pathlib.PosixPath, selected_video: str,
                       selected_frame_index: int):
    path_to_segmentation_map_backend_001: pathlib.PosixPath = folder_cora / selected_video / \
        f'{str(selected_frame_index).zfill(5)}.seg.npy'
    path_to_depth_map_backend_001: pathlib.PosixPath = folder_cora / selected_video / \
        f'{str(selected_frame_index).zfill(5)}.depth.npy'
    path_to_segmentation_map_backend_002: pathlib.PosixPath = folder_ai2thor / selected_video / \
        f'{str(selected_frame_index).zfill(5)}.seg.npy'
    path_to_depth_map_backend_002: pathlib.PosixPath = folder_ai2thor / selected_video / \
        f'{str(selected_frame_index).zfill(5)}.depth.npy'
    segmentation_map_backend_001 = np.load(str(path_to_segmentation_map_backend_001))
    depth_map_backend_001 = np.load(str(path_to_depth_map_backend_001))
    segmentation_map_backend_002 = np.load(str(path_to_segmentation_map_backend_002))
    depth_map_backend_002 = np.load(str(path_to_depth_map_backend_002))

    # If we wanted to show individual image, we could use plt.imshow(map_array)
    fig, ax_array = plt.subplots(2, 2)
    ax_array[0][0].imshow(segmentation_map_backend_001)
    ax_array[0][1].imshow(depth_map_backend_001)
    ax_array[1][0].imshow(segmentation_map_backend_002)
    ax_array[1][1].imshow(depth_map_backend_002)
    # Could save to file...
    # img_file = folder_ai2thor / f'{str(selected_frame_index).zfill(5)}_maps'
    # plt.savefig(img_file)
    # plt.close(fig)
    # ...and create video via: ffmpeg -framerate 24 -i "%05d_maps.png" -pix_fmt yuv420p -y video_maps.mp4

    buffer = io.BytesIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    pil_image.save(buffer, 'PNG')
    plt.close()
    return buffer.getvalue()

# Basic Declarations
folder_ai2thor = pathlib.Path('/home/fplk/data/subset_rendered_in_ai2thor_mcs')
folder_cora = pathlib.Path('/home/fplk/data/subset_pybullet_input')
NUM_FRAMES = 50

# selected_video_index = 1
# selected_video = f'video_{str(selected_video_index).zfill(5)}'
# video_list = [selected_video]
video_list = ["video_00001",
              "video_00002",
              "video_00003",
              "video_00005",
              "video_00009",
              "video_00011",
              "video_00013",
              "video_00017",
              "video_00020",
              "video_00023",
              "video_00024",
              "video_00025",
              "video_00036",
              "video_00039",
              "video_00040",
              "video_00042",
              "video_00043",
              "video_00044",
              "video_00046",
              "video_00048",
              "video_00053",
              "video_00055",
              "video_00062",
              "video_00065",
              "video_00067",
              "video_00070",
              "video_00078",
              "video_00080",
              "video_00081",
              "video_00082"
]


# GUI Declarations
image = st.empty()
image2 = st.empty()
st.sidebar.markdown('# Conversion Analyzer')
video = st.sidebar.selectbox("Video", video_list, 0)
selected_frame_index = st.sidebar.slider("Frame Number", 1, NUM_FRAMES, 1, 1)
progress_bar = st.sidebar.progress(0)
frame_text = st.sidebar.empty()
button_animate = st.sidebar.button("Animate")

# Update images
maps_as_image = load_maps_as_image(folder_cora, folder_ai2thor, video, selected_frame_index)
position_plot = plot_object_positions(folder_cora, folder_ai2thor, video, selected_frame_index)
image.image(maps_as_image, use_column_width=True, caption='Depth and Segmentation Maps in Cora (top row) and AI2Thor '
                                                          'MCS (bottom row)')
image2.image(position_plot, use_column_width=True, caption='Position plot of Cora (left) and AI2Thor MCS (right)')


if button_animate:
    for i in range(1, NUM_FRAMES + 1):
        progress_bar.progress(int(i*(100/NUM_FRAMES)))  # Slider expects progress to go to 100 - multiply accordingly
        selected_frame_index = i
        frame_text.text(f'Frame {i}/{NUM_FRAMES}')
        maps_as_image = load_maps_as_image(folder_cora, folder_ai2thor, video, selected_frame_index)
        position_plot = plot_object_positions(folder_cora, folder_ai2thor, video, selected_frame_index)
        image.image(maps_as_image, use_column_width=True,
                    caption='Depth and Segmentation Maps in Cora (top row) and AI2Thor '
                            'MCS (bottom row)')
        image2.image(position_plot, use_column_width=True,
                     caption='Position plot of Cora (left) and AI2Thor MCS (right)')
