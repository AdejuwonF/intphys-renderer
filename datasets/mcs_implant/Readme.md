# MCS Implant

## Setup
The files in this directory are meant to be used from within MCS. Please clone it and run this code from within MCS,
i.e. `MCS/mcs_implant`.

## Usage
### Bulk Rendering of First-Frame Scenes
To render 99999 new scenes, use:
```
renderer = AI2ThorRenderer()
renderer.create_first_frame_until_n('<some_path>/generated_scenegraphs_renderings/', 99999, 1)
```

1. Render example scenes via agents
```
    renderer = AI2ThorRenderer()
    renderer.render_example_scenes()
```

2. Render minimal scenegraphs

- 2.1 First run scenegraph generator via
```
    scenegraph_generator = ScenegraphGenerator()
    scenegraph_generator.generate_scenegraphs_in_bulk(10, 'generated_scenegraphs_minimal', True)
```
- 2.2 Then run renderer via
```
    renderer = AI2ThorRenderer()
    renderer.render_first_frame_in_bulk_for_directory('generated_scenegraphs_minimal/')
```

### Conversion from Cora to AI2Thor MCS

You can first convert the scenegraphs with the `ScenegraphConverter`: 
```python
    converter = ScenegraphConverter()
    converter.convert_cora_dataset_folder_to_ai2thor('/home/fplk/data/subset_pybullet_input',
                                                     '/home/fplk/data/subset_pybullet_scenegraphs')
```
Afterwards, you can render out the resulting scenegraphs via `AI2ThorRenderer`: 
```python
    renderer = AI2ThorRenderer()
    input_data_path = pathlib.Path('/home/fplk/data/subset_pybullet_scenegraphs')
    scenes = sorted(input_data_path.glob('**/*.json'))
    renderer.render_via_agent(scenes, '/home/fplk/data/subset_rendered_in_ai2thor_mcs', True)
```

## Other MCS Notes
Run human loop with
```shell
mcs_run_in_human_input_mode ${MCS_ROOT_PATH}AI2Thor_MCS/MCS-AI2-THOR-Unity-App-v0.0.3.x86_64 ~/data/subset_rendered_in_ai2thor_mcs/video_00001/scenegraph.json
or
mcs_run_in_human_input_mode ${MCS_ROOT_PATH}AI2Thor_MCS/MCS-AI2-THOR-Unity-App-v0.0.3.x86_64 ${MCS_AI2THOR_MCS_PATH}python_api/scenes/playroom.json
```

If you get the error
```
pkg_resources.DistributionNotFound: The 'machine-common-sense' distribution was not found and is required by the
application
```
Please reinstall MCS via `pip install git+https://github.com/NextCenturyCorporation/MCS@latest`.
Most likely you needed an upgrade.

If you run into
```
File "/home/fplk/.pyenv/versions/3.7.6/lib/python3.7/site-packages/machine_common_sense/run_mcs_human_input.py", line 59, in input_commands
    userInput = previous_output.action_list[1]
IndexError: list index out of range
```
Please check whether you have an overly restrictive action list in your JSON file, e.g. one only consisting of `Pass`
entries. If so, please copy it and remove the entire goal entry and you should be able to run accordingly.