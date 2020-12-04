import json
import random

from json import encoder
from pathlib import Path
from typing import Dict

import logging as log


class ScenegraphGenerator:
    """
    Generates AI2Thor MCS Scenegraphs

    Example:
        scenegraph_generator = ScenegraphGenerator()
        scenegraph_generator.generate_scenegraphs_in_bulk(9999, 'generated_scenegraphs_minimal', True)
    """
    materials_ceramics = ['AI2-THOR/Materials/Ceramics/ConcreteBoards1',
                          'AI2-THOR/Materials/Ceramics/GREYGRANITE',
                          'AI2-THOR/Materials/Ceramics/KitchenFloor',
                          'AI2-THOR/Materials/Ceramics/RedBrick',  # SourceTextures/Materials/RedBricks
                          'AI2-THOR/Materials/Ceramics/TexturesCom_BrickRound0044_1_seamless_S',
                          'AI2-THOR/Materials/Ceramics/WhiteCountertop']
    materials_fabrics = ['AI2-THOR/Materials/Fabrics/Carpet2',
                         'AI2-THOR/Materials/Fabrics/Carpet4',
                         'AI2-THOR/Materials/Fabrics/CarpetDark',
                         'AI2-THOR/Materials/Fabrics/CarpetWhite 3',
                         'AI2-THOR/Materials/Fabrics/HotelCarpet3',
                         'AI2-THOR/Materials/Fabrics/RugPattern224',
                         'Fabrics/RUG4']
    materials_metals = ['AI2-THOR/Materials/Metals/Brass 1',
                        'AI2-THOR/Materials/Metals/BrownMetal 1',
                        'AI2-THOR/Materials/Metals/GenericStainlessSteel',
                        'AI2-THOR/Materials/Metals/Metal']
    materials_plastics = ['AI2-THOR/Materials/Plastics/BlueRubber',
                          'AI2-THOR/Materials/Plastics/GreenPlastic',
                          'AI2-THOR/Materials/Plastics/OrangePlastic',
                          'AI2-THOR/Materials/Plastics/YellowPlastic2']
    materials_walls = ['AI2-THOR/Materials/Walls/Drywall',
                       'AI2-THOR/Materials/Walls/DrywallBeige',
                       'AI2-THOR/Materials/Walls/DrywallOrange',
                       'AI2-THOR/Materials/Walls/Drywall4Tiled',
                       'AI2-THOR/Materials/Walls/EggshellDrywall',
                       'AI2-THOR/Materials/Walls/RedDrywall',
                       'AI2-THOR/Materials/Walls/WallDrywallGrey',
                       'Walls/WallDrywallWhite',
                       'AI2-THOR/Materials/Walls/YellowDrywall']
    materials_woods = ['AI2-THOR/Materials/Wood/BedroomFloor1',
                       'AI2-THOR/Materials/Wood/LightWoodCounters 1',
                       'AI2-THOR/Materials/Wood/LightWoodCounters4',
                       'AI2-THOR/Materials/Wood/TexturesCom_WoodFine0050_1_seamless_S',
                       'AI2-THOR/Materials/Wood/WoodFloorsCross',
                       'AI2-THOR/Materials/Wood/WoodGrain_Brown']
    materials = materials_ceramics + materials_fabrics + materials_metals + materials_plastics + materials_walls + \
        materials_woods
    # Also see https://ai2thor.allenai.org/ithor/documentation/objects/object-types/
    object_types = ['chair_2', 'cup_6', 'sphere', 'sofa_chair_1', 'table_6', 'cup_2', 'table_5', 'cube',
                    'sofa_1', 'plate_1', 'box_2', 'table_1', 'apple_1', 'chair_1', 'apple_2', 'bowl_4', 'box_3',
                    'plate_3', 'bowl_3']
    object_types_restricted = ['sphere', 'cube']
    salient_materials = ['wood', 'plastic', 'metal', 'ceramic', 'rubber']

    def __init__(self):
        """
        Initializes scenegraph renderer with feature flags
        """
        # Feature flags for object generation
        self.flag_movable = True
        self.flag_pickupable = False  # Can lead to problems in MCS when object like sofa is flagged pickupable -> False
        self.flag_rotate = True
        self.flag_scale = True

    def generate_scenegraph_dict(self, minimal: bool = False) -> Dict:
        """
        Generate scenegraph as dictionary

        Args:
            minimal: Whether to restrict itself to basic shapes and most likely visible parameter space

        Returns:
            Scenegraph in dictionary form
        """
        # Throw dices to draw decisions: ceiling material (wall), floor mat. (fabric), wall mat. (wall), no. of objects
        if minimal:
            number_of_objects = random.randint(2, 4)
        else:
            number_of_objects = random.randint(2, 10)
        ceiling_material_index = random.randint(1, len(self.materials_walls)) - 1
        ceiling_material = self.materials_walls[ceiling_material_index]
        floor_material_index = random.randint(1, len(self.materials_fabrics)) - 1
        floor_material = self.materials_walls[floor_material_index]
        wall_material_index = random.randint(1, len(self.materials_walls)) - 1
        wall_material = self.materials_walls[wall_material_index]

        scenegraph: Dict = {'ceilingMaterial': ceiling_material, 'floorMaterial': floor_material,
                            'wallMaterial': wall_material}
        # Add 'objects' list
        object_list = []
        for i in range(0, number_of_objects):
            object_list.append(self.generate_object(f'index_{i}', minimal))
        scenegraph['objects'] = object_list

        return scenegraph

    def generate_object(self, object_id: str, minimal: bool = False) -> Dict:
        """
        Generates individual object for scenegraph

        Args:
            object_id: ID to assign to new object
            minimal: Whether to restrict itself to basic shapes and most likely visible parameter space

        Returns:
            Dictionary of new object
        """
        # Throw dices for type, movable, pickupable, mass, x, y, z, scale x, scale y, scale z, rot x, rot y, rot z
        if minimal:
            type_index = random.randint(1, len(self.object_types_restricted)) - 1
            obj_type = self.object_types_restricted[type_index]
        else:
            type_index = random.randint(1, len(self.object_types)) - 1
            obj_type = self.object_types[type_index]

        mass = random.uniform(0.5, 5)

        object_spec = {
            'id': object_id,
            'type': obj_type,
            'mass': mass
        }

        if not minimal and self.flag_movable:
            movable = random.random() < 0.5
            if movable:
                object_spec['moveable'] = True

        if not minimal and self.flag_pickupable:
            pickup = self.__pick_pickupable()
            if pickup is not None and len(pickup) == 2:
                object_spec['salientMaterials'] = pickup[0]
                object_spec['materialFile'] = pickup[1]
                object_spec['pickupable'] = True
        elif minimal:
            material_index = random.randint(1, len(self.materials)) - 1
            object_spec['materialFile'] = self.materials[material_index]

        if minimal:
            z = random.uniform(2, 4)
            x_max = 0.5*z - 0.5  # Empirically found x=-1.5 to 1.5 for z=4, x=-1 to 1 for z=3, x=-0.5 to 0.5 for z=2
            x = random.uniform(-x_max, x_max)
            y = random.uniform(0, x_max)
        else:
            # Full room coordinate range should be [-5, 5] for x and z as well as [0, 3] for y
            x = random.uniform(-5, 5)
            y = random.uniform(0, 3)
            z = random.uniform(-5, 5)
        # We just begin at 0 for now
        shows = [{'stepBegin': 0, 'position': {'x': x, 'y': y, 'z': z}}]

        if self.flag_rotate and random.random() < 0.5:  # If flag enabled and roll succeeds, add rotation
            # => Need to roll pitch, roll, yaw
            rot_x = random.randint(0, 180)
            rot_y = random.randint(0, 180)
            rot_z = random.randint(0, 180)
            shows[0]['rotation'] = {'x': rot_x, 'y': rot_y, 'z': rot_z}

        if self.flag_scale and random.random() < 0.5:  # If flag enabled and roll succeeds, add scale
            # => Need to roll scale x, scale y, scale z
            if minimal:
                scale_x = random.uniform(0.05, 0.5)
                scale_y = random.uniform(0.05, 0.5)
                scale_z = random.uniform(0.05, 0.5)
            else:
                scale_x = random.uniform(0.025, 2)
                scale_y = random.uniform(0.025, 2)
                scale_z = random.uniform(0.025, 2)
            shows[0]['scale'] = {'x': scale_x, 'y': scale_y, 'z': scale_z}

        object_spec['shows'] = shows

        return object_spec

    def __pick_pickupable(self):
        """
        Generates parameters that usually appear together (material_file, salient_material and pickupable)
        Returns:
            Salient material and material file if pickupable, None otherwise
        """
        # Note that the entry says salientMaterials (plural) and takes a list, so this might take multiple values in the
        # future
        pickupable = random.random() < 0.5  # => Need to roll salientMaterials, materialFile
        if pickupable:
            salient_material_index = random.randint(1, len(self.salient_materials)) - 1
            salient_material = self.salient_materials[salient_material_index]
            if salient_material == 'rubber':
                # Note: Currently if 'rubber' is picked, we always choose AI2-THOR/Materials/Plastics/BlueRubber
                material_file = 'AI2-THOR/Materials/Plastics/BlueRubber'
            elif salient_material == 'wood':
                material_file_index = random.randint(1, len(self.materials_woods)) - 1
                material_file = self.materials_woods[material_file_index]
            elif salient_material == 'plastic':
                material_file_index = random.randint(1, len(self.materials_plastics)) - 1
                material_file = self.materials_plastics[material_file_index]
            elif salient_material == 'metal':
                material_file_index = random.randint(1, len(self.materials_metals)) - 1
                material_file = self.materials_metals[material_file_index]
            elif salient_material == 'ceramic':
                material_file_index = random.randint(1, len(self.materials_ceramics)) - 1
                material_file = self.materials_ceramics[material_file_index]
            else:
                log.warning(f'Impossible value encountered in pickupable generation: {salient_material}')
                return None
            return [salient_material], material_file
        return None

    def save_scenegraph_dict_to_json(self, dictionary, target_file_path: str, pretty: bool = True) -> None:
        """
        Saves generated scenegraph to JSON file.
        Args:
            dictionary: Dictionary with scenegraph
            target_file_path: Path to target file location (i.e. of JSON to write)
            pretty: whether to indent JSON

        Returns:
            None
        """
        encoder.FLOAT_REPR = lambda o: format(o, '.2f')
        with open(target_file_path, 'w') as target_file:
            if pretty:
                json.dump(dictionary, target_file, indent=4)
            else:
                json.dump(dictionary, target_file)

    def generate_scenegraphs_in_bulk(self, number_of_scenegraphs: int, target_directory: str, minimal: bool = False) \
            -> None:
        """
        Generates scenegraphs in bulk into target directory

        Args:
            number_of_scenegraphs: How many scenegraphs to generate
            target_directory: Which directory to place scenegraphs into
            minimal: Whether to restrict itself to basic shapes and most likely visible parameter space

        Returns:
            None
        """
        target_path = Path(target_directory)
        target_path.mkdir(parents=True, exist_ok=True)
        for i in range(0, number_of_scenegraphs):
            file_path = target_path / f'scenegraph_{i}.json'
            dictionary = self.generate_scenegraph_dict(minimal)
            self.save_scenegraph_dict_to_json(dictionary, str(file_path))
