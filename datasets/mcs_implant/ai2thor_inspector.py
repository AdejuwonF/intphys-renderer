import codecs
import json
from os import walk


class McsInspector:
    """
    Inspector used to extract materials and object types from provided scenegraphs. Needed since object types were not
    documented when we started experimenting. Also helped find undocumented options for wider dataset generation.
    """
    materials_ceramics_documented = {'AI2-THOR/Materials/Ceramics/ConcreteBoards1',
                                     'AI2-THOR/Materials/Ceramics/GREYGRANITE',
                                     'AI2-THOR/Materials/Ceramics/KitchenFloor',
                                     'AI2-THOR/Materials/Ceramics/RedBrick',
                                     'AI2-THOR/Materials/Ceramics/TexturesCom_BrickRound0044_1_seamless_S',
                                     'AI2-THOR/Materials/Ceramics/WhiteCountertop'}
    materials_fabrics_documented = {'AI2-THOR/Materials/Fabrics/Carpet2',
                                    'AI2-THOR/Materials/Fabrics/Carpet4',
                                    'AI2-THOR/Materials/Fabrics/CarpetDark',
                                    'AI2-THOR/Materials/Fabrics/CarpetWhite 3',
                                    'AI2-THOR/Materials/Fabrics/HotelCarpet3',
                                    'AI2-THOR/Materials/Fabrics/RugPattern224'}
    materials_metals_documented = {'AI2-THOR/Materials/Metals/Brass 1',
                                   'AI2-THOR/Materials/Metals/BrownMetal 1',
                                   'AI2-THOR/Materials/Metals/GenericStainlessSteel',
                                   'AI2-THOR/Materials/Metals/Metal'}
    materials_plastics_documented = {'AI2-THOR/Materials/Plastics/BlueRubber',
                                     'AI2-THOR/Materials/Plastics/GreenPlastic',
                                     'AI2-THOR/Materials/Plastics/OrangePlastic',
                                     'AI2-THOR/Materials/Plastics/YellowPlastic2'}
    materials_walls_documented = {'AI2-THOR/Materials/Walls/Drywall',
                                  'AI2-THOR/Materials/Walls/DrywallBeige',
                                  'AI2-THOR/Materials/Walls/DrywallOrange',
                                  'AI2-THOR/Materials/Walls/Drywall4Tiled',
                                  'AI2-THOR/Materials/Walls/EggshellDrywall',
                                  'AI2-THOR/Materials/Walls/RedDrywall',
                                  'AI2-THOR/Materials/Walls/WallDrywallGrey',
                                  'AI2-THOR/Materials/Walls/YellowDrywall'}
    materials_woods_documented = {'AI2-THOR/Materials/Wood/BedroomFloor1',
                                  'AI2-THOR/Materials/Wood/LightWoodCounters 1',
                                  'AI2-THOR/Materials/Wood/LightWoodCounters4',
                                  'AI2-THOR/Materials/Wood/TexturesCom_WoodFine0050_1_seamless_S',
                                  'AI2-THOR/Materials/Wood/WoodFloorsCross',
                                  'AI2-THOR/Materials/Wood/WoodGrain_Brown'}
    documented_materials = [materials_ceramics_documented, materials_fabrics_documented, materials_metals_documented,
                            materials_plastics_documented, materials_walls_documented, materials_woods_documented]

    ceiling_materials_extracted = {'Walls/WallDrywallWhite', 'AI2-THOR/Materials/Walls/Drywall'}
    floor_materials_extracted = {'AI2-THOR/Materials/Fabrics/Carpet4', 'Fabrics/RUG4',
                                 'AI2-THOR/Materials/Fabrics/CarpetDark', 'AI2-THOR/Materials/Fabrics/HotelCarpet3',
                                 'AI2-THOR/Materials/Fabrics/CarpetWhite 3', 'AI2-THOR/Materials/Fabrics/RugPattern224'}
    wall_materials_extracted = {'AI2-THOR/Materials/Walls/YellowDrywall', 'AI2-THOR/Materials/Walls/Drywall4Tiled',
                                'AI2-THOR/Materials/Walls/EggshellDrywall', 'AI2-THOR/Materials/Walls/DrywallBeige',
                                'AI2-THOR/Materials/Walls/DrywallOrange', 'Walls/YellowDrywall'}
    material_files_extracted = {'AI2-THOR/Materials/Plastics/GreenPlastic', 'Plastics/BlueRubber',
                                'AI2-THOR/Materials/Ceramics/GREYGRANITE', 'Plastics/GreenPlastic',
                                'AI2-THOR/Materials/Wood/WoodFloorsCross', 'AI2-THOR/Materials/Plastics/OrangePlastic',
                                'AI2-THOR/Materials/Metals/GenericStainlessSteel',
                                'AI2-THOR/Materials/Ceramics/RedBrick', 'AI2-THOR/Materials/Walls/DrywallBeige',
                                'AI2-THOR/Materials/Plastics/YellowPlastic2', 'SourceTextures/Materials/RedBricks',
                                'AI2-THOR/Materials/Wood/WoodGrain_Brown', 'AI2-THOR/Materials/Plastics/BlueRubber'}
    salient_materials_extracted = {'wood', 'plastic', 'metal', 'rubber', 'ceramic'}
    # More candidates: Metal, Wood, Plastic, Glass, Ceramic, Stone, Fabric, Rubber, Food, Paper, Wax, Soap, Sponge,
    #                  Organic - cmp. https://ai2thor.allenai.org/ithor/documentation/objects/material-properties/
    extracted_materials = [ceiling_materials_extracted, floor_materials_extracted, wall_materials_extracted,
                           material_files_extracted, salient_materials_extracted]
    # => Fabrics/RUG4, SourceTextures/Materials/RedBricks, Walls/WallDrywallWhite undocumented
    # [Full list: {'metal', 'ceramic', 'Fabrics/RUG4', 'rubber', 'wood', 'SourceTextures/Materials/RedBricks',
    # 'Walls/WallDrywallWhite', 'Walls/YellowDrywall', 'plastic', 'Plastics/BlueRubber', 'Plastics/GreenPlastic'}]

    # Also see https://ai2thor.allenai.org/ithor/documentation/objects/object-types/
    shape_set_extracted = {'chair_2', 'cup_6', 'sphere', 'sofa_chair_1', 'table_6', 'cup_2', 'table_5', 'cube',
                           'sofa_1', 'plate_1', 'box_2', 'table_1', 'apple_1', 'chair_1', 'apple_2', 'bowl_4', 'box_3',
                           'plate_3', 'bowl_3'}

    def find_objects(self):
        """
        Traverses the scenes in `python_api/scenes` and returns a catalog of all the materials and objects that appear in them.
        Returns:
            A tuple containing:
                    * The set of all ceiling materials that occur in any scene
                    * The set of all floor materials that occur in any scene
                    * The set of all wall materials that occur in any scene
                    * The set of all shapes that occur in any scene
                    * The set of all material files that occur in any scene
                    * The set of all salient materials that occur in any scene
        """
        path: str = '../python_api/scenes'

        ceiling_materials = set()
        floor_materials = set()
        wall_materials = set()

        shape_set = set()
        material_files = set()
        salient_materials = set()

        for (dir_path, dir_names, file_names) in walk(path):
            for file_name in file_names:
                if str(file_name).endswith('.json'):
                    print(f'Parsing {dir_path}/{file_name}...')
                    with open(f'{dir_path}/{file_name}', 'rb') as file:
                        # Some JSON files contain weird characters, so we need to properly decode them
                        file_content = codecs.decode(file.read(), 'utf-8-sig')
                        json_file = json.loads(file_content)

                        ceiling_materials.add(json_file.get('ceilingMaterial'))
                        floor_materials.add(json_file.get('floorMaterial'))
                        wall_materials.add(json_file.get('wallMaterial'))

                        objects = json_file.get('objects', [])
                        for obj in objects:
                            shape_set.add(obj.get('type'))
                            material_files.add(obj.get('materialFile'))
                            for salient_material in obj.get('salientMaterials', []):
                                salient_materials.add(salient_material)
        return ceiling_materials, floor_materials, wall_materials, shape_set, material_files, salient_materials

    def find_objects_print(self):
        """
        Wrapper to call find_objects() and print its findings to stdout.
        Returns:

        """
        ceiling_materials, floor_materials, wall_materials, shape_set, material_files, salient_materials = \
            self.find_objects()
        print(f'Ceiling Materials: {ceiling_materials}')
        print(f'Floor Materials: {floor_materials}')
        print(f'Wall Materials: {wall_materials}')
        print(f'Shapes: {shape_set}')
        print(f'Material Files: {material_files}')
        print(f'Salient Materials: {salient_materials}')

    def compare_documented_extracted(self):
        """
        Compares known materials and objects with extracted ones.
        Returns:
            None - prints to stdout.
        """
        documented_set = set().union(*self.documented_materials)
        extracted_set = set().union(*self.extracted_materials)
        # Following line same as: extracted_set.intersection(documented_set)
        print(f'Common elements: {documented_set.intersection(extracted_set)}')
        only_extracted = {item for item in extracted_set if item not in documented_set}
        only_documented = {item for item in documented_set if item not in extracted_set}
        print(f'Elements documented, but not extracted: {only_documented}')
        print(f'Elements extracted, but not documented: {only_extracted}')


if __name__ == '__main__':
    inspector = McsInspector()
    inspector.compare_documented_extracted()

