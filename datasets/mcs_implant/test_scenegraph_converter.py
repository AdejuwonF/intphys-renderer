from mcs_implant.scenegraph_converter import ScenegraphConverter


def test_rendering():
    converter = ScenegraphConverter()
    test_coords_001 = converter.convert_coordinates_from_cora_to_ai2thor(-10, -10, 3)
    print(test_coords_001)

test_rendering()
