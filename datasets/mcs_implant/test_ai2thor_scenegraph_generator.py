from datasets.mcs_implant.ai2thor_scenegraph_generator import ScenegraphGenerator


def test_scenegraph_generation():
    scenegraph_generator = ScenegraphGenerator()
    scenegraph = scenegraph_generator.generate_scenegraph_dict(True)
    assert 'ceilingMaterial' in scenegraph.keys(), 'No ceiling material found in scenegraph.'
    assert 'floorMaterial' in scenegraph.keys(), 'No floor material found in scenegraph.'
    assert 'wallMaterial' in scenegraph.keys(), 'No wall material found in scenegraph.'
    assert 'objects' in scenegraph.keys() and len(scenegraph.get('objects', [])) > 0, 'No objects found in scenegraph.'
