from .shapes_world_attributes import ShapesWorldAttributes

_ATTRIBUTES_MAP = {"shapes_world": ShapesWorldAttributes}

def build_attributes(cfg):
    return _ATTRIBUTES_MAP[cfg.ATTRIBUTES.NAME](cfg)
