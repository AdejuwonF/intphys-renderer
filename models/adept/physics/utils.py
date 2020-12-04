
TYPES2SHAPES = {
    'occluder': 'cube',
    'object': 'sphere',
}

COLORS2RGB = {
    "red": [173, 35, 35],
    "blue": [42, 75, 215],
    "green": [29, 105, 20],
    "brown": [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan": [41, 208, 208],
    "yellow": [255, 238, 51]
}


def iou(m1, m2):
    intersect = m1 * m2
    union = 1 - (1 - m1) * (1 - m2)
    return intersect.sum() / union.sum()


def reverse_xyz(t):
    """Point an 3d vector to the opposite direction"""
    return [-t[0], -t[1], -t[2]]


def reverse_euler(t):
    """Point a xyz euler to the opposite direction"""
    return [-t[2], -t[1], -t[0]]