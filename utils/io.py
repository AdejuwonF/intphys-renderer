import json
import os

import yaml


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def write_serialized(var, file_name):
    """Write json and yaml file"""
    assert file_name is not None
    if file_name.endswith(".json"):
        with open(file_name, "w") as f:
            json.dump(var, f, indent=4)
    elif file_name.endswith(".yaml"):
        with open(file_name, "w") as f:
            yaml.safe_dump(var, f, indent=4)
    else:
        raise ValueError("Unrecognized filename extension", file_name)

def read_serialized(file_name):
    with open(file_name, "r") as f:
        if file_name.endswith(".json"):
            return json.load(f)
        elif file_name.endswith(".yaml"):
            return yaml.full_load(f)
        else:
            raise NotImplementedError
