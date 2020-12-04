from datasets import intphys, shapes_world, adept

def import_constants_from_dataset(object, dataset_base_name, add_object_id=False):
    ######### set the terms and a validation map ##########
    [object.__setattr__(a, eval("{}.{}".format(dataset_base_name,a.upper())))
     for a in ["continuous_terms","categorical_terms","rotation_terms","valid_map","quantized_terms"]]
    if add_object_id and not "object_id" in object.categorical_terms:
        object.categorical_terms += ["object_id"]

    object.terms = object.categorical_terms + object.continuous_terms + object.quantized_terms
    # if "object_id" not in object.terms:
    #     object.terms = []

    ######### each categorical variable except visible must have a string->int map #######
    [object.__setattr__(a,eval("{}.{}".format(dataset_base_name,a.upper())))
     for a in [n+"_map" for n in object.categorical_terms if n not in {"visible", "existance"}]]
    ######### existance and visibility don't depend on the dataset ##########
    [object.__setattr__(a,{0:0, 1:1}) for a in ["visible_map", "existance_map"]]

    ######## all  quantized terms must have a quantization array ############
    [object.__setattr__(a, eval("{}.{}".format(dataset_base_name, a.upper())))
     for a in [n + "_array" for n in object.quantized_terms]]

    ######### import positive terms and ranged terms ##########
    # object.ranged_terms = eval("{}.RANGED_TERMS".format(dataset_base_name))
    object.positive_terms = eval("{}.POSITIVE_TERMS".format(dataset_base_name))

    ######### import camera if required ###############
    if dataset_base_name == "intphys":
        object.camera_terms = eval("{}.CAMERA_TERMS".format(dataset_base_name))
    # else:
    #     object.camera_terms = []

    object.max_num_objects = len(eval("{}.OBJECT_ID_MAP".format(dataset_base_name)))
