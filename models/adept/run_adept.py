import os
import subprocess
from collections import defaultdict
from copy import deepcopy
from itertools import repeat, chain
from multiprocessing import Pool, cpu_count, Manager, Process
from time import sleep

import numpy as np
from easydict import EasyDict
from sklearn.metrics import roc_auc_score
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from PIL import Image

from datasets.adept import adept_group_by_control_surprise
from datasets.intphys import intphys_group_by_control_surprise
from datasets.json_generator import get_jsons_directory
from models.adept.particle_filter import FilterUpdater
from utils.io import read_serialized, write_serialized

CONTROL_SURPRISE_GROUPERS = {"intphys": intphys_group_by_control_surprise,
                             "adept": adept_group_by_control_surprise}

def send_results_to_tim(out_file, tim_key):
    tries = 0
    while True:
        pr = subprocess.call(["sshpass",
                         "-p",
                         tim_key,
                         "rsync",
                         "-ratlz",
                         "--relative",
                         out_file,
                         "aldo@problem-manipulator.sl.cloud9.ibm.com:/home/aldo/cora-derender/"])
        if pr==0:
            return
        # elif tries > 10:
            # break
        sleep(0.5)
        tries += 1


def split_auc_scores(results_group):
    data = [(r["is_possible"], r["max_score"]) for r in results_group]
    labels, scores = map(np.array,zip(*data))
    scores[scores==float('inf')] = 999999.0
    auc = roc_auc_score(labels, -scores).item()
    return auc

def relative_score(control_surprise_group):
    per_label = defaultdict(list)
    [per_label[c["is_possible"]].append(c['scores']["max"]) for c in control_surprise_group]

    correct_order = 0.0
    for pos in per_label[True]:
        for neg in per_label[False]:
            if neg>pos:
                correct_order+=1.0
    return correct_order/(len(per_label[True])*len(per_label[False]))


def encode_mask_all_objects(scene_state):
    for obj in  scene_state["objects"]:
        obj["mask"]["counts"] = obj["mask"]["counts"].encode("ascii")
    return scene_state

def compute_scores(cfg,video_dict, n_filter, out_dir, tim_key, distributed):
    video_dict = deepcopy(video_dict)
    scene_dict = read_serialized(video_dict["scene_file"])
    video_dict.update(scene_dict["debug"])
    observations = scene_dict['scene_states']
    observations = list(map(encode_mask_all_objects, observations))

    initial_belief = observations[0]["objects"]
    camera = EasyDict(observations[0]["suggested_view"]["camera"])

    filter = FilterUpdater(cfg.MODULE_CFG, initial_belief, camera, n_filter)
    if cfg.MODULE_CFG.DEBUG:
        filter.run(observations[1:])
    else:
        filter.run(observations[1:])
    score = filter.get_score()
    video_dict["scores"] = score

    results_folder = os.path.join(out_dir,video_dict["perception"])
    out_file = scene_dict["debug"]["original_video"].replace("/","--")+".json"
    out_file = os.path.join(results_folder, out_file)
    os.makedirs(results_folder, exist_ok=True)

    write_serialized(video_dict,out_file)

    if distributed:
        send_results_to_tim(out_file, tim_key)

    print("done with {}".format(out_file))
    return video_dict


def run_adept(cfg, rank, num_machines, tim_key, distributed):
    if cfg.MODULE_CFG.ANALYZE_RESULTS_FOLDER == "None":
        all_scenes = []
        for attributes_key in cfg.MODULE_CFG.ATTRIBUTES_KEYS:
            for dataset_name in cfg.MODULE_CFG.DATASETS.TEST:
                dataset_jsons_dir = get_jsons_directory(cfg.DATA_CFG,
                                                      "adept",
                                                      attributes_key,
                                                      dataset_name)

                dataset_files = sorted(os.listdir(dataset_jsons_dir))
                all_scenes.extend([{"scene_file":os.path.join(dataset_jsons_dir,d),
                                    "dataset_split":dn,
                                    "perception":attr}
                                   for d,dn,attr in zip(dataset_files,
                                                        repeat(dataset_name),
                                                        repeat(attributes_key))])
        manager = Manager()
        n_filter = manager.Semaphore(1)
        if cfg.MODULE_CFG.DEBUG:
            if len(cfg.MODULE_CFG.DEBUG_VIDEOS) > 0:
                all_scenes = [s for s in all_scenes if s["scene_file"] in cfg.MODULE_CFG.DEBUG_VIDEOS]
            results = [compute_scores(*w) for w in zip(repeat(cfg),
                                                       all_scenes,#[:3][rank::num_machines],
                                                       repeat(n_filter),
                                                       repeat(cfg.MODULE_CFG.OUTPUT_DIR),
                                                       repeat(tim_key),
                                                       repeat(distributed))]
        else:
            with Pool(int(cpu_count())) as p:
                results = p.starmap(compute_scores, zip(repeat(cfg),
                                                       all_scenes[rank::num_machines],
                                                       repeat(n_filter),
                                                       repeat(cfg.MODULE_CFG.OUTPUT_DIR),
                                                        repeat(tim_key),
                                                       repeat(distributed)))

        # send_results_to_tim(cfg.MODULE_CFG.OUTPUT_DIR, tim_key)
            # write_serialized(results,os.path.join(cfg.MODULE_CFG.OUTPUT_DIR,
            #                                       str(attributes_key)+"results.json"))
    else:
        cfg.MODULE_CFG.OUTPUT_DIR = cfg.MODULE_CFG.ANALYZE_RESULTS_FOLDER
        # results = read_serialized(os.path.join(cfg.MODULE_CFG.OUTPUT_DIR, "results.json"))
    if not distributed:
        base_dir = cfg.MODULE_CFG.OUTPUT_DIR
        for attributes_key in cfg.MODULE_CFG.ATTRIBUTES_KEYS:
            # group by dataset
            attri_dir = os.path.join(base_dir,attributes_key)
            results = [read_serialized(os.path.join(attri_dir,v)) for v in os.listdir(attri_dir)]

            #group by matched surprise/control for relative scores
            group_by_control_surprise = CONTROL_SURPRISE_GROUPERS[cfg.DATA_CFG.BASE_NAME]
            grouped_dataset = group_by_control_surprise(results)
            scores_per_stimuli = defaultdict(list)
            for stimuli in grouped_dataset:
                for control_surprise_g in grouped_dataset[stimuli]:
                    g_score = relative_score(grouped_dataset[stimuli][control_surprise_g])
                    scores_per_stimuli[stimuli].append(g_score)
                    scores_per_stimuli['total'].append(g_score)
            avg_relative_scores = {k: bs.bootstrap(np.array(v), stat_func=bs_stats.mean)
                                   for k, v in scores_per_stimuli.items()}


            write_serialized(avg_relative_scores,
                             os.path.join(cfg.MODULE_CFG.OUTPUT_DIR,
                                          str(attributes_key) + "_relative_scores.json"))

            print(scores_per_stimuli)
            # results = read_serialized("/all/home/aldo/cora-derenderer/output/adept/adept/exp_00001/pred_attr_43044results.json")
            # grouped_results = {}
            # for dataset_name in cfg.MODULE_CFG.DATASETS.TEST:
            #     grouped_results[dataset_name] = [{"index":i,
            #                                       "original_video":r["original_video"],
            #                                       "max_score": r["scores"]["max"],
            #                                       "mean_score": r["scores"]["mean"],
            #                                       "is_possible": r["is_possible"]}
            #                                      for i,r in enumerate(results) if r["dataset_split"] == dataset_name]
            # grouped_results["total"] = list(chain.from_iterable(grouped_results.values()))
            # auc_scores = {k:split_auc_scores(v) for k,v in grouped_results.items()}
            #
            # write_serialized(auc_scores, os.path.join(cfg.MODULE_CFG.OUTPUT_DIR,
            #                                           str(attributes_key) + "_auc_scores.json"))
            #
            # #group by matched surprise/control for relative scores
            # group_by_control_surprise = CONTROL_SURPRISE_GROUPERS[cfg.DATA_CFG.BASE_NAME]
            # grouped_results = {k: group_by_control_surprise(v) for k, v in grouped_results.items()}
            # relative_scores = {k: {q:relative_score(p) for q,p in v.items()} for  k,v in grouped_results.items()}
            # relative_scores = {"per_group": relative_scores,
            #                    "average_per_dataset": {k: np.array(list(q.values())).mean().item()
            #                                            for k, q in relative_scores.items()}}
            #



