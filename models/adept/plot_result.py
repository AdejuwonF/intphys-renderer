import argparse
import os
import shutil
from itertools import repeat
from multiprocessing import Process, cpu_count
from multiprocessing.pool import ThreadPool

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from result_analysis.render_comparative_videos import render_video
from utils.io import mkdir, read_serialized

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str)
    return parser.parse_args()

def get_images(data_folder,video_result):
    images_folder = video_result["original_video"].split(os.path.basename(data_folder))[1]
    images_folder = images_folder[1:] #remove the leading slash
    images_folder = os.path.join(data_folder, images_folder, "imgs")
    images_files = map(lambda x: os.path.join(images_folder,x), sorted(os.listdir(images_folder)))
    images = list(map(Image.open, images_files))[5:]
    return images

# def plot_case(images, all_scores, raw_scores, locations, derender_objects, gt_objects, case_name,
#               output_folder):
# images, score["all"], score["raw"], score["location"], scenes[1:], [None] * len(images),
#               case_name, output_folder
def plot_result(data_folder,video_file,output_folder):
    video_result = read_serialized(video_file)
    images = get_images(data_folder,video_result)
    raw_scores = video_result["scores"]["raw"]
    locations = video_result["scores"]["location"]
    all_scores = video_result["scores"]["all"]
    case_name = video_result["original_video"].replace("/","--")

    derender_objects = read_serialized(video_result["scene_file"])["scene_states"]

    fig, (ax1, ax3, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(4.5, 10))
    line, = ax2.plot([], [], "k")
    images_folder = "{}/.tmp_imgs/".format(output_folder)
    shutil.rmtree(images_folder,ignore_errors=True)
    mkdir(images_folder)

    for i, (image, raw_score, xs, ys, derender_object, gt_object) in enumerate(
            zip(images, raw_scores, locations[0], locations[1], derender_objects, repeat(None)), 1):
        ax1.imshow(image)
        ax1.axis('off')

        ax2.clear()
        line.set_xdata(range(i))
        line.set_ydata(all_scores[:i])
        ax2.plot(range(i), all_scores[:i])
        ax2.axvline(x=i, color="r", linestyle='--')
        plt.draw()

        perturbed_score = []
        for score in raw_score:
            perturbed_score.append(score + np.random.rand() * .001)
        bp = ax2.boxplot(perturbed_score, positions=[i], showfliers=False, showcaps=False, whis=[25, 75])
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color="#1f77b4")

        ax2.set_xlim(0, len(images))
        ax2.set_ylim(0, 12)
        ax2.get_xaxis().set_ticklabels([])
        ax2.axes.get_yaxis().set_ticklabels([])

        ax3.clear()
        ax3.scatter(ys, [-x for x in xs], 40, alpha=.2)

        derender_xs = [obj["location"][1] for obj in derender_object["objects"]]
        derender_ys = [-obj["location"][0] for obj in derender_object["objects"]]
        ax3.scatter(derender_xs, derender_ys, 10)

        if gt_object is not None:
            gt_xs = [obj["location"][1] for obj in gt_object["objects"]]
            gt_ys = [-obj["location"][0] for obj in gt_object["objects"]]
            ax3.scatter(gt_xs, gt_ys, 10)

        ax3.set_xlim(-4, 4)
        ax3.set_ylim(-1., 2.5)

        ax3.get_xaxis().set_ticklabels([])
        ax3.get_yaxis().set_ticklabels([])
        os.makedirs(output_folder,exist_ok=True)
        fig.savefig(os.path.join(images_folder,"{:05d}.png".format(i)))
        print("{}/.tmp_imgs/{:05d}.png generated".format(output_folder, i))
    fig.savefig("{}/{}_score.png".format(output_folder, case_name))
    render_video(images_folder,output_folder,case_name)


if __name__ == "__main__":
    args = parse_args()
    plot_result(args.data_folder,args.result_file,args.output_folder)
