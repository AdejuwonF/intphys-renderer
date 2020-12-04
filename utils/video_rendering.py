import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_video", action="store_true")
    parser.add_argument("--input_folder", "-if", type=str, help="folder containing per video folders with pngs rendered from the output of the derender")
    parser.add_argument("--out_dir", "-o", type=str, help="output directory")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    if  not args.multi_video:
        video_folder = args.input_folder
        out_path = args.out_dir
        print("starting with {}".format(video_folder))
        render_command = ["ffmpeg", "-framerate", "3", "-i",
                          "{}/scene_%03d.png".format(video_folder),
                          # "{}/%05d_rgba.png".format(video_folder),
                          "-c:v libx264 -vf fps=25 -pix_fmt yuv420p",
                          "{}.mp4".format(out_path)]
        render_command = " ".join(render_command)
        print(render_command)
        process = subprocess.Popen(render_command, shell=True)
        _, _ = process.communicate()
        print("done")
    else:
        for folder in os.listdir(args.input_folder):
            print("starting with {}".format(folder))
            video_folder = os.path.join(args.input_folder,folder)
            out_path = os.path.join(args.out_dir, folder)
            render_command = ["ffmpeg", "-framerate", "10", "-i",
                              # "{}/scene_%03d.png".format(video_folder),
                              "{}/%05d_rgba.png".format(video_folder),
                              "-c:v libx264 -vf fps=25 -pix_fmt yuv420p",
                              "{}.mp4".format(out_path)]
            render_command = " ".join(render_command)
            print(render_command)
            process = subprocess.Popen(render_command, shell=True)
            _, _ = process.communicate()
            print("done")