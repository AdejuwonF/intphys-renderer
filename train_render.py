import argparse
import traceback
# from datasets.shapes_world_detection import build_dataset
# from datasets.utils import get_datasets_splits_names
import logging as log
import os
import sys
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import time
import wandb
# from detectron2.engine import launch, default_setup

from datasets.build_dataset import build_dataset
from models.adept.run_adept import run_adept
from trainers import build_trainer
from utils.misc import setup_cfg
import json_render_dataloader as jrd

import trainers.trainable_derender
import configs.dataset_config as data_cfg
from datasets import utils, intphys

from utils.misc import setup_cfg, image_based_to_annotation_based, read_image
from run_experiment import parse_args
from structure.derender_attributes import DerenderAttributes

sys.path.append("Recurrent-Interaction-Network")
from cnr.model import Core_CNR as CNR


def parse_args():
    # Arguments for dataset/cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="derender",
                        choices=["derender", "detection", "dynamics", "adept"])
    parser.add_argument("--dataset", default="intphys",
                        choices=["adept", "intphys", "ai2thor-intphys"])
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--num_machines", type=int)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--tim_key", type=str)
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    #Arguments for model
    parser.add_argument("--height", type=int, default=288, help="Height of the output image")
    parser.add_argument("--width", type=int, default=288, help="Width of the output image")
    parser.add_argument("--n_obj", type=int, default=100, help="Maximum number of objects in a scene")
    parser.add_argument("--attr_len", type=int, default=39, help="Size of the attribute vector taken from dataset")
    parser.add_argument("--nc_in", type=int, default=32, help="Desired size of transformed attribute vector, after initial transformation MLP")
    parser.add_argument("--nc_out", type=int, default=0, help="Number of output channels of the network, not including depth channel which is always produced")
    parser.add_argument("--nf", type=int, default=32, help="Number of feature channels for intermediary layers")
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of blocks of coonvoltions in the network")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of layers per block, each layer contains 3 filters")
    parser.add_argument("--depth", action="store_false")
    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--upsample_mode", default="bilinear")
    parser.add_argument("--iterations", type=int, default = 4000000)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--checkpoint_interval", type=int, default=4000)
    parser.add_argument("--lr" , type=float, default=1e-3)
    parser.add_argument("--batch_size",type=int, default=8)

    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--ck", type=str, default=None)


    return parser.parse_args()

def train(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    wandb.init(project='Intphys-Renderer', entity='adejuwonf')#, resume="2n8g6bku", id="2n8g6bku")
    wandb.config.update(args)

    start_iter = 1
    running_loss = 0
    model = nn.DataParallel(CNR(args).to(device))
    # model = CNR(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    """ck = wandb.restore("checkpoint_iteration_3720000.tar")
    d = torch.load(ck.name)
    model.load_state_dict(d["model_state_dict"])
    start_iter = d["iteration"] + 1
    optimizer.load_state_dict(d["optimizer_state_dict"])
    optimizer.lr = args.lr"""


    wandb.watch(model, log="all", log_freq=10)

    train_data = jrd.IntphysJsonTensor(cfg, "_train")
    val_data = jrd.IntphysJsonTensor(cfg, "_val")

    #Test block using split of validation set b/c full training set takes too long
    # json_data = jrd.IntphysJsonTensor(cfg, "_val")
    # train_size = round(len(json_data)*1)
    # val_size = len(json_data) - train_size
    # For now use a manual seed for reproducibility
    # train_data, val_data = torch.utils.data.random_split(json_data, [train_size, val_size], generator = torch.Generator().manual_seed(42))

    # val_data = train_data

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True)

    # val_loader = train_loader

    dataset_utils = jrd.DatasetUtils(val_data, device)
    start = time.time()
    train_iter = iter(train_loader)
    for i in range(start_iter, start_iter + args.iterations):
        batch = next(train_iter, None)
        if (batch is None):
            train_iter = iter(train_loader)
            batch = next(train_iter, None)

        attributes, depth, n_objs = batch[0].to(device), batch[1].to(device), batch[2]

        attributes = dataset_utils.normalize_attr(attributes)

        out  = model(attributes, n_objs)

        loss = criterion(out[1], depth.view(-1, 1, 288, 288))

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss

        if i % args.log_interval == 0:
            running_loss/=args.log_interval
            with torch.no_grad():
                v_loss = 0
                val_iter = iter(val_loader)
                for j in range(len(val_iter)):
                    batch = next(val_iter)
                    attributes, depth, n_objs = batch[0].to(device), batch[1].to(device), batch[2]

                    attributes = dataset_utils.normalize_attr(attributes)
                    out  = model(attributes, n_objs)
                    loss = criterion(out[1], depth.view(-1, 1, 288, 288))

                    v_loss += loss.item()


                v_loss /= len(val_iter)

                rand_index = torch.randint(depth.shape[0], (1,1)).item()
                gt_depth = depth[rand_index].view(288,288).to("cpu")*10
                out_depth = out[1][rand_index].view(288, 288).to("cpu")*10

                wandb.log({"iteration" : i,
                    "training loss" : running_loss,
                    "validation_loss" : v_loss,
                    "examples" : [wandb.Image(gt_depth, caption="Ground Truth"),
                                  wandb.Image(out_depth, caption="Predicted Depth")]})

                print("\nIteration {0}/{1}\nRunning Loss: {2}\nValidation Loss: {3}\n{4} seconds".format(i, start_iter + args.iterations, running_loss, v_loss, time.time()-start))

                del attributes
                del depth
                del n_objs
                del loss
                del v_loss
                torch.cuda.empty_cache()

                running_loss=0
                start = time.time()

        if (i % args.checkpoint_interval) == 0 or i == (start_iter + args.iterations-1):
            d = {"model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "iteration" : i}
            torch.save(d, os.path.join(wandb.run.dir, "checkpoint_iteration_{0}.tar".format(i)))
            wandb.save("checkpoint_iteration_{0}.tar".format(i))

    return model



def main(args):
    cfg = setup_cfg(args, args.distributed)
    model = train(cfg, args)
#    model = CNR(args).to("cuda")

#    val_dataset = jrd.IntphysJsonTensor(cfg, "_val")
#    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False,
#            num_workers=0)
#    batch = next(iter(val_loader))
#    out = model([x.view(1, -1).to("cuda") for x in batch[0]])
#    utils = jrd.DatasetUtils(val_dataset)


if __name__ == "__main__":
    print(sys.argv)
    args = parse_args()
    main(args)
