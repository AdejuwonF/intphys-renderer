import argparse

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
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--n_obj", type=int, default=100)
    parser.add_argument("--nc_in", type=int, default=39)
    parser.add_argument("--nc_out", type=int, default=0)
    parser.add_argument("--nf", type=int, default=32)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--depth", action="store_false")
    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--upsample_mode", default="bilinear")
    parser.add_argument("--iterations", type=int, default = 200000)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--lr" , type=float, default=1e-2)
    parser.add_argument("--batch_size",type=int, default=8)

    return parser.parse_args()

def train(cfg, args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    wandb.init(project='Intphys-Renderer', entity='adejuwonf')
    wandb.config.update(args)

    model = CNR(args).to(device)
    wandb.watch(model, log="all", log_freq=1)

    #train_data = jrd.IntphysJsonTensor(cfg, "_train")
    #val_data = jrd.IntphysJsonTensor(cfg, "_val")

    #Test block using split of validation set b/c full training set takes too long
    json_data = jrd.IntphysJsonTensor(cfg, "_val")
    train_size = round(len(json_data)*1)
    val_size = len(json_data) - train_size
    #For now use a manual seed for reproducibility
    train_data, val_data = torch.utils.data.random_split(json_data, [train_size, val_size], generator = torch.Generator().manual_seed(42))

    val_data = train_data

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True)

    dataset_utils = jrd.DatasetUtils(val_data, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    train_loss = []
    val_loss = []
    start_iter = 1
    running_loss = 0
    start = time.time()
    for i in range(start_iter, start_iter + args.iterations):

        batch = next(iter(train_loader))
        attributes, depth, n_objs = batch[0].to(device), batch[1].to(device), batch[2]
        # attributes = dataset_utils.normalize_attr(attributes)
        out  = model([attributes[i][:n_objs[i]] for i in range(attributes.shape[0])])
        loss = criterion(out[1], depth.view(-1, 1, 288, 288))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0 and i!=0:
            running_loss/=args.log_interval
            with torch.no_grad():
                v_loss = 0
                for j in range(len(val_loader)):
                    batch = next(iter(val_loader))
                    attributes, depth, n_objs = batch[0].to(device), batch[1].to(device), batch[2]
                    # attributes = dataset_utils.normalize_attr(attributes)
                    out  = model([attributes[i][:n_objs[i]] for i in range(attributes.shape[0])])
                    loss = criterion(out[1], depth.view(-1, 1, 288, 288))

                    v_loss += loss.item()


                v_loss /= len(val_loader)

                rand_index = torch.randint(depth.shape[0], (1,1)).item()
                print(depth.shape[0])
                gt_depth = depth[rand_index].view(288,288).to("cpu")*10
                out_depth = out[1][rand_index].view(288, 288).to("cpu")*10

                wandb.log({"iteration" : i,
                    "training loss" : running_loss,
                    "validation_loss" : v_loss,
                    "examples" : [wandb.Image(gt_depth, caption="Ground Truth"),
                                  wandb.Image(out_depth, caption="Predicted Depth")]})
                del attributes
                del depth
                del n_objs
                del loss
                val_loss.append(v_loss)
                train_loss.append(running_loss)

            print("\nIteration {0}/{1}\nRunning Loss: {2}\nValidatiion Loss: {3}\n{4} seconds".format(i, args.iterations, running_loss, v_loss, time.time()-start))
            running_loss=0
            start = time.time()

        """if (i % (iterations//10)) == 0 or i == iterations:
            d = {"model_state_dict": model.state_dict(),
                 "optimizer_stat_dict": optimizer.state_dict(),
                 "training_loss" : train_loss,
                 "validation_loss": val_loss,
                 "epochs" : epoch}
            torch.save(d, "./checkpoints/renderer_1/checkpoint_epoch_{0}".format(epoch))"""




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
