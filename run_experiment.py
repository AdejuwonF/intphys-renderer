import argparse

# from datasets.shapes_world_detection import build_dataset
# from datasets.utils import get_datasets_splits_names
import logging as log
import os
import sys

# from detectron2.engine import launch, default_setup

from datasets.build_dataset import build_dataset
from models.adept.run_adept import run_adept
from trainers import build_trainer
from utils.misc import setup_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module",
                        choices=["derender", "detection", "dynamics", "adept"])
    parser.add_argument("--dataset",
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
    return parser.parse_args()

def main(args):
    cfg = setup_cfg(args, args.distributed)
    # default_setup(cfg.MODULE_CFG, args)

    build_dataset(cfg)

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    print(sys.argv)
    args = parse_args()
    if args.module == 'adept':
        cfg = setup_cfg(args, args.distributed)
        log.basicConfig(filename='log', level=log.INFO)
        build_dataset(cfg)
        run_adept(cfg,
                  args.rank,
                  args.num_machines,
                  args.tim_key,
                  args.distributed)

    elif args.distributed:
        launch(
            main,
            args.num_gpus,
            num_machines= args.num_machines,
            machine_rank= args.rank,
            dist_url= args.dist_url,
            args = (args,)
        )
    else:
        main(args)

