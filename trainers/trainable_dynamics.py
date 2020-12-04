import datetime
import os
import time

import psutil
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.rnn_dynamics import RNNDynamics
from models.interaction_dynamics import InteractionDynamics
from trainers.dynamics_dataset import DynamicsDataset
from trainers.trainable_derender import Nan_Exception, create_new_timer
from utils.build import make_lr_scheduler, make_optimizer
from utils.checkpoint import Checkpointer
import logging as log

from utils.metric_logger import MetricLogger
from utils.misc import to_cuda
import torch.nn.functional as F
import json

proc_id = psutil.Process(os.getpid())


def build_dynamics_model(cfg, train_dataset):
    if cfg.MODEL.ARCHITECTURE == "rnn":
        return RNNDynamics(cfg, train_dataset)
    elif cfg.MODEL.ARCHITECTURE == "interaction":
        return InteractionDynamics(cfg, train_dataset)


class TrainableDynamics:
    def __init__(self,cfg):
        # TEST = ("intphys_dev-meta_O1",
        #         "intphys_dev-meta_O2",
        #         "intphys_dev-meta_O3")
        val_datasets = {k: DynamicsDataset(cfg, cfg.DATASETS.TEST,k)
                        for k in cfg.ATTRIBUTES_KEYS}
        # train_dataset = DynamicsDataset(cfg, cfg.DATASETS.TRAIN)
        train_dataset = DynamicsDataset(cfg, cfg.DATASETS.TRAIN,
                                        # cfg.DATASETS.ATTRIBUTES_KEYS[0]
                                        #TODO:(YILUN) if you use any other types of attributes seen in
                                        #cfg.ATTRIBUTES_KEYS, you will train on outputs of the derender
                                        #the object_ids will still be ok.
                                        "attributes")
        #TODO: add torch data parallel or distributed or sth
        model = build_dynamics_model(cfg, val_dataset).cuda().train()
        ckpt = torch.load("/all/home/yilundu/repos/cora-derenderer/output/intphys/dynamics/exp_00073/model_0052474.pth")

        model.load_state_dict(ckpt['models'])
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)
        checkpointer = Checkpointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, log)

        workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKERS
        train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE,
                                  num_workers=workers)
        val_loader = DataLoader(val_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE,
                                  num_workers=workers, shuffle=False)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpointer = checkpointer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.output_dir = cfg.OUTPUT_DIR
        self.start_iteration = 0
        self.cfg = cfg

    def train(self):
        # save new  configuration
        with open(os.path.join(self.output_dir, "cfg.yaml"), 'w') as f:
            x = self.cfg.dump(indent=4)
            f.write(x)

        log.info(f'New training run with configuration:\n{self.cfg}\n\n')
        train_metrics = MetricLogger(delimiter="  ")
        summary_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "summary"))

        self.model.train()
        timers = create_new_timer()
        # Initialize timing

        done = False
        while not done:
            for iteration, inputs in enumerate(self.train_loader, self.start_iteration):
                iter_time = time.time()
                data_time = iter_time - timers.batch
                inputs = to_cuda(inputs)

                out = self.model(inputs)
                loss_dict = out['loss_dict']
                loss = loss_dict["loss"]

                if torch.isnan(loss).any():
                    raise Nan_Exception()

                train_metrics.update(**loss_dict)
                summary_writer.add_scalars("train_non_smooth", train_metrics.last_item, iteration)

                batch_time = iter_time - timers.batch
                timers.batch = iter_time
                train_metrics.update(time=batch_time, data=data_time)
                eta_seconds = timers.start + self.cfg.SOLVER.MAX_TIME_SECS - iter_time
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if (iter_time - timers.log > self.cfg.SOLVER.PRINT_METRICS_TIME):
                    timers.log = iter_time
                    log.info(train_metrics.delimiter.join(["eta: {eta}", "iter: {iter}", "{meters}", "lr: {lr:.6f}",
                                                           "max mem: {memory:.0f}"]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(train_metrics),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=proc_id.memory_info().rss / 1e9)
                    )
                    summary_writer.add_scalars("train", train_metrics.mean, iteration)

                if iter_time - timers.checkpoint > self.cfg.SOLVER.CHECKPOINT_SECS:  # iteration % checkpoint_period == 0:
                    timers.checkpoint = iter_time
                    self.checkpointer.save("model_{:07d}".format(iteration))

                if iter_time - timers.tensorboard > self.cfg.SOLVER.TENSORBOARD_SECS or self.cfg.DEBUG:
                    timers.tensorboard = iter_time
                    summary_writer.add_scalars("train", train_metrics.mean, iteration)

                if iter_time - timers.start > self.cfg.SOLVER.MAX_TIME_SECS:
                    log.info("finished training loop in {}".format(iter_time - timers.start))
                    done = True
                    break

                if iter_time - timers.validation > self.cfg.SOLVER.VALIDATION_SECS:
                    err_dict = self.eval(iteration, summary_writer)
                    timers.validation = time.time()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            log.info("*******  epoch done  after {}  *********".format(time.time() - timers.epoch))
            timers.epoch = time.time()
            self.start_iteration = iteration

    def eval(self, iteration, summary_writer):
        start = time.time()
        all_preds = []
        all_labels = []

        evals = []
        with torch.no_grad():
            self.model.eval()
            # self.derenderer.eval()
            val_metrics = MetricLogger(delimiter="  ")
            val_loss_logger = MetricLogger(delimiter="  ")
            for i, inputs in enumerate(self.val_loader, iteration):
                # data_time = time.time() - last_batch_time

                if torch.cuda.is_available():
                    inputs = to_cuda(inputs)

                output = self.model(inputs, match=True)
                loss_dict = output["loss_dict"]
                is_possible = inputs['is_possible']
                magic_penalty = output['magic_penalty']

                for i in range(len(magic_penalty)):
                    frame = {}
                    frame['is_possible'] = bool(is_possible[i])
                    frame['inverse_likelihood'] = float(magic_penalty[i])
                    evals.append(frame)

                # target = inputs['targets']
                # output = output['output']
                # is_possible = inputs['is_possible']

                # loc_x_gt = target['location_x']
                # loc_y_gt = target['location_y']
                # loc_z_gt = target['location_z']

                # output_x = output['location_x'].squeeze()
                # output_y = output['location_y'].squeeze()
                # output_z = output['location_z'].squeeze()
                # existance = target['existance'][:, 1:]

                # loss_trans_x = torch.pow(output_x - loc_x_gt[:, 1:], 2) * existance
                # loss_trans_y = torch.pow(output_y - loc_y_gt[:, 1:], 2) * existance
                # loss_trans_z = torch.pow(output_z - loc_z_gt[:, 1:], 2) * existance

                # loss_trans_x = loss_trans_x.mean(dim=2).mean(dim=1)
                # loss_trans_y = loss_trans_y.mean(dim=2).mean(dim=1)
                # loss_trans_z = loss_trans_z.mean(dim=2).mean(dim=1)

                # loss = loss_trans_z + loss_trans_y + loss_trans_x
                # energy_pos = loss[is_possible]
                # energy_neg = loss[~is_possible]

                # energy_pos = energy_pos.detach().cpu().numpy()
                # energy_neg = energy_neg.detach().cpu().numpy()

                # for i in range(energy_pos.shape[0]):
                #     frame = {}
                #     frame['is_possible'] = True
                #     frame['likelihood'] = float(energy_pos[i])
                #     evals.append(frame)

                # for i in range(energy_neg.shape[0]):
                #     frame = {}
                #     frame['is_possible'] = False
                #     frame['likelihood'] = float(energy_neg[i])
                #     evals.append(frame)

                # print("possible: ", energy_pos.mean())
                # print("not possible: ", energy_neg.mean())




                val_loss_logger.update(**loss_dict)
                # summary_writer.add_scalars("val_non_smooth", val_loss_logger.last_item, i)

                # all_preds.append({k: v.cpu().numpy() for k, v in output["output"].items()})
                # all_labels.append({k: v.cpu().numpy() for k, v in inputs["attributes"].items()})
                # all_labels = self.attributes.cat_by_key(all_labels, inputs["attributes"])
                # all_preds = self.attributes.cat_by_key(all_preds, output['output'])

                # batch_time = time.time() - last_batch_time
                # val_metrics.update(time=batch_time, data=data_time)
                # if time.time() - start > self.cfg.SOLVER.VALIDATION_MAX_SECS:
                #     raise Val_Too_Long

            # all_preds, all_labels = map(lambda l: {k: np.concatenate([a[k] for a in l]) for k in l[0].keys()},
            #                             [all_preds, all_labels])
            # all_preds = {k: np.concatenate([a[k] for a in all_preds]) for k in all_preds[0].keys()}
            # all_labels = {k: np.concatenate([a[k] for a in all_labels]) for k in all_labels[0].keys()}
            # err_dict = self.attributes.pred_error(all_preds, all_labels)
            # val_metrics.update(**err_dict)
            # log.info(val_metrics.delimiter.join(["VALIDATION", "iter: {iter}", "{meters}"])
            #          .format(iter=iteration, meters=str(val_metrics)))
            log.info(val_metrics.delimiter.join(["VALIDATION", "iter: {iter}", "{meters}"])
                     .format(iter=iteration, meters=str(val_loss_logger)))
            if summary_writer is not None:
                # summary_writer.add_scalars("val_error", val_metrics.mean, iteration)
                summary_writer.add_scalars("val", val_loss_logger.mean, iteration)
            # self.derenderer.train()
        json.dump(evals, open("output.json", "w"))
        self.model.train()
        return None
