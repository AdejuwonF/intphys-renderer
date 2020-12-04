import datetime
import itertools
import os
import time
from collections import defaultdict
from copy import deepcopy
from itertools import cycle
from multiprocessing import Pool, cpu_count
from warnings import warn

import numpy as np
import psutil
from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset
from easydict import EasyDict

# from structure.attributes.shapes_world_attributes import CONTINUOUS_TERMS, CATEGORICAL_TERMS
from structure.derender_attributes import DerenderAttributes

proc_id = psutil.Process(os.getpid())

from utils.build import make_optimizer, make_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.metric_logger import MetricLogger
from utils.misc import to_cuda, gather_loss_dict, image_based_to_annotation_based, \
    read_image  # , image_based_to_annotation_based

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from models.derender import Derender
# from configs.derender_config_old import _C as cfg
import logging as log
# log = log.getLogger("experiment")
# See log level hierarchy: https://docs.python.org/3/library/logging.html#levels
# from structure.attributes import build_attributes


class Nan_Exception(Exception):
    pass

class Val_Too_Long(Exception):
    pass

class DerenderPredictor():
    def __init__(self,cfg):
        self.cfg = cfg.clone()
        self.attributes = DerenderAttributes(cfg)
        self.derenderer = Derender(self.cfg, self.attributes)
        checkpointer = Checkpointer(self.derenderer, logger=log)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.derenderer = self.derenderer.to("cuda").eval()

    def __call__(self, input_batch):
        outputs = self.derenderer(input_batch)
        outputs = {k: v.cpu().numpy() for k, v in outputs["output"].items()}
        outputs = self.attributes.predict(outputs)
        return outputs

def create_new_timer():
    init_time = time.time()
    timers = EasyDict(dict(start=init_time,
                           batch=init_time,
                           log=init_time,
                           checkpoint=init_time,
                           validation=init_time,
                           epoch=init_time,
                           tensorboard=init_time))
    return timers

class DerenderMapper():
    def __init__(self, use_inferred_boxes, attributes ,for_inference, use_depth=True):
        self.box_key = "pred_box" if use_inferred_boxes else "bbox"
        self.for_inference = for_inference
        self.attributes = attributes
        self.use_depth = use_depth

    def __call__(self, el):

        box = el[self.box_key]

        if self.use_depth:
            depth = read_image(el["file_name"])
            norm_depth = 1.0 / (1.0 + depth)

            norm_depth_masked = np.zeros_like(norm_depth)
            norm_depth_masked[box[1]:box[3] + 1, box[0]:box[2] + 1] = \
                norm_depth[box[1]:box[3] + 1, box[0]:box[2] + 1]

            img_tuple = torch.FloatTensor(np.stack([norm_depth, norm_depth_masked], axis=0))
        else:
            img, img_4, img_2 = map(read_image,
                                    [el["file_name"]] + list(el["prev_images"]))
            segmented_image = np.zeros_like(img)
            segmented_image[:,box[1]:box[3] + 1, box[0]:box[2] + 1] = \
                img[:,box[1]:box[3] + 1, box[0]:box[2] + 1]
            img_tuple =  np.concatenate([segmented_image,
                                         img, img-img_2,
                                         img-img-4],
                                        axis=0)/255
            img_tuple = torch.FloatTensor(img_tuple)

        if hasattr(self.attributes, "camera_terms"):
            camera = {k: torch.FloatTensor([el["camera"][k]])
                  for k in self.attributes.camera_terms}
        else:
            camera = {}


        if not self.for_inference:
            attributes = {k:torch.FloatTensor([el["attributes"][k]])
                          for k in self.attributes.continuous_terms}
            attributes.update({k:int(el["attributes"][k])
                               for k in self.attributes.categorical_terms +
                                        self.attributes.quantized_terms})
            return {"img_tuple": img_tuple,
                    "camera": camera,
                    "attributes": attributes}

        return {"img_tuple": img_tuple,
                "camera": camera}

def derender_dataset(cfg, dataset_names, attributes,for_inference=False):
    print("reading datasets {}".format(dataset_names))
    start = time.time()
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    required_fields = ["pred_box"] if cfg.DATASETS.USE_PREDICTED_BOXES else ["bbox"]
    required_fields += ["attributes"] if not for_inference else []
    _, dataset_dicts = image_based_to_annotation_based(dataset_dicts, required_fields)

    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapper = DerenderMapper(cfg.DATASETS.USE_PREDICTED_BOXES,
                            attributes,
                            for_inference,
                            use_depth=cfg.DATASETS.USE_DEPTH)
    dataset = MapDataset(dataset, mapper)
    print("done after {}".format(time.time()-start))
    return dataset

class Trainable_Derender():
    def __init__(self, cfg):
        log.getLogger("derender")
        # log.basicConfig(level=log.INFO,file=cfg.LOG_FILE)
        self.cfg = cfg
        self.reset_data()
        self.reset()

    def reset_data(self):
        if not hasattr(self,'attributes'):
            attributes = DerenderAttributes(self.cfg)
            self.attributes = attributes

        dataset_load_time = time.time()
        val_dataset, train_dataset = map(lambda d: derender_dataset(self.cfg, d, attributes),
                                         [self.cfg.DATASETS.TEST,self.cfg.DATASETS.TRAIN])

        dataset_load_time = time.time() - dataset_load_time
        log.info(f'Dataset loaded in {dataset_load_time:.2f} seconds.')

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


    def reset(self, output_dir=None, new_params=[], log_flag=True, fixed_seed=True):
        if output_dir == None:
            output_dir = self.cfg.OUTPUT_DIR

        #merge new parameters
        self.cfg.merge_from_list(new_params)


        #save new  configuration
        with open(os.path.join(output_dir,"cfg.yaml"), 'w') as f:
            x = self.cfg.dump(indent=4)
            f.write(x)

        log.info(f'New training run with configuration:\n{self.cfg}\n\n')


        workers = self.cfg.DATALOADER.NUM_WORKERS if not self.cfg.DEBUG else 0
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.cfg.DATALOADER.OBJECTS_PER_BATCH,
                                  num_workers=workers, shuffle=True)
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.cfg.DATALOADER.VAL_BATCH_SIZE,
                                num_workers=self.cfg.DATALOADER.NUM_WORKERS, shuffle=True)

        # Instantiate training pipeline components
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if fixed_seed:
            torch.manual_seed(1234)
        derenderer = Derender(self.cfg, self.attributes).to(device)

        optimizer = make_optimizer(self.cfg, derenderer)
        scheduler = make_lr_scheduler(self.cfg, optimizer)
        checkpointer = Checkpointer(derenderer, optimizer, scheduler, output_dir, log)
        self.start_iteration = checkpointer.load()

        # Multi-GPU Support
        if device == 'cuda' and not self.cfg.DEBUG:
            gpu_ids = [_ for _ in range(torch.cuda.device_count())]
            derenderer = torch.nn.parallel.DataParallel(derenderer, gpu_ids)

        self.optimizer = optimizer
        self.derenderer = derenderer
        self.scheduler = scheduler
        self.device = device
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.output_dir  = output_dir
        self.checkpointer = checkpointer

    def eval(self, iteration=-1, summary_writer=None):
        start = time.time()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            # self.derenderer.eval()
            val_metrics = MetricLogger(delimiter="  ")
            val_loss_logger = MetricLogger(delimiter="  ")
            for i, inputs in enumerate(self.val_loader, iteration):
                # data_time = time.time() - last_batch_time

                if torch.cuda.is_available():
                    inputs = to_cuda(inputs)

                output = self.derenderer(inputs)

                loss_dict = gather_loss_dict(output)
                val_loss_logger.update(**loss_dict)
                summary_writer.add_scalars("val_non_smooth", val_loss_logger.last_item, i)

                all_preds.append({k:v.cpu().numpy() for k,v in output["output"].items()})
                all_labels.append({k:v.cpu().numpy() for k,v in inputs["attributes"].items()})
                # all_labels = self.attributes.cat_by_key(all_labels, inputs["attributes"])
                # all_preds = self.attributes.cat_by_key(all_preds, output['output'])


                # batch_time = time.time() - last_batch_time
                # val_metrics.update(time=batch_time, data=data_time)
                if time.time() - start > self.cfg.SOLVER.VALIDATION_MAX_SECS:
                    break

            all_preds, all_labels = map(lambda l: {k: np.concatenate([a[k] for a in l]) for k in l[0].keys()},
                                        [all_preds, all_labels])
            # all_preds = {k: np.concatenate([a[k] for a in all_preds]) for k in all_preds[0].keys()}
            # all_labels = {k: np.concatenate([a[k] for a in all_labels]) for k in all_labels[0].keys()}
            err_dict = self.attributes.pred_error(all_preds, all_labels)
            val_metrics.update(**err_dict)
            log.info(val_metrics.delimiter.join(["VALIDATION", "iter: {iter}", "{meters}"])
                     .format(iter=iteration, meters=str(val_metrics)))
            log.info(val_metrics.delimiter.join(["VALIDATION", "iter: {iter}", "{meters}"])
                     .format(iter=iteration, meters=str(val_loss_logger)))
            if summary_writer is not None:
                summary_writer.add_scalars("val_error", val_metrics.mean, iteration)
                summary_writer.add_scalars("val", val_loss_logger.mean, iteration)
            # self.derenderer.train()
        return err_dict



    def train(self, log_flag=True):
        train_metrics = MetricLogger(delimiter="  ")
        summary_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "summary"))

        self.derenderer.train()

        # Initialize timing
        timers  =  create_new_timer()

        done = False
        while not done:
            for iteration, inputs in enumerate(self.train_loader, self.start_iteration):
                iter_time = time.time()
                data_time = iter_time - timers.batch

                if torch.cuda.is_available():
                    inputs = to_cuda(inputs)

                output = self.derenderer(inputs)

                loss_dict = gather_loss_dict(output)
                loss = loss_dict['loss']
                # loss = sum([loss_dict[term] for term in ['x', 'y', 'z']])

                if torch.isnan(loss).any():
                    raise Nan_Exception()

                train_metrics.update(**loss_dict)
                summary_writer.add_scalars("train_non_smooth", train_metrics.last_item, iteration)


                batch_time = iter_time - timers.batch
                timers.batch = iter_time
                train_metrics.update(time=batch_time, data=data_time)
                eta_seconds = timers.start + self.cfg.SOLVER.MAX_TIME_SECS - iter_time
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if (iter_time - timers.log > self.cfg.SOLVER.PRINT_METRICS_TIME and log_flag):
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

                if iter_time - timers.checkpoint > self.cfg.SOLVER.CHECKPOINT_SECS: #iteration % checkpoint_period == 0:
                    timers.checkpoint = iter_time
                    self.checkpointer.save("model_{:07d}".format(iteration))

                if iter_time - timers.tensorboard > self.cfg.SOLVER.TENSORBOARD_SECS or self.cfg.DEBUG:
                    timers.tensorboard = iter_time
                    summary_writer.add_scalars("train", train_metrics.mean, iteration)


                if iter_time - timers.start > self.cfg.SOLVER.MAX_TIME_SECS:
                    log.info("finished training loop in {}".format(iter_time-timers.start))
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

        err_dict = self.eval(iteration, summary_writer)

        self.checkpointer.save("model_{:07d}".format(iteration))
        summary_writer.close()
        return err_dict


if __name__ ==  "__main__":
    trainable = Trainable_Derender(cfg)
    trainable.reset()
    trainable.train()
