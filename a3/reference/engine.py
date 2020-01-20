import math
import sys
import time
import torch
import logging

import torchvision.models.detection.mask_rcnn
from pathlib import Path

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils
from ..eval import get_tb_logger, EarlyStopping


logger = logging.getLogger(__name__)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, logdir):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    tb_logger = get_tb_logger(logdir=logdir)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    i = (epoch + 1) * len(data_loader)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        float_metrics = {k: v.value for k, v in metric_logger.meters.items()}
        tb_logger.add_scalars('Loss/train', float_metrics, i)
        i += 1


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, logdir, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")    
    header = 'Test:'
    tb_logger = get_tb_logger(logdir=logdir)

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()    
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats
    ap_stat_names = [
        'AP@0.50:0.95@all',
        'AP@0.50@all',
        'AP@0.75@all',
        'AP@0.50:0.95@small',
        'AP@0.50:0.95@medium',
        'AP@0.50:0.95@large',
    ]
    ar_stat_names = [
        'AR@0.50:0.95@all',
        'AR@0.50:0.95@all',
        'AR@0.50:0.95@all',
        'AR@0.50:0.95@small',
        'AR@0.50:0.95@medium',
        'AR@0.50:0.95@large',
    ]
    ap_eval_stats = dict(zip(ap_stat_names, stats[:5]))
    ar_eval_stats = dict(zip(ar_stat_names, stats[5:]))
    tb_logger.add_scalars('AP/test', ap_eval_stats, epoch)
    tb_logger.add_scalars('AR/test', ar_eval_stats, epoch)

    torch.set_num_threads(n_threads)
    return coco_evaluator


def get_model(pretrained, num_classes=11, pretrained_backbone=True):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, 
                                                                num_classes=num_classes, 
                                                                pretrained_backbone=pretrained_backbone)

def get_optimizer(params):
    return torch.optim.SGD(params, 
                            lr=0.005,
                            momentum=0.9, 
                            weight_decay=0.0005)

def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                    step_size=5,
                                    gamma=0.1)


class Experiment:

    def __init__(self, name, model_factory, optimizer_factory, lr_scheduler_factory):
        self.exp_name = name
        self.checkpoint_dir = Path(f'./models/{name}/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_fn = f'./models/{name}/best-ap-model.pt'
        self.logdir = f'./logdir/{name}'        
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory


    def train(
        self,
        train_data_loader,
        val_data_loader,        
        num_epoch=100,
        resume=True,
        device=torch.device('cuda:0')        
    ):
        model = self.model_factory()
        torch.cuda.empty_cache()
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.optimizer_factory(params)
        lr_scheduler = self.lr_scheduler_factory(optimizer)
        es = EarlyStopping(mode='max', patience=5)

        checkpoints = sorted(list(Path(self.checkpoint_dir).iterdir()))
        epoch = 0
        if checkpoints and resume:
            logger.info("Loading from checkpoint")    
            epoch, model, optimizer, lr_scheduler, _ = utils.load_checkpoint(checkpoints[-1], 
                                                                    model=model, 
                                                                    optimizer=optimizer, 
                                                                    lr_scheduler=lr_scheduler)
            assert lr_scheduler is not None
            logger.info(f"Checkpoint loaded, resuming training from {epoch + 1} epoch")

        try:
            for epoch in range(epoch, num_epoch):
                train_one_epoch(model, optimizer, train_data_loader, device, epoch, 20, self.logdir)
                lr_scheduler.step()
                # evaluate after every epoch
                coco_evaluator = evaluate(model, val_data_loader, device=device, logdir=self.logdir, epoch=epoch)

                stats = coco_evaluator.coco_eval['bbox'].stats
                ap = stats[0]
                prev_best = 0 if es.best is None else es.best
                utils.save_checkpoint(
                    state=dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        lr_scheduler=lr_scheduler.state_dict(),
                        stats=stats,
                        epoch=epoch
                    ), 
                    is_best=ap > prev_best,
                    filename=f"{self.checkpoint_dir}/{epoch}.pt", 
                    best_filename=self.best_model_fn
                )
                if es.step(ap):
                    logger.info(f"Metric did not improve over {es.patience} epoch, stopping")
                    break
        except Exception as e:
            raise
        finally:
            del model, optimizer, params, lr_scheduler, es 
            del train_data_loader, val_data_loader
            torch.cuda.empty_cache()


    def evaluate(self):
        pass
