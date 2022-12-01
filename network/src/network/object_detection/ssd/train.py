import argparse
import logging
import os
from pathlib import Path

import torch
# import torch.distributed as dist
from network.object_detection.ssd.engine.inference import do_evaluation
from network.object_detection.ssd.config import cfg
from network.object_detection.ssd.data.build import make_data_loader, make_data_loader_denso
from network.object_detection.ssd.engine.trainer import do_train
from network.object_detection.ssd.modeling.detector import build_detection_model
from network.object_detection.ssd.solver.build import make_optimizer, make_lr_scheduler
from network.object_detection.ssd.utils import dist_util, mkdir
from network.object_detection.ssd.utils.checkpoint import CheckPointer
from network.object_detection.ssd.utils.dist_util import synchronize
from network.object_detection.ssd.utils.logger import setup_logger
from network.object_detection.ssd.utils.misc import str2bool


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, args.checkpoints, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    # train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])
    train_loader = make_data_loader_denso(cfg, args, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments, args)
    return model


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--dataset_dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=-1, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.checkpoints:
        mkdir(args.checkpoints)

    logger = setup_logger("SSD", dist_util.get_rank(), args.checkpoints)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
