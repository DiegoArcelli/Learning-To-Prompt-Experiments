# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):

    # if args.freeze_keys:
    #     args.pull_constraint = False
    #     args.pull_constraint_coeff = 0.0

    # args.size = 10
    # args.top_k = 10
    # args.prompt_key=False
    # args.embedding_key='mean'
    # args.pull_constraint=False
    # args.pull_constraint_coeff=0.0
    # args.top_k = 2
    # if args.train_type == "task_wise":
    #     args.train_mask = True
    #     args.use_prompt_mask=True
    #     args.batchwise_prompt=True

    args.batchwise_prompt = False
    args.use_prompt_mask=True
    
    print(args)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        class_inc=args.class_incremental,
        task_inc=args.task_incremental,
        frequency_penalization=args.frequency_penalization
    )
    original_model.to(device)
    model.to(device) 
    # torch.save(original_model.state_dict(), "original_pytorch.pt")
    # torch.save(model.state_dict(), "l2p.pt")

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():

            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

            if args.freeze_head and n.startswith("head"):
                p.requires_grad = False
    # if args.freeze_keys:
    #     model.prompt.prompt_key.requires_grad = False



    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    
    # if args.train_type != "task_wise":
    #     args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()


    if args.init_tasks_prompts:

        if config == 'cifar100_l2p':
            dataset = "cifar100"
        else:
            dataset = "5-datasets"

        prompt_per_task = args.prompts_per_task

        if args.task_incremental:
            for t in range(args.num_tasks):
                task_prompt = torch.load(f"./init_prompts/{dataset}/task/prompt_{t}.pt")
                task_prompt = task_prompt.repeat(prompt_per_task, 1)
                model.prompt.prompt_key[t] = task_prompt
                # model.prompt.prompt_key[t] = task_prompt
        else:
            model.prompt.prompt_key.requires_grad = False
            for t in range(args.num_tasks):
                task_prompt = torch.load(f"./init_prompts/{dataset}/task/prompt_{t}.pt")
                task_prompt = task_prompt.repeat(prompt_per_task, 1)
                model.prompt.prompt_key[t*prompt_per_task:(t+1)*prompt_per_task] = task_prompt
                # model.prompt.prompt_key[t] = task_prompt
            model.prompt.prompt_key.requires_grad = True

    if args.init_class_prompts:

        if config == 'cifar100_l2p':
            dataset = "cifar100"
            num_classes = 100
        else:
            dataset = "5-datasets"
            num_classes = 50

        prompt_per_class = args.prompts_per_class

        if args.class_incremental:
            for c in range(num_classes):
                class_prompt = torch.load(f"./init_prompts/{dataset}/class/prompt_{c}.pt")
                class_prompt = class_prompt.repeat(prompt_per_class, 1)
                model.prompt.prompt_key[c] = class_prompt
        else:
            model.prompt.prompt_key.requires_grad = False
            for t in range(num_classes):
                class_prompt = torch.load(f"./init_prompts/{dataset}/class/prompt_{t}.pt")
                class_prompt = class_prompt.repeat(prompt_per_class, 1)
                model.prompt.prompt_key[t*prompt_per_class:(t+1)*prompt_per_class] = class_prompt
            model.prompt.prompt_key.requires_grad = True


    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)
