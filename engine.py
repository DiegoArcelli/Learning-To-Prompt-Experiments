# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    # iter = 0
    # curr_prompt, prev_prompt = None, None
    curr_class = -1
    # model.prompt.prompt.requires_grad = False
    # model.prompt.prompt_key.requires_grad = False
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        if args.class_incremental:
            max_batch_class = max(target.tolist())
            if max_batch_class not in model.prompt.seen_classes:
                model.prompt.seen_classes.append(max_batch_class)
                print(model.prompt.seen_classes)

            if max_batch_class != curr_class:
                for c in range(model.num_classes):
                    # model.prompt.prompt[c].requires_grad = False
                    # model.prompt.prompt_key[c].requires_grad = False
                    if c == max_batch_class:
                        model.prompt.prompt[c].requires_grad = True
                        model.prompt.prompt_key[c].requires_grad = True
                    else:
                        model.prompt.prompt[c].requires_grad = False
                        model.prompt.prompt_key[c].requires_grad = False
                    # print(c, model.prompt.prompt[c].requires_grad, model.prompt.prompt_key[c].requires_grad)
                curr_class = max_batch_class
                
        
        # curr_prompt = model.prompt.prompt_key[0].detach().clone()
        # # print(model.prompt.prompt_key[0].requires_grad)
        # if prev_prompt is None:
        #     prev_prompt = curr_prompt.detach().clone()
        # else:
        #     print((prev_prompt == curr_prompt).all())
        #     prev_prompt = curr_prompt.detach().clone()

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        

        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)

        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        batch_size = logits.shape[0]
        top_k = len(output["prompt_idx"][0])
        for i in range(batch_size):
            label = target[i].item() # get the true class of the i-th element of the batch
            # iterate over each prompt selected prompt for the i-th element of the batch
            for j in range(top_k): 
                key_id = output["prompt_idx"][i][j].item() # get the j-th selected prompt of the i-th batch element
                # increase the key-class counter
                task = label // 10
                model.key_class_counts[key_id][label] += 1
                model.key_task_counts[key_id][task] += 1

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # c = 23
        # curr_prompt = model.prompt.prompt_key[c].detach().clone()
        # print(model.prompt.prompt_key[c].grad)
        # print(model.prompt.prompt[c].grad)
        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # curr_prompt_3 = model.prompt.prompt_key[c].detach().clone()
        # print((curr_prompt == curr_prompt_3).all())
        # print("")


        # torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        # iter += 1
        # if iter == 20:
        #     break

    print(model.key_task_counts)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None, last_task=None, use_task_id=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    iter = 0
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features, train=use_task_id)
            logits = output['logits']

            batch_size = logits.shape[0]
            top_k = len(output["prompt_idx"][0])
            for i in range(batch_size):
                label = target[i].item() # get the true class of the i-th element of the batch
                # iterate over each prompt selected prompt for the i-th element of the batch
                for j in range(top_k): 
                    key_id = output["prompt_idx"][i][j].item() # get the j-th selected prompt of the i-th batch element
                    # increase the key-class counter
                    task = label // 10
                    model.class_key_counts_test[last_task][label][key_id] += 1
                    model.task_key_counts_test[last_task][task][key_id] += 1

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            # iter += 1
            # if iter == 20:
            #     break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None, use_task_id=False):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args, last_task=task_id, use_task_id=use_task_id)
        
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    print(model.task_key_counts_test[task_id])
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):

       # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()


        if args.task_incremental:

            for t in range(args.num_tasks):
                if t == task_id:
                    model.prompt.prompt[t].requires_grad = True
                    model.prompt.prompt_key[t].requires_grad = True
                else:
                    model.prompt.prompt[t].requires_grad = False
                    model.prompt.prompt_key[t].requires_grad = False

            for t in range(args.num_tasks):    
                print(t, model.prompt.prompt[t].requires_grad,  model.prompt.prompt_key[t].requires_grad)

            if task_id > 0:

                model.prompt.seen_tasks += 1
                
                # prompt_per_task = model.prompt.prompt_per_task
                # prev_start = (task_id - 1) * prompt_per_task
                # prev_end = task_id * prompt_per_task

                # cur_start = prev_end
                # cur_end = (task_id + 1) * prompt_per_task

                # cur_idx = (slice(cur_start, cur_end))
                # prev_idx = (slice(prev_start, prev_end))

                # print("Transfering parameters ", (cur_idx, prev_idx))

                # with torch.no_grad():
                #     for t in range(len(model.prompt.prompt)):
                #         model.prompt.prompt[t].grad_zero_()
                #     model.prompt.prompt[model.prompt.seen_tasks][0:] = model.prompt.prompt[model.prompt.seen_tasks - 1][0:]
                #     # model.prompt.prompt.grad.zero_()
                #     # model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                #     optimizer.param_groups[0]['params'] = model.parameters()
        
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)
        
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, use_task_id=args.eval_task_id, args=args)
        
        if args.frequency_penalization:
            model.prompt.key_counts = model.prompt.key_counts_temp.detach().clone()

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)
            

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')