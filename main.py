#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
import logging
import argparse
import numpy as np
import random
import copy
import pandas as pd

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader
from utils import (str2bool, load_model, save_checkpoint, create_optimizer,
                   AverageMeter, mixup, CrossEntropyLoss, onehot)
from argparser import get_config
import mixup_utils.apx_losses as apx
import mixup_utils.taylor_losses as taylor

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str)
    parser.add_argument('--config', type=str)

    # model config (VGG)
    parser.add_argument('--n_channels', type=str)
    parser.add_argument('--n_layers', type=str)
    parser.add_argument('--use_bn', type=str2bool)
    #
    parser.add_argument('--base_channels', type=int)
    parser.add_argument('--block_type', type=str)
    parser.add_argument('--depth', type=int)
    # model config (ResNet-preact)
    parser.add_argument('--remove_first_relu', type=str2bool)
    parser.add_argument('--add_last_bn', type=str2bool)
    parser.add_argument('--preact_stage', type=str)
    # model config (WRN)
    parser.add_argument('--widening_factor', type=int)
    # model config (DenseNet)
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--compression_rate', type=float)
    # model config (WRN, DenseNet)
    parser.add_argument('--drop_rate', type=float)
    # model config (PyramidNet)
    parser.add_argument('--pyramid_alpha', type=int)
    # model config (ResNeXt)
    parser.add_argument('--cardinality', type=int)
    # model config (shake-shake)
    parser.add_argument('--shake_forward', type=str2bool)
    parser.add_argument('--shake_backward', type=str2bool)
    parser.add_argument('--shake_image', type=str2bool)
    # model config (SENet)
    parser.add_argument('--se_reduction', type=int)

    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test_first', type=str2bool, default=True)
    parser.add_argument('--gpu', type=str, default='0')

    # TensorBoard configuration
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_train_images', action='store_true')
    parser.add_argument('--tensorboard_test_images', action='store_true')
    parser.add_argument('--tensorboard_model_params', action='store_true')

    # configuration of optimizer
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    # configuration for SGD
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--nesterov', type=str2bool)
    # configuration for learning rate scheduler
    parser.add_argument(
        '--scheduler', type=str, choices=['none', 'multistep', 'cosine'])
    # configuration for multi-step scheduler]
    parser.add_argument('--milestones', type=str)
    parser.add_argument('--lr_decay', type=float)
    # configuration for cosine-annealing scheduler]
    parser.add_argument('--lr_min', type=float, default=0)
    # configuration for Adam
    parser.add_argument('--betas', type=str)

    # configuration of data loader
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'MiniMNIST'])
    parser.add_argument('--num_per_class', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=7)
    # cutout configuration
    parser.add_argument('--use_cutout', action='store_true', default=False)
    parser.add_argument('--cutout_size', type=int, default=16)
    parser.add_argument('--cutout_prob', type=float, default=1)
    parser.add_argument('--cutout_inside', action='store_true', default=False)
    # random erasing configuration
    parser.add_argument(
        '--use_random_erasing', action='store_true', default=False)
    parser.add_argument('--random_erasing_prob', type=float, default=0.5)
    parser.add_argument(
        '--random_erasing_area_ratio_range', type=str, default='[0.02, 0.4]')
    parser.add_argument(
        '--random_erasing_min_aspect_ratio', type=float, default=0.3)
    parser.add_argument('--random_erasing_max_attempt', type=int, default=20)
    # mixup configuration
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--fixlam', type=float, default=-1) # lambda used in computing apx losses
    parser.add_argument('--fixtrainlam', type=float, default=-1) # lambda used in training
    parser.add_argument('--mixup_alpha', type=float, default=1)
    parser.add_argument('--doublesum_batches', type=int, default=20) # how many batches should I use when computing double sum loss?
    parser.add_argument('--compute_mixup_reg', type=int, default=0) # 1 to compute mixup regularization (normal), 0 to skip
    parser.add_argument('--cov_components', type=int, default=-1) # number of components to take when computing approximate covariance

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    config = get_config(args)

    return config


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config,
          writer, moment_dict):
    global global_step

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    loss_before_meter = AverageMeter() # re-evaluate error on images before the gradient update, first 20 batches
    loss_after_meter = AverageMeter() # re-evaluate error on images after the gradient update, first 20 batches
    accuracy_meter = AverageMeter()

    # approximate losses and average meters are assembled here
    apx_meters = {
        'vanilla': AverageMeter(),
        'mixup': AverageMeter(),
        'doublesum': AverageMeter()
    }

    apx_callbacks = {
        'vanilla': lambda imgs, lbls, mdl: apx.vanilla_loss(imgs, lbls, mdl, run_config['use_gpu']),
        'mixup': lambda imgs, lbls, mdl: apx.mixup_loss(imgs, lbls, data_config['mixup_alpha'], data_config['n_classes'], data_config['fixlam'], mdl, run_config['use_gpu']),
        'doublesum': lambda imgs, lbls, mdl: apx.doublesum_loss(imgs, lbls, data_config['mixup_alpha'], data_config['n_classes'], data_config['fixlam'], mdl, run_config['use_gpu'])
    }

    ### take 2 for computing apx losses ###
    # we're going to store all the images that we saw throughout this epoch
    # and compute our loss on these at the end - this is effectively what is being optimised

    # then recompute a mixed up dataset and compute the loss on that - this is our (likely close) approximation
    # for double sum loss

    # actually, for validity: do this batchwise too - makes your life easier too since you don't need to load up all images and labels
    images_train = [] #: all images that were encountered in this epoch
    labels_train = [] #: all labels that were encountered in this epoch
    images_eval = [] # images that we're lining up for eval at the end of the epoch
    labels_eval = []
    images_eval2 = [] # second trial to check concentration
    labels_eval2 = []

    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        images = copy.deepcopy(data)
        labels = copy.deepcopy(targets)

        if data_config['use_mixup']:
            data, targets = mixup(data, targets, data_config['mixup_alpha'],
                                  data_config['n_classes'], data_config['fixtrainlam'], True)

            # assembling the data for our doublesum apx test here
            images_train.append(copy.deepcopy(data))
            labels_train.append(copy.deepcopy(targets))

        if run_config['tensorboard_train_images']:
            if step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Train/Image', image, epoch)

        if optim_config['scheduler'] == 'multistep':
            scheduler.step(epoch - 1)
        elif optim_config['scheduler'] == 'cosine':
            scheduler.step()

        if run_config['tensorboard']:
            if optim_config['scheduler'] != 'none':
                lr = scheduler.get_lr()[0]
            else:
                lr = optim_config['base_lr']
            writer.add_scalar('Train/LearningRate', lr, global_step)

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()

        # compute loss after the gradient update
        outputs = model(data)
        newloss = criterion(outputs, targets)
        newloss_ = newloss.item()

        if data_config['use_mixup']:
            _, targets = targets.max(dim=1)
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        # this is where the approximate losses are computed
        # loss_before_meter.update(loss_, num) # now we're not restricting batches
        # loss_after_meter.update(loss_, num)
        #if step < data_config['doublesum_batches']:
            #for k in apx_meters.keys():
                #l = apx_callbacks[k](images, labels, model)
                #apx_meters[k].update(l.item(), num)
            #loss_before_meter.update(loss_, num)
            #loss_after_meter.update(newloss_, num)

        if data_config['compute_mixup_reg'] > 0:
            # batch sizee
            N = data_config['batch_size']

            # original shape of images
            data_shape = data.shape
            # datavar = torch.autograd.Variable(data), requires_grad=True)
            # hello = model(data)
            # data_flat is a stack of rows, where each row
            # is a flattened data point:
            # --- data_flat[i,:] = data[i,:,...,:].reshape((1, int(data.numel() / N)))
            data_flat = data.reshape((N, int(data.numel() / N)))

            # y_vec is a stack of rows, where each row is the one_hot version
            # of the correct label
            y_vec = torch.zeros((N, targets.max() + 1)).cuda()
            y_vec[np.arange(N), targets] = 1

            # vec to take action of hessian on
            V = (data_flat - data_flat.sum(axis=0)).detach().clone()
            W = (y_vec - y_vec.sum(axis=0)).detach().clone()
            X = 2*torch.ones((2, 2)).cuda()
            Y = 2*torch.ones((2, 2)).cuda()
            V = torch.ones((2,2)).cuda()
            hvprod = taylor.hess_quadratic(lambda x, y : torch.sum(x.pow(2) + y.pow(2)), lambda x: x, X.shape, X, Y, 'x', 'x', V, V)
            print(hvprod)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))

    ret = [epoch, loss_meter.avg, accuracy_meter.avg]

    if data_config['use_mixup'] and (epoch <= 4 or epoch % 5 == 0):
        # reiterating through trainloader to completely separate the construction of the eval sets from the train set
        for step, (data, targets) in enumerate(train_loader):
            old_data = copy.deepcopy(data)
            old_targets = copy.deepcopy(targets)
            data_eval, targets_eval = mixup(old_data, old_targets, data_config['mixup_alpha'],
                                    data_config['n_classes'], data_config['fixlam'], True)

            images_eval.append(copy.deepcopy(data_eval))
            labels_eval.append(copy.deepcopy(targets_eval))

        for step, (data, targets) in enumerate(train_loader):
            old_data = copy.deepcopy(data)
            old_targets = copy.deepcopy(targets)

            data_eval2, targets_eval2 = mixup(old_data, old_targets, data_config['mixup_alpha'],
                                    data_config['n_classes'], data_config['fixlam'], True)

            images_eval2.append(copy.deepcopy(data_eval2))
            labels_eval2.append(copy.deepcopy(targets_eval2))

        # evaluating approximate losses
        images_train = torch.cat(images_train)
        labels_train = torch.cat(labels_train)
        images_eval = torch.cat(images_eval)
        labels_eval = torch.cat(labels_eval)
        images_eval2 = torch.cat(images_eval2)
        labels_eval2 = torch.cat(labels_eval2)

        apxloss_train = apx.compute_loss(images_train, labels_train, model, run_config['use_gpu'])
        apxloss_eval = apx.compute_loss(images_eval, labels_eval, model, run_config['use_gpu'])
        apxloss_eval2 = apx.compute_loss(images_eval2, labels_eval2, model, run_config['use_gpu'])

        logger.info('Train {:.4f}, Eval {:.4f}, Eval retrial {:.4f}'.format(
            apxloss_train,
            apxloss_eval,
            apxloss_eval2
        ))

        ret.append(apxloss_train.item())
        ret.append(apxloss_eval.item())
        ret.append(apxloss_eval2.item())

    # compute Taylor approximate loss
    if data_config['cov_components'] > 0 and (epoch <= 4 or epoch % 5 == 0):
        base_meter = AverageMeter()
        de_meter = AverageMeter()

        d2_meters = {}
        d2e_meters = {}

        num_components_list = [1, 2, 5, 20, 50, 200]

        for k in num_components_list:
            d2_meters[k] = AverageMeter()
            d2e_meters[k] = AverageMeter()

        d2_batch_counts = {}

        for k in num_components_list:
            d2_batch_counts[k] = 10
        
        d2e_batch_counts = {
            1: 10,
            2: 10,
            5: 4,
            20: 2,
            50: 2,
            200: 1
        }

        max_batch_count = 10

        for step, (data, targets) in enumerate(train_loader):
            if step == max_batch_count:
                break

            num = data.shape[0]

            # base term
            base = taylor.taylor_loss_base(
                data.cuda(), targets.cuda(), model,
                moment_dict['xbar'],
                moment_dict['ybar'],
                moment_dict['Uxx'],
                moment_dict['Sxx'],
                moment_dict['Vxx'],
                moment_dict['Uxy'],
                moment_dict['Sxy'],
                moment_dict['Vxy'],
                moment_dict['T_U'],
                moment_dict['T_S'],
                moment_dict['T_V'],
            )

            base_meter.update(base, num)

            # de term
            de = taylor.taylor_loss_de(
                data.cuda(), targets.cuda(), model,
                moment_dict['xbar'],
                moment_dict['ybar'],
                moment_dict['Uxx'],
                moment_dict['Sxx'],
                moment_dict['Vxx'],
                moment_dict['Uxy'],
                moment_dict['Sxy'],
                moment_dict['Vxy'],
                moment_dict['T_U'],
                moment_dict['T_S'],
                moment_dict['T_V'],
            )

            de_meter.update(de, num)

            # d2 term
            d2_dict = taylor.taylor_loss_d2(
                data.cuda(), targets.cuda(), model,
                moment_dict['xbar'],
                moment_dict['ybar'],
                moment_dict['Uxx'],
                moment_dict['Sxx'],
                moment_dict['Vxx'],
                moment_dict['Uxy'],
                moment_dict['Sxy'],
                moment_dict['Vxy'],
                moment_dict['T_U'],
                moment_dict['T_S'],
                moment_dict['T_V'],
            )

            for k in num_components_list:
                d2_meters[k].update(d2_dict[k], num)

            # d2e term
            kmax = max([k for k in num_components_list if d2e_batch_counts[k] > step])
            d2e_dict = taylor.taylor_loss_d2e(
                data.cuda(), targets.cuda(), model,
                moment_dict['xbar'],
                moment_dict['ybar'],
                moment_dict['Uxx'],
                moment_dict['Sxx'],
                moment_dict['Vxx'],
                moment_dict['Uxy'],
                moment_dict['Sxy'],
                moment_dict['Vxy'],
                moment_dict['T_U'][:, :, :kmax],
                moment_dict['T_S'][:, :kmax],
                moment_dict['T_V'][:, :, :kmax],
            )

            for k in num_components_list:
                if k <= kmax:
                    d2e_meters[k].update(d2e_dict[k], num)

            logger.info("Done batch", step)
        

        logger.info("CHECKS")
        print("Base", base_meter.count, base_meter.avg)
        print("DE", de_meter.count, de_meter.avg)
        for k in num_components_list:
            print("d2", k, d2_meters[k].count, d2_meters[k].avg)
        for k in num_components_list:
            print("d2e", k, d2e_meters[k].count, d2e_meters[k].avg)

        ret.append(base_meter.avg)
        ret.append(de_meter.avg)
        for k in num_components_list:
            ret.append(d2_meters[k].avg)
        for k in num_components_list:
            ret.append(d2e_meters[k].avg)

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    #logger.info('Vanilla {:.2f}, Mixup {:.2f}, Double sum {:.2f}, Train before {:.2f}, Train after {:.2f}'.format(
    #    apx_meters['vanilla'].avg,
    #    apx_meters['mixup'].avg,
    #    apx_meters['doublesum'].avg,
    #    loss_before_meter.avg,
    #    loss_after_meter.avg
    #))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

    if epoch <= 4 or epoch % 5 == 0:
        return ret
    else:
        return []

def test(epoch, model, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(test_loader):
        if run_config['tensorboard_test_images']:
            if epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Test/Image', image, epoch)

        if run_config['use_gpu']:
            data = data.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            outputs = model(data)

        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if run_config['tensorboard_model_params']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return loss_meter.avg, accuracy


def update_state(state, epoch, accuracy, model, optimizer):
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state['accuracy'] = accuracy

    # update best accuracy
    if accuracy > state['best_accuracy']:
        state['best_accuracy'] = accuracy
        state['best_epoch'] = epoch

    return state


def main():
    # parse command line argument and generate config dictionary
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    # TensorBoard SummaryWriter
    if run_config['tensorboard']:
        writer = SummaryWriter()
    else:
        writer = None

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = run_config['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save config as json file in output directory
    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # load data loaders
    train_loader, test_loader = get_loader(config['data_config'])

    # compute covariances
    full_train_loader, _ = get_loader(config['data_config'], return_full=True)
    for batch in full_train_loader:
        full_images, full_labels = batch

    num_classes = config['data_config']['n_classes']
    full_targets = onehot(full_labels, num_classes)
    xbar, ybar, xxcov, xycov, T = taylor.compute_moments(full_images, full_targets)
    #torch.save(xbar, 'xbar.pt')
    #torch.save(ybar, 'ybar.pt')
    #torch.save(xxcov, 'xxcov.pt')
    #torch.save(xycov, 'xycov.pt')
    num_components = config['data_config']['cov_components']
    Uxx, Sxx, Vxx = taylor.decomposition(xxcov, num_components)
    Uxy, Sxy, Vxy = taylor.decomposition(xycov, 10)

    xdim = T.shape[1]
    # svd's of T[i,:,:] slices
    T_U = torch.zeros((num_classes, xdim, num_components))
    T_S = torch.zeros((num_classes, num_components))
    T_V = torch.zeros((num_classes, xdim, num_components))

    for i in range(num_classes):
        T_U[i,:,:], T_S[i,:], T_V[i,:,:] = taylor.decomposition(T[i,:,:], num_components)

    #torch.save(Uxx, 'Uxx.pt')
    #torch.save(Sxx, 'Sxx.pt')
    #torch.save(Vxx, 'Vxx.pt')
    #torch.save(Uxy, 'Uxy.pt')
    #torch.save(Sxy, 'Sxy.pt')
    #torch.save(Vxy, 'Vxy.pt')
    if run_config['use_gpu']:
        moment_dict = {
            'Uxx': Uxx.cuda(),
            'Uxy': Uxy.cuda(),
            'Sxx': Sxx.cuda(),
            'Sxy': Sxy.cuda(),
            'Vxx': Vxx.cuda(),
            'Vxy': Vxy.cuda(),
            'xbar': xbar.reshape(full_images.shape[1:]).cuda(),
            'ybar': ybar.cuda(),
            'T_U': T_U.cuda(),
            'T_S': T_S.cuda(),
            'T_V': T_V.cuda()
        }
    else:
        moment_dict = {
            'Uxx': Uxx,
            'Uxy': Uxy,
            'Sxx': Sxx,
            'Sxy': Sxy,
            'Vxx': Vxx,
            'Vxy': Vxy,
            'xbar': xbar.reshape(full_images.shape[1:]),
            'ybar': ybar,
            'T_U': T_U,
            'T_S': T_S,
            'T_V': T_V
        }

    # set up dataframe for recording results:
    dfcols = []
    dfcols.append('epoch')
    dfcols.append('train_loss')
    dfcols.append('train_acc')
    if config['data_config']['use_mixup']:
        dfcols.append('doublesum_train')
        dfcols.append('doublesum_eval')
        dfcols.append('doublesum_eval2')
    if config['data_config']['cov_components'] > 0:
        dfcols.append('taylor_base')
        dfcols.append('taylor_de')
        for k in [1, 2, 5, 20, 50, 200]:
            dfcols.append('taylor_d2_' + str(k))
        for k in [1, 2, 5, 20, 50, 200]:
            dfcols.append('taylor_d2e_' + str(k))
    dfcols.append('test_loss')
    dfcols.append('test_acc')

    resultsdf = pd.DataFrame(columns=dfcols)

    # load model
    logger.info('Loading model...')
    model = load_model(config['model_config'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))
    if run_config['use_gpu']:
        model = nn.DataParallel(model)
        model.cuda()
    logger.info('Done')

    if config['data_config']['use_mixup']:
        train_criterion = CrossEntropyLoss(size_average=True)
    else:
        train_criterion = nn.CrossEntropyLoss(size_average=True)
    test_criterion = nn.CrossEntropyLoss(size_average=True)

    # create optimizer
    optim_config['steps_per_epoch'] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    # run test before start training
    if run_config['test_first']:
        test(0, model, test_criterion, test_loader, run_config, writer)

    state = {
        'config': config,
        'state_dict': None,
        'optimizer': None,
        'epoch': 0,
        'accuracy': 0,
        'best_accuracy': 0,
        'best_epoch': 0,
    }
    for epoch in range(1, optim_config['epochs'] + 1):
        # train
        dfrow = train(epoch, model, optimizer, scheduler, train_criterion,
              train_loader, config, writer, moment_dict)

        # test
        test_loss, accuracy = test(epoch, model, test_criterion, test_loader, run_config,
                        writer)

        dfrow.append(test_loss)
        dfrow.append(accuracy)

        if epoch <= 4 or epoch % 5 == 0:
            resultsdf.loc[resultsdf.shape[0]] = list(dfrow)
            resultsdf.to_csv(os.path.join(outdir, 'results.csv'))

        # update state dictionary
        state = update_state(state, epoch, accuracy, model, optimizer)

        # save model
        save_checkpoint(state, outdir)

    if run_config['tensorboard']:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)


if __name__ == '__main__':
    main()
