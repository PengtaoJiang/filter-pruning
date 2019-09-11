import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from vltools import Logger
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from vltools.pytorch.datasets import ilsvrc2012
import vltools.pytorch as vlpytorch

from dali_ilsvrc import dali_ilsvrc_loader

import vgg_cifar
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', metavar='STR', default=None, help='model')
parser.add_argument('--data', metavar='DIR', default="/media/ssd0/ilsvrc12/rec", help='path to dataset')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--milestones', default="30,60,90", type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/vgg16bn-imagenet")
parser.add_argument('--randseed', type=int, help='random seed', default=None)
#
parser.add_argument('--use-dali', action="store_true")
parser.add_argument('--no-retrain', action="store_true")
parser.add_argument('--sparsity', type=float, default=1e-5, help='sparsity regularization')
parser.add_argument('--retrain', action="store_true")
parser.add_argument('--prune-type', type=int, default=0, help="prune method")
parser.add_argument('--percent', type=float, default=0.5, help='pruning percent')
args = parser.parse_args()

milestones = [int(i) for i in args.milestones.split(',')]

if args.randseed == None:
    args.randseed = np.random.randint(1000)
args.tmp = args.tmp.strip("/")
args.tmp = args.tmp+"-seed%d"%args.randseed

# Random seed
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed_all(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

THIS_DIR = abspath(dirname(__file__))
os.makedirs(args.tmp, exist_ok=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()

tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
logger = Logger(join(args.tmp, "log.txt"))

# prune types
# 0: prune with bn factor globally
# 1: prune with (bn factor x next conv weight) globally
# 2: prune with bn factor locally.
#    Filters whose factors are less than the 0.01 times maximal factors within the same layer will be pruned
def get_factors(model):

    factors = {}
    named_modules = dict(model.named_modules())
    modules = list(named_modules.values())
    for idx, (name, m) in enumerate(named_modules.items()):
        if isinstance(m, nn.BatchNorm2d):
            nextid = idx+1
            next_conv = modules[nextid]
            while not (isinstance(next_conv, nn.Conv2d) or isinstance(next_conv, nn.Linear)):
                nextid += 1
                next_conv = modules[nextid]

            next_weight = next_conv.weight.data
            if False:
                next_weight = next_weight.transpose(dim0=0, dim1=1).contiguous()
                next_weight = next_weight.view(next_weight.shape[0], -1)
                next_weight = torch.norm(next_weight, p=2, dim=1)
            else:
                if isinstance(next_conv, nn.Conv2d):
                    next_weight = next_weight.abs().mean(dim=(0,2,3))
                elif isinstance(next_conv, nn.Linear):
                    next_weight = next_weight.abs().mean(dim=(0))

            bnw = m.weight.data.abs()

            if next_weight.numel() == 25088:
                # in case of the last conv
                factor = bnw

            # _, idx0 = bnw.sort()
            # _, idx1 = factor.sort()

            if args.prune_type == 0:
                factors[name] = bnw
            elif args.prune_type == 1:
                factors[name] = factor
            elif args.prune_type == 2:
                factors[name] = factor
            else:
                raise ValueError("?")

    return factors

def get_sparsity(factors, thres=0.01):
    total0 = 0
    total = 0
    for v in factors.values():
        total0 += (v <= v.max()*thres).sum()
        total += v.numel()
    return (total0.float() / total).item()

def main():

    logger.info(args)

    # model and optimizer
    model_name = "torchvision.models.vgg11_bn()"
    model = eval(model_name)

    # reinitiate bn factors to 0.5
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(0.5)
    
    model = nn.DataParallel(model.cuda())

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    logger.info("Model details:")
    logger.info(model)
    logger.info("Optimizer details:")
    logger.info(optimizer)

    # dataloaders
    if args.use_dali:
        train_loader, val_loader = dali_ilsvrc_loader(args.data, num_gpus=2, batch_size=args.batch_size, num_threads_per_gpu=2)
    else:
        train_loader, val_loader = ilsvrc2012(args.data, bs=args.batch_size)

    # records
    best_acc1 = 0

    # save initial weights
    save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'best_acc1': -1,
            }, False, path=args.tmp, filename="initial-weights.pth")

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                      milestones=milestones,
                      gamma=args.gamma)

    last_sparsity = get_sparsity(get_factors(model))
    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        loss = train(train_loader, model, optimizer, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)
        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=args.tmp)

        logger.info("Best acc1=%.5f" % best_acc1)

        model_sparsity = get_sparsity(get_factors(model))

        # for prune-type == 2
        if args.prune_type == 2:
            target_sparsity = args.percent
            sparsity_gain = (model_sparsity - last_sparsity)
            expected_sparsity_gain = (target_sparsity - model_sparsity) / (args.epochs - epoch)
            if model_sparsity < target_sparsity:
                # not sparse enough
                if sparsity_gain < expected_sparsity_gain:
                    logger.info("Sparsity gain %f (expected%f), increasing sparse penalty."%(sparsity_gain, expected_sparsity_gain))
                    args.sparsity += 1e-5
            elif model_sparsity > target_sparsity:
                # over sparse
                if model_sparsity > last_sparsity and args.sparsity > 0:
                    args.sparsity -= 1e-5

            logger.info("Model sparsity=%f (last=%f, target=%f), args.sparsity=%f" %\
                (model_sparsity, last_sparsity, target_sparsity, args.sparsity))
            last_sparsity = model_sparsity

        lr = optimizer.param_groups[0]["lr"]
        bn_l1 = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_l1 += m.weight.abs().mean()

        tfboard_writer.add_scalar('train/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('train/lr_epoch', lr, epoch)
        tfboard_writer.add_scalar('train/BN-L1', bn_l1, epoch)
        tfboard_writer.add_scalar('train/model sparsity', model_sparsity, epoch)
        tfboard_writer.add_scalar('train/sparse penalty', args.sparsity, epoch)

        tfboard_writer.add_scalar('test/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5_epoch', acc5, epoch)

    logger.info("Optimization done, ALL results saved to %s." % args.tmp)

    # evaluate before pruning
    logger.info("evaluating before pruning...")
    validate(val_loader, model, args.epochs)

    factors = get_factors(model)
    sparsity = get_sparsity(factors)
    factors_all = torch.cat(list(factors.values()))
    factors_all, _ = torch.sort(factors_all)
    thres = factors_all[int(factors_all.numel() * args.percent)]
    logger.info("Model sparsity: %f"%sparsity)
    logger.info("Pruning threshold: %f"%thres)

    # mask pruning
    prune_mask = {}
    total_filters = 0
    pruned_filters = 0
    for idx, (name, m) in enumerate(model.named_modules()):
        if not isinstance(m, nn.BatchNorm2d):
            continue
        factor = factors[name]
        if args.prune_type == 0 or args.prune_type == 1:
            prune_mask[name] = (factor >= thres)
        elif args.prune_type == 2:
            prune_mask[name] = factor >= factor.max() * 0.01

        total_filters += factor.numel()
        pruned_filters += (prune_mask[name].bitwise_not()).sum()

        m.weight.data[prune_mask[name].bitwise_not()] = 0
        m.bias.data[prune_mask[name].bitwise_not()] = 0

    prune_rate = float(pruned_filters)/total_filters
    logger.info("Totally %d filters, %d has been pruned, pruning rate %f"%(total_filters, pruned_filters, prune_rate))

    logger.info("evaluating after masking...")
    validate(val_loader, model, args.epochs)

    # reload model
    model = eval(model_name).cuda()
    model.load_state_dict(torch.load(join(args.tmp, "checkpoint.pth"))["state_dict"])

    # do real pruning
    modules = list(model.modules())
    named_modules = dict(model.named_modules())
    with torch.no_grad(): 
        for idx, (name, m) in enumerate(named_modules.items()):
            if not isinstance(m, nn.BatchNorm2d):
                continue

            previous_conv = modules[idx-1]
            
            nextid = idx+1
            next_conv = modules[nextid]
            while not (isinstance(next_conv, nn.Conv2d) or isinstance(next_conv, nn.Linear)):
                next_conv = modules[nextid]
                nextid += 1

            assert isinstance(previous_conv, nn.Conv2d), type(previous_conv)
            assert isinstance(next_conv, nn.Conv2d) or isinstance(next_conv, nn.Linear), type(next_conv)

            mask = prune_mask[name]

            m.weight.data = m.weight.data[mask]
            m.bias.data = m.bias.data[mask]
            m.running_mean = m.running_mean[mask]
            m.running_var = m.running_var[mask]

            previous_conv.weight.data = previous_conv.weight.data[mask]
            previous_conv.bias.data = previous_conv.bias.data[mask]

            next_conv.weight.data = next_conv.weight.data[:, mask]

    # clear gradients
    for p in model.parameters():
        p.grad = None

    logger.info("evaluating after real pruning...")
    acc1, acc5 = validate(val_loader, model, args.epochs)
    tfboard_writer.add_scalar('retrain/acc1_epoch', acc1, -1)
    tfboard_writer.add_scalar('retrain/acc5_epoch', acc5, -1)

    # shutdown when `args.no-retrain` is triggered
    if args.no_retrain: return

    # retrain
    optimizer_retrain = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler_retrain = lr_scheduler.MultiStepLR(optimizer_retrain,
                      milestones=milestones,
                      gamma=args.gamma)
    best_acc1 = 0
    for epoch in range(0, args.epochs):

        # train and evaluate
        loss = train(train_loader, model, optimizer_retrain, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)
        scheduler_retrain.step()

        lr = optimizer_retrain.param_groups[0]["lr"]

        tfboard_writer.add_scalar('retrain/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('retrain/lr_epoch', lr, epoch)
        tfboard_writer.add_scalar('retrain/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('retrain/acc5_epoch', acc5, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1

        if is_best:
            best_acc1 = acc1

        logger.info("Best acc1=%.5f" % best_acc1)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=args.tmp, filename="checkpoint-retrain0.pth")


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.use_dali:
        train_loader_len = int(train_loader._size / 100)
    else:
        train_loader_len = len(train_loader)


    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_dali:
            target = torch.cat([i["label"].to(torch.device('cuda:0')) for i in data], dim=0)
            data = torch.cat([i["data"].to(torch.device('cuda:0')) for i in data], dim=0)
            target = target.cuda().squeeze().long()
        else:
            data, target = data
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # impose L1 penalty to BN factors
        if args.sparsity != 0:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.grad.data.add_(args.sparsity*torch.sign(m.weight.data))  # L1

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if i % args.print_freq == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr:.4f}'.format(
                   epoch, args.epochs, i, train_loader_len,
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5,
                   lr=lr))

    return losses.avg

def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.use_dali:
        val_loader_len = int(val_loader._size / 100)
    else:
        val_loader_len = len(val_loader)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            
            if args.use_dali:
                target = torch.cat([i["label"].to(torch.device('cuda:0')) for i in data], dim=0)
                data = torch.cat([i["data"].to(torch.device('cuda:0')) for i in data], dim=0)
                target = target.cuda(non_blocking=True).squeeze().long()
            else:
                data, target = data
                data = data.cuda()
                target = target.cuda(non_blocking=True)

            data = data.cuda()
            # compute output
            output = model(data)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                       i, val_loader_len, loss=losses, top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
