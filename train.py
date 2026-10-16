import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
import cv2
from seg_metric import SegmentationMetric
import random
import shutil
import setproctitle
import time
import logging

from dataset import *
from custom_transforms import Mixup, edge_contour
from loss import *


class FullModel(nn.Module):

    def __init__(self, model, args2):
        super(FullModel, self).__init__()
        self.model = model
        self.use_mixup = args2.use_mixup
        self.use_edge = args2.use_edge

        self.ce_loss = CrossEntropyLoss()

        self.edge_loss1 = CrossEntropyLoss_binary()
        self.edge_loss2 = BoundaryLoss()

        if self.use_mixup:
            self.mixup = Mixup(use_edge=args2.use_edge)

    def forward(self, input, label=None, train=True):

        if train and self.use_mixup and label is not None:
            if self.use_edge:
                loss = self.mixup(input, label, [self.ce_loss, self.edge_loss], self.model)
            else:
                loss = self.mixup(input, label, self.ce_loss, self.model)
            return loss

        output = self.model(input)
        if train:
            losses = 0
            if isinstance(output, (list, tuple)):
                if self.use_edge:
                    for i in range(len(output) - 1):
                        loss = self.ce_loss(output[i], label)
                        losses += loss
                    # losses += self.edge_loss(output[-1], edge_contour(label).long())
                    losses += self.edge_loss1(output[-1], edge_contour(label).long())
                    losses += self.edge_loss2(output[-1], label)* 0.7 
                else:
                    for i in range(len(output)):
                        loss = self.ce_loss(output[i], label)
                        losses += loss
            else:
                losses = self.ce_loss(output, label)
            return losses
        else:
            if isinstance(output, (list, tuple)):
                return output[0]
            else:
                return output


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


class params():
    def __init__(self, args2):
        models = args2.models
        if models == 'HRNet_32':
            self.STAGE2 = {'NUM_MODULES': 1,
                           'NUM_BRANCHES': 2,
                           'NUM_BLOCKS': [4, 4],
                           'NUM_CHANNELS': [32, 64],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE3 = {'NUM_MODULES': 4,
                           'NUM_BRANCHES': 3,
                           'NUM_BLOCKS': [4, 4, 4],
                           'NUM_CHANNELS': [32, 64, 128],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE4 = {'NUM_MODULES': 3,
                           'NUM_BRANCHES': 4,
                           'NUM_BLOCKS': [4, 4, 4, 4],
                           'NUM_CHANNELS': [32, 64, 128, 256],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
        elif models == 'HRNet_48':
            "hrnet48"
            self.STAGE2 = {'NUM_MODULES': 1,
                           'NUM_BRANCHES': 2,
                           'NUM_BLOCKS': [4, 4],
                           'NUM_CHANNELS': [48, 96],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE3 = {'NUM_MODULES': 4,
                           'NUM_BRANCHES': 3,
                           'NUM_BLOCKS': [4, 4, 4],
                           'NUM_CHANNELS': [48, 96, 192],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}
            self.STAGE4 = {'NUM_MODULES': 3,
                           'NUM_BRANCHES': 4,
                           'NUM_BLOCKS': [4, 4, 4, 4],
                           'NUM_CHANNELS': [48, 96, 192, 384],
                           'BLOCK': 'BASIC',
                           'FUSE_METHOD': 'SUM'}


def save_model_file(save_dir, save_name):
    save_dir = os.path.join(save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/weights/')
        os.makedirs(save_dir + '/outputs/')
    # for file in os.listdir('.'):
    #     if os.path.isfile(file):
    #         shutil.copy(file, save_dir)
    if not os.path.exists(os.path.join(save_dir, 'models')):
        shutil.copytree('./models', os.path.join(save_dir, 'models'))
    logging.basicConfig(filename=save_dir + '/train.log', level=logging.INFO)


def get_model(args2, device, models='DANet'):
    if models in ['swinT', 'resT']:
        print(models, args2.head)
    else:
        print(models)
    # if args2.dataset in ['SL1']:
    nclass = 2
    # assert models in ['danet', 'bisenetv2', 'pspnet', 'segbase', 'swinT',
    #                   'deeplabv3', 'fcn', 'fpn', 'unet', 'resT', 'bs']
    if models == 'bs':
        from models.bs import BSModel
        model = BSModel(nclass=nclass, backbone='resnet50', aux=True, edge_aux=True,
                        pretrained_base=True)  # edge_aux=args2.use_edge)
    if models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=nclass, backbone='resnet50', pretrained_base=True)
    if models == 'bisenetv2':
        from models.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(nclass=nclass)
    if models == 'pspnet':
        from models.pspnet import PSPNet
        model = PSPNet(nclass=nclass, backbone='resnet50', pretrained_base=True)
    if models == 'segbase':
        from models.segbase import SegBase
        model = SegBase(nclass=nclass, backbone='resnet50', pretrained_base=True)
    if models == 'swinT':
        from models.swinT import swin_tiny as swinT
        model = swinT(nclass=nclass, pretrained=True, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'resT':
        from models.resT import rest_tiny as resT
        model = resT(nclass=nclass, pretrained=True, aux=True, head=args2.head, edge_aux=args2.use_edge)
    if models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=nclass, backbone='resnet50', pretrained_base=True)
    if models == 'fcn':
        from models.fcn import FCN16s
        model = FCN16s(nclass=nclass)
    if models == 'fpn':
        from models.fpn import FPN
        model = FPN(nclass=nclass)
    if models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=nclass)
    if models == 'hed_unet':
        from models.hde_unet import HEDUNet
        model = HEDUNet(input_channels=3, output_channels=nclass, base_channels=16, merging='attention', stack_height=5,
                        deep_supervision=False)
    if models == 'dbenet':
        from models.dbenet import DBENet
        model = DBENet(num_classes=nclass)
    if models == 'acunet':
        from models.acunet import ACUNet
        model = ACUNet(nclass=nclass)
    if models == 'CGGLNet':
        from models.CGGLNet import CGGLNet
        model = CGGLNet(num_classes=nclass, )
    if models == 'DCNet':
        from models.DCNnet import DCNet
        model = DCNet(num_classes=nclass, img_size=512, )

    model = FullModel(model, args2)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print(model)
    print("-----------------------------")
    model = model.to(device)
    # model = nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args2.local_rank], output_device=args2.local_rank, find_unused_parameters=True)
    return model


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='SL1')
    parser.add_argument("--end_epoch", type=int, default=200)
    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--train_batchsize", type=int, default=8)
    parser.add_argument("--val_batchsize", type=int, default=2)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[512, 512], help='H, W')
    parser.add_argument("--information", type=str, default='fusion')
    parser.add_argument("--models", type=str, default='bs')
    parser.add_argument("--head", type=str, default='seghead')
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default='./work_dir')
    parser.add_argument("--use_edge", type=int, default=0)
    parser.add_argument("--use_mixup", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


def train():
    """###############  Notice  ###############"""
    args2 = parse_args()
    torch.manual_seed(args2.seed)
    torch.cuda.manual_seed(args2.seed)
    random.seed(args2.seed)
    np.random.seed(args2.seed)

    save_name = "{}_lr{}_epoch{}_batchsize{}_{}".format(args2.models, args2.lr, args2.end_epoch,
                                                        args2.train_batchsize , args2.information)
    save_dir = args2.save_dir
    if args2.local_rank == 0:
        save_model_file(save_dir=save_dir, save_name=save_name)
    device = torch.device(('cuda:{}').format(args2.local_rank))

    model = get_model(args2, device, models=args2.models)

    if args2.dataset == "SL1":
        dataset_train = SL1(train=True, dataset=args2.dataset, crop_szie=args2.crop_size)
        dataset_val = SL1(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)
    elif args2.dataset == "SLSD":
        dataset_train = SLSD(train=True, dataset=args2.dataset, crop_szie=args2.crop_size)
        dataset_val = SLSD(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)


    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args2.train_batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args2.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    optimizer = torch.optim.AdamW(#[{'params':
                                  #      filter(lambda p: p.requires_grad,
                                   #            model.parameters()),
                                    #'lr': args2.lr}],
                                  model.parameters(),
                                  lr=args2.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  )
    # 打印优化器的参数组
    print("Optimizer parameters:")
    for param_group in optimizer.param_groups:
        print(param_group.keys())
        for param in param_group['params']:
            if isinstance(param, nn.Parameter):
                print("  Found parameter:", param.shape, "requires_grad:", param.requires_grad)
            else:
                print("  Found non-parameter:", type(param))

    start = time.time()
    miou = 0
    acc = 0
    f1 = 0
    bf1 = 0
    precision = 0
    recall = 0
    best_miou = 0
    class_iou = [0, 0]
    best_acc = 0
    best_f1 = 0
    best_bf1 = 0
    last_epoch = 0
    test_epoch = args2.end_epoch - 3
    ave_loss = AverageMeter()

    weight_save_dir = os.path.join(save_dir, save_name + '/weights')
    model_state_file = weight_save_dir + "/{}_lr{}_epoch{}_batchsize{}_{}.pkl.tar" \
        .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize , args2.information)
    if os.path.isfile(model_state_file):
        print('loaded successfully')
        logging.info("=> loading checkpoint '{}'".format(model_state_file))
        checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
        checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
        best_miou = checkpoint['best_miou']
        best_acc = checkpoint['best_acc']
        best_f1 = checkpoint['best_f1']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(
            model_state_file, checkpoint['epoch']))

    for epoch in range(last_epoch, args2.end_epoch):

        model.train()
        setproctitle.setproctitle("xzy:" + str(epoch) + "/" + "{}".format(args2.end_epoch))

        for i, sample in enumerate(dataloader_train):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            label = label.long().squeeze(1)
            losses = model(image, label)
            loss = losses.mean()
            ave_loss.update(loss.item())
            lenth_iter = len(dataloader_train)
            lr = adjust_learning_rate(optimizer,
                                      args2.lr,
                                      args2.end_epoch * lenth_iter,
                                      i + epoch * lenth_iter,
                                      args2.warm_epochs * lenth_iter
                                      )

            if i % 50 == 0:
                print_loss = ave_loss.average()

                if args2.local_rank == 0:
                    time_cost = time.time() - start
                    start = time.time()
                    print("epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, "
                        "best_miou:{:.4f}, miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}".
                        format(epoch, args2.end_epoch, i, len(dataloader_train), print_loss, time_cost, lr,
                                best_miou, miou, acc, f1, precision, recall))  #  .item()

                    logging.info(
                        "epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, "
                        "best_miou:{:.4f}, miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}".
                            format(epoch, args2.end_epoch, i, len(dataloader_train), print_loss, time_cost, lr,
                                best_miou, miou, acc, f1, precision, recall))  # .item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch > 0:  # test_epoch:
            miou, acc, f1, precision, recall, class_iou, bf1 = validate(dataloader_val, device, model, args2)
            # miou = miou
            # acc = acc
            # f1 = f1
            # precision = precision
            # recall = recall
            # bf1 = bf1
            # for i in range(2):
            #     class_iou[i] = class_iou[i]

        if args2.local_rank == 0:
            print('miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}, bf1:{:.4f}'.format(
                            miou, acc, f1, precision, recall, bf1))
            print('sea:{}, land:{}'.format(class_iou[0], class_iou[1]))
            if (epoch % 10 == 0 and epoch > 50) or (epoch > test_epoch):
                torch.save(model.state_dict(),
                           weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_{}_xzy_{}.pkl'
                           .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize ,
                                   args2.information, epoch))

            if miou >= best_miou and miou != 0:
                best_miou = miou
                best_acc, best_f1 = acc, f1
                if best_bf1 < bf1:
                    best_bf1 = bf1
                best_weight_name = weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_{}_best.pkl'.format(
                    args2.models, args2.lr, args2.end_epoch, args2.train_batchsize , args2.information)
                torch.save(model.state_dict(), best_weight_name)
                torch.save(model.state_dict(), weight_save_dir + '/best_weight.pkl')

            torch.save({
                'epoch': epoch + 1,
                'best_miou': best_miou,
                'best_acc': best_acc,
                'best_f1': best_f1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_{}.pkl.tar'
               .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize , args2.information))
    if args2.local_rank == 0:
        torch.save(model.state_dict(),
                   weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_{}_xzy_{}.pkl'
                   .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize ,
                           args2.information, args2.end_epoch))
        try:
            print("epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, best_miou:{:.4f}, "
                  "miou:{:.4f}, acc:{:.4f} f1:{:.4f}, precision:{:.4f}, recall:{:.4f}".
                  format(epoch, args2.end_epoch, i, len(dataloader_train),
                         print_loss, time_cost, lr, best_miou, miou, acc, f1, precision, recall))

            logging.info(
                "epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, best_miou:{:.4f}, "
                "miou:{:.4f}, acc:{:.4f} f1:{:.4f}, precision:{:.4f}, recall:{:.4f}".
                    format(epoch, args2.end_epoch, i, len(dataloader_train),
                           print_loss, time_cost, lr, best_miou, miou, acc, f1, precision, recall))
        except:
            pass

        logging.info("***************super param*****************")
        logging.info(
            "dataset:{} information:{} lr:{} epoch:{} batchsize:{} best_miou:{} best_acc:{} best_f1:{} best_bf1:{}"
            .format(args2.dataset, args2.information, args2.lr, args2.end_epoch, args2.train_batchsize ,
                    best_miou, best_acc, best_f1, best_bf1))
        logging.info("***************end*************************")

        print("***************super param*****************")
        print("dataset:{} information:{} lr:{} epoch:{} batchsize:{} best_miou:{} best_acc:{} best_f1:{}"
              .format(args2.dataset, args2.information, args2.lr, args2.end_epoch, args2.train_batchsize ,
                      best_miou, best_acc, best_f1))
        print("***************end*************************")


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr * ((1 - float(cur_iters - warmup_iter) / (max_iters - warmup_iter)) ** (power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def validate(dataloader_val, device, model, args2):
    model.eval()
    # MIOU = [0] # 移除这些列表初始化
    # CLASS_IOU = [0, 0]
    # ACC = [0]
    # F1 = [0]
    # Precision = [0]
    # Recall = [0]
    # BF1 = [0.0]

    nclass = 2
    total_bf1 = torch.zeros(1, device=device)
    metric = SegmentationMetric(nclass)
    with torch.no_grad():
        Precision_tensor = torch.tensor(0.0, device=device)  # 初始化 Precision_tensor
        Recall_tensor = torch.tensor(0.0, device=device)     # 初始化 Recall_tensor
        for i, sample in enumerate(dataloader_val):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            label = label.long().squeeze(1)
            label_ori = label
            logit = model(image, label, train=False)
            logit_ori = logit
            logit = logit.argmax(dim=1)
            logit = logit.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            metric.addBatch(logit, label)
            bf1 = metric.BoundaryF1(logit_ori, label_ori, device)
            total_bf1 += bf1

    iou = metric.IntersectionOverUnion()
    acc = metric.Accuracy()
    precision = metric.Precision()
    recall = metric.Recall()
    class_iou = iou[0:nclass]
    miou = np.nanmean(iou[0:nclass])
    mprecision = np.nanmean(precision[0:nclass])
    mrecall = np.nanmean(recall[0:nclass])
    total_bf1 = total_bf1 / len(dataloader_val)

    Precision_tensor = torch.tensor(mprecision, device=device)
    Recall_tensor = torch.tensor(mrecall, device=device)
    # F1 的计算也要使用张量，确保它们在同一设备上
    # 这里的 F1 计算如果 mprecision + mrecall 为 0 会有问题，但通常不会在验证中发生
    F1_tensor = 2 * Precision_tensor * Recall_tensor / (Precision_tensor + Recall_tensor + 1e-8) # 加一个小的 epsilon 防止除以零
    # 直接将计算出的 NumPy 值转换为 PyTorch 张量，而不是通过列表累加


    MIOU_tensor = torch.tensor(miou, device=device)
    CLASS_IOU_tensor = torch.tensor(class_iou, device=device)
    ACC_tensor = torch.tensor(acc, device=device)
    F1_tensor = torch.tensor(F1_tensor, device=device)
    Precision_tensor = torch.tensor(mprecision, device=device)
    Recall_tensor = torch.tensor(mrecall, device=device)
    
    
    
    
    
    return (
        MIOU_tensor.item(),   # Convert to float
        ACC_tensor.item(),    # Convert to float
        F1_tensor.item(),     # Convert to float
        Precision_tensor.item(),  # Convert to float
        Recall_tensor.item(),   # Convert to float
        CLASS_IOU_tensor.cpu().numpy(), # Convert to numpy array
        total_bf1.item()      # Convert to float
    )


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ.setdefault('RANK', '0')
    # os.environ.setdefault('WORLD_SIZE', '1')
    # os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    # os.environ.setdefault('MASTER_PORT', '29556')

    cudnn.benchmark = True
    cudnn.enabled = True

    # don't use cudnn
    # cudnn.benchmark = False
    # cudnn.deterministic = True

    train()
