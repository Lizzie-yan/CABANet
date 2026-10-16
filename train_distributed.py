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
import torch.distributed as dist

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
                loss = self.mixup(input, label, [self.ce_loss, self.edge_loss1], self.model)
            else:
                loss = self.mixup(input, label, self.ce_loss, self.model)
            return loss

        output = self.model(input) # 模型的输出

        if train:
            losses = 0
            
            # 判断模型输出是列表（多任务，如BSModel）还是单个张量（单任务，如DBBANet）
            if isinstance(output, (list, tuple)):
                # --- 这是为 BSModel 设计的多任务损失计算路径 ---
                
                # 只有当 use_edge=1 并且输出列表长度>1时，才计算边界损失
                if self.use_edge and len(output) > 1:
                    # 主分割和辅助分割损失
                    for i in range(len(output) - 1):
                        losses += self.ce_loss(output[i], label)
                    
                    # 边界损失
                    # 注意：为了让代码更健壮，我们假设边界损失只在列表长度>1时计算
                    if len(output) > 2: # 假设有主、辅、边界三个输出
                        losses += self.edge_loss1(output[-1], edge_contour(label).long())
                        losses += self.edge_loss2(output[-1], label) * 0.7
                    elif len(output) == 2: # 假设只有主分割和边界两个输出
                        losses += self.edge_loss1(output[-1], edge_contour(label).long())
                        losses += self.edge_loss2(output[-1], label) * 0.7

                else: # use_edge=0, 对所有输出都用CE loss
                    for out_tensor in output:
                        losses += self.ce_loss(out_tensor, label)

            else: # 如果输出是单个张量
                # --- 这是为 DBBANet 设计的单任务损失计算路径 ---
                # 无论 use_edge 是什么，都只计算交叉熵损失
                losses = self.ce_loss(output, label)
                
            return losses

        else: # 推理阶段
            # 推理逻辑基本保持不变，因为它已经能处理两种情况
            if isinstance(output, (list, tuple)):
                return output[0] # 多任务模型，返回第一个（主预测）
            else:
                return output # 单任务模型，直接返回


def save_model_file(save_dir, save_name):
    save_dir = os.path.join(save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/weights/')
        os.makedirs(save_dir + '/outputs/')
    if not os.path.exists(os.path.join(save_dir, 'models')):
        shutil.copytree('./models', os.path.join(save_dir, 'models'))
    logging.basicConfig(filename=save_dir + '/train.log', level=logging.INFO)


def get_model(args2):
    models = args2.models
    if models in ['swinT', 'resT']:
        print(f"创建模型: {models}, Head: {args2.head}")
    else:
        print(f"创建模型: {models}")

    nclass = 2
    if models == 'bs':
        from models.bs import BSModel
        # 边界模块单元级消融：EDGE_ABLATION = none | bau | bau_ceu | full
        _abl = os.environ.get('EDGE_ABLATION', 'full').lower()
        _units = {
            'none': dict(bau=False, ceu=False, bffu=False),
            'bau': dict(bau=True, ceu=False, bffu=False),
            'bau_ceu': dict(bau=True, ceu=True, bffu=False),
            'full': dict(bau=True, ceu=True, bffu=True),
        }.get(_abl, dict(bau=True, ceu=True, bffu=True))
        print('BSModel edge units: %s -> %s' % (_abl, _units))
        model = BSModel(nclass=nclass, backbone='resnet50', aux=True, edge_aux=True, pretrained_base=True, edge_units=_units)
    elif models == 'dbbanet': # 我们给新模型起一个名字叫 'dbbanet'
        # 假设你的DBBANet代码保存在 'models/dbbanet.py'
        from models.DBBANet import DBBANet 
        model = DBBANet(num_class=nclass)
    elif models == 'cmlformer':
        # 对比方法 CMLFormer（官方源码在 work_dir/CMLFormer/，保持字节不变；
        # thop stub、super() 修正、NHWC<->NCHW 转换都在 cmlformer_wrapper.py 里）
        import sys as _sys
        _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir', 'CMLFormer')
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from cmlformer_wrapper import build_cmlformer
        model = build_cmlformer(num_classes=nclass)
    elif models == 'mcatunet':
        # 对比方法 MCAT-UNet（官方源码逐字节复制进 work_dir/MCAT-UNet/mmseg_shim/）
        import sys as _sys
        _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work_dir', 'MCAT-UNet')
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from mcatunet_wrapper import build_mcatunet
        model = build_mcatunet(num_classes=nclass)
    elif models == 'hed_unet':
        from models.hde_unet import HEDUNet
        model = HEDUNet(input_channels=3, output_channels=nclass, base_channels=16,
                        merging='attention', stack_height=5, deep_supervision=False)
    elif models == 'dbenet':
        from models.dbenet import DBENet
        model = DBENet(num_classes=nclass)
    elif models == 'acunet':
        from models.acunet import ACUNet
        model = ACUNet(nclass=nclass)
    elif models == 'CGGLNet':
        from models.CGGLNet import CGGLNet
        model = CGGLNet(num_classes=nclass)
    elif models == 'DCNet':
        from models.DCNnet import DCNet
        model = DCNet(num_classes=nclass, img_size=512)
    elif models == 'crfnet':
        import os as _os, sys as _sys
        _r = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           'work_dir', 'CRFNet')
        _sys.path.insert(0, _r)
        from crfnet_wrapper import build_crfnet
        model = build_crfnet(num_classes=nclass)
    elif models == 'hstunet':
        import os as _os, sys as _sys
        _r = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           'work_dir', 'HSTUNet')
        _sys.path.insert(0, _r)
        from hstunet_wrapper import build_hstunet
        model = build_hstunet(num_classes=nclass, img_size=512)
    elif models == 'danet':
        from models.danet import DANet
        model = DANet(nclass=nclass, backbone='resnet50', pretrained_base=True)
    # ... (此处省略了您原来代码中所有模型的if/elif分支，它们不需要改变)
    elif models == 'bisenetv2':
        from models.bisenetv2 import BiSeNetV2
        model = BiSeNetV2(nclass=nclass)
    elif models == 'pspnet':
        from models.pspnet import PSPNet
        model = PSPNet(nclass=nclass, backbone='resnet50', pretrained_base=True)
    elif models == 'segbase':
        from models.segbase import SegBase
        model = SegBase(nclass=nclass, backbone='resnet50', pretrained_base=True)
    elif models == 'swinT':
        from models.swinT import swin_tiny as swinT
        model = swinT(nclass=nclass, pretrained=True, aux=True, head=args2.head, edge_aux=args2.use_edge)
    elif models == 'resT':
        from models.resT import rest_tiny as resT
        model = resT(nclass=nclass, pretrained=True, aux=True, head=args2.head, edge_aux=args2.use_edge)
    elif models == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(nclass=nclass, backbone='resnet50', pretrained_base=True)
    elif models == 'fcn':
        from models.fcn import FCN16s
        model = FCN16s(nclass=nclass)
    elif models == 'fpn':
        from models.fpn import FPN
        model = FPN(nclass=nclass)
    elif models == 'unet':
        from models.unet import UNet
        model = UNet(nclass=nclass)
    # ... 依此类推，补全所有模型
    else:
        raise NotImplementedError(f"模型 {models} 未实现")

    # FullModel包装器保持不变
    model = FullModel(model, args2)
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
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
    parser.add_argument("--train_batchsize", type=int, default=8, help="这是每个GPU上的批量大小")
    parser.add_argument("--val_batchsize", type=int, default=2)
    parser.add_argument("--crop_size", type=int, nargs='+', default=[512, 512], help='H, W')
    parser.add_argument("--information", type=str, default='scsefusion')
    parser.add_argument("--models", type=str, default='bs')
    parser.add_argument("--head", type=str, default='seghead')
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default='./work_dir')
    parser.add_argument("--use_edge", type=int, default=0)
    parser.add_argument("--use_mixup", type=int, default=0)
    
    # --- 新增的分布式训练参数 ---
    parser.add_argument('--world_size', default=1, type=int,
                        help='分布式训练中的进程总数 (number of distributed processes)')
    parser.add_argument('--rank', default=0, type=int,
                        help='当前进程在所有进程中的序号 (rank of current process)')
    parser.add_argument('--dist_url', default='env://', 
                        help='用于设置分布式训练的URL (url used to set up distributed training)')
    parser.add_argument('--distributed', action='store_true',
                        help='是否使用分布式训练 (Use distributed training)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='要使用的GPU的ID。对于分布式训练，这通常是自动设置的 (GPU id to use.)')

    parser.add_argument('opts',
                        help="通过命令行修改配置选项 (Modify config options using the command-line)",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                        help='Local rank for distributed training. Automatically set.')

    parser.add_argument('opts',
                        help="通过命令行修改配置选项",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


def init_distributed_mode(args):
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK']) # LOCAL_RANK由torch.distributed.launch或run提供
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier() # 等待所有进程都初始化完毕

def main_worker(args2):
    """这是主要的训练和验证逻辑"""
    torch.manual_seed(args2.seed)
    torch.cuda.manual_seed(args2.seed)
    random.seed(args2.seed)
    np.random.seed(args2.seed)

    # 只有主进程(rank 0)进行文件和目录操作
    if args2.rank == 0:
        save_name = "{}_lr{}_epoch{}_batchsize{}_{}".format(args2.models, args2.lr, args2.end_epoch,
                                                            args2.train_batchsize * args2.world_size, args2.information) # 总batchsize
        save_dir = args2.save_dir
        save_model_file(save_dir=save_dir, save_name=save_name)
    
    # 设置当前进程使用的设备
    device = torch.device(args2.gpu)

    model = get_model(args2)
    model.to(device)

    # 如果是分布式训练，则用DDP包装模型
    if args2.distributed:
        # find_unused_parameters=True对于某些有辅助分支的模型是必要的
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args2.gpu], find_unused_parameters=True)

    if args2.dataset == "SL1":
        dataset_train = SL1(train=True, dataset=args2.dataset, crop_szie=args2.crop_size)
        dataset_val = SL1(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)
    elif args2.dataset == "SLSD":
        dataset_train = SLSD(train=True, dataset=args2.dataset, crop_szie=args2.crop_size)
        dataset_val = SLSD(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)
    else:
        raise NotImplementedError(f"数据集 {args2.dataset} 未实现")

    # 为分布式训练创建Sampler
    if args2.distributed:
        train_sampler = DistributedSampler(dataset_train, num_replicas=args2.world_size, rank=args2.rank, shuffle=True)
        val_sampler = DistributedSampler(dataset_val, num_replicas=args2.world_size, rank=args2.rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args2.train_batchsize,
        shuffle=(train_sampler is None), # 当sampler存在时，shuffle必须为False
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args2.val_batchsize,
        shuffle=False, # 验证时通常不shuffle
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )

    # 根据模型是否被DDP包装来获取参数
    model_without_ddp = model.module if args2.distributed else model
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args2.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    start = time.time()
    best_miou = 0
    best_acc = 0
    best_f1 = 0
    best_bf1 = 0
    last_epoch = 0
    test_epoch = args2.end_epoch - 3
    ave_loss = AverageMeter()

    # 模型加载（也只在主进程中操作，然后广播）
    if args2.rank == 0:
        weight_save_dir = os.path.join(save_dir, save_name, 'weights')
        model_state_file = os.path.join(weight_save_dir, "{}_lr{}_epoch{}_batchsize{}_{}.pkl.tar".format(
            args2.models, args2.lr, args2.end_epoch, args2.train_batchsize * args2.world_size, args2.information
        ))
        if os.path.isfile(model_state_file):
            print('=> loading checkpoint')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=device, weights_only=False)
            best_miou = checkpoint.get('best_miou', 0)
            last_epoch = checkpoint.get('epoch', 0)
            model_without_ddp.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{model_state_file}' (epoch {last_epoch})")
    
    # 同步检查点信息
    if args2.distributed:
        dist.barrier() # 确保主进程加载完模型后，其他进程再继续
        # 你也可以选择广播 last_epoch 和 best_miou
        # last_epoch_tensor = torch.tensor(last_epoch, device=device)
        # dist.broadcast(last_epoch_tensor, src=0)
        # last_epoch = last_epoch_tensor.item()
        
    for epoch in range(last_epoch, args2.end_epoch):
        model.train()
        if args2.distributed:
            # 确保每个epoch的shuffle都不同
            dataloader_train.sampler.set_epoch(epoch)
        
        # 设置进程标题，方便监控
        if args2.rank == 0:
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
                                      args2.warm_epochs * lenth_iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0 and args2.rank == 0:
                print_loss = ave_loss.average()
                time_cost = time.time() - start
                start = time.time()
                print("epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, best_miou:{:.4f}".
                      format(epoch, args2.end_epoch, i, len(dataloader_train), print_loss, time_cost, lr, best_miou))
                logging.info("epoch:[{}/{}], iter:[{}/{}], loss:{:.4f}, time:{:.4f}, lr:{:.4f}, best_miou:{:.4f}".
                             format(epoch, args2.end_epoch, i, len(dataloader_train), print_loss, time_cost, lr, best_miou))
        
        # 验证只在一个epoch结束后进行
        miou, acc, f1, precision, recall, class_iou, bf1 = validate(dataloader_val, device, model, args2)

        if args2.rank == 0:
            print('Validation Result --> miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}, bf1:{:.4f}'.format(
                miou, acc, f1.mean(), precision.mean(), recall.mean(), bf1))
            print('Class IoU --> sea:{:.4f}, land:{:.4f}'.format(class_iou[0], class_iou[1]))
            logging.info('Validation Result --> miou:{:.4f}, acc:{:.4f}, f1:{:.4f}, bf1:{:.4f}'.format(miou, acc, f1.mean(), bf1))
            logging.info('Class IoU --> sea:{:.4f}, land:{:.4f}'.format(class_iou[0], class_iou[1]))

            if miou >= best_miou:
                best_miou = miou
                best_acc, best_f1, best_bf1 = acc, f1.mean(), bf1
                
                print(f"*** New Best mIoU: {best_miou:.4f} ***")
                torch.save(model_without_ddp.state_dict(), os.path.join(weight_save_dir, 'best_weight.pkl'))

            # 保存周期性检查点和最新检查点
            if (epoch % 10 == 0 and epoch > 50) or (epoch > test_epoch):
                torch.save(model_without_ddp.state_dict(),
                           os.path.join(weight_save_dir, f'epoch_{epoch}.pkl'))

            torch.save({
                'epoch': epoch + 1,
                'best_miou': best_miou,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_state_file)

    if args2.rank == 0:
        print("*************** Training Finished ***************")
        logging.info("*************** Final Summary ***************")
        logging.info("dataset:{} info:{} lr:{} epoch:{} batch:{} best_miou:{:.4f} best_acc:{:.4f} best_f1:{:.4f} best_bf1:{:.4f}".
                     format(args2.dataset, args2.information, args2.lr, args2.end_epoch, args2.train_batchsize * args2.world_size,
                            best_miou, best_acc, best_f1, best_bf1))


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, warmup_iter=None, power=0.9):
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
    nclass = 2
    metric = SegmentationMetric(nclass)
    total_bf1 = torch.zeros(1, device=device)

    with torch.no_grad():
        for i, sample in enumerate(dataloader_val):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            label = label.long().squeeze(1)
            
            # 模型输出
            logit = model(image, label, train=False)
            
            # 计算BF1
            bf1_score = metric.BoundaryF1(logit, label, device)
            total_bf1 += bf1_score

            # 计算其他指标
            pred = logit.argmax(dim=1)
            metric.addBatch(pred.cpu().numpy(), label.cpu().numpy())

    # --- 指标聚合 ---
    # 聚合混淆矩阵
    # print(dir(metric)) 
    confusion_matrix = torch.from_numpy(metric.confusionMatrix).to(device)
    if args2.distributed:
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
    
    # 聚合BF1分数
    # total_bf1已经是每个进程的累加和，我们只需要求所有进程的平均值
    if args2.distributed:
        dist.all_reduce(total_bf1, op=dist.ReduceOp.SUM)
        # total_bf1 累加的是每个 batch 的 BoundaryF1 均值，all_reduce 之后它是
        # 所有进程所有 batch 的均值之和，所以要除以所有进程的 batch 总数
        # (DistributedSampler 会把各进程补齐到相同长度)，而不是除以 world_size。
        # 原来除以 world_size 会让 bf1 远大于 1（例如 11.29）。
        total_bf1 = total_bf1 / (len(dataloader_val) * args2.world_size)
    else:
        # 单GPU模式下，dataloader的长度是正确的
        total_bf1 = total_bf1 / len(dataloader_val)

    # 只有主进程计算最终指标
    if args2.rank == 0:
        metric.confusionMatrix = confusion_matrix.cpu().numpy()
        miou = metric.meanIntersectionOverUnion()
        acc = metric.Accuracy()
        f1 = metric.F1()
        precision = metric.Precision()
        recall = metric.Recall()
        class_iou = metric.IntersectionOverUnion()

        return miou, acc, f1, precision, recall, class_iou, total_bf1.item()
    else:
        # 其他进程返回0值
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros(nclass), 0.0


def main():
    args = parse_args()
    
    # 如果指定了分布式训练，则初始化分布式环境
    if args.distributed:
        init_distributed_mode(args)

    cudnn.benchmark = True
    cudnn.enabled = True
    
    main_worker(args)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 在分布式启动脚本中设置，此处注释掉
    main()