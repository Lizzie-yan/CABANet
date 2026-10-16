#!/usr/bin/env python
"""独立测试/评估脚本：加载训练好的权重，在 val 划分上评测并写出指标。

用法示例：
  python eval_model.py --models cmlformer --dataset SL1 \
      --checkpoint work_dir/CMLFormer/xxx/weights/best_weight.pkl \
      --out_dir work_dir/CMLFormer/xxx --gpu 0

输出:
  <out_dir>/metrics.json   机器可读
  <out_dir>/metrics.txt    人可读
  <out_dir>/predictions/   可选，前 N 张预测图（--save_preds N）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SL1, SLSD                      # noqa: E402
from seg_metric import SegmentationMetric          # noqa: E402
import train_distributed as TD                     # noqa: E402


def build_model(models, nclass=2):
    ns = argparse.Namespace()
    ns.models = models
    ns.head = 'seghead'
    ns.use_edge = 0
    ns.use_mixup = 0
    return TD.get_model(ns)


def load_state(model, ckpt):
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        obj = obj['state_dict']
    missing, unexpected = model.load_state_dict(obj, strict=False)
    return list(missing), list(unexpected)


def main():
    ap = argparse.ArgumentParser('evaluate a trained checkpoint')
    ap.add_argument('--models', required=True)
    ap.add_argument('--dataset', required=True, choices=['SL1', 'SLSD'])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--information', default='')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--save_preds', type=int, default=0)
    ap.add_argument('--crop_size', type=int, nargs='+', default=[512, 512])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda:%d' % args.gpu)

    print('=' * 90)
    print('model     : %s' % args.models)
    print('dataset   : %s' % args.dataset)
    print('checkpoint: %s' % args.checkpoint)
    print('device    : %s (%s)' % (device, torch.cuda.get_device_name(args.gpu)))
    print('=' * 90)

    model = build_model(args.models)
    missing, unexpected = load_state(model, args.checkpoint)
    print('load_state_dict: missing=%d unexpected=%d' % (len(missing), len(unexpected)))
    if missing:
        print('  missing[:5]   :', missing[:5])
    if unexpected:
        print('  unexpected[:5]:', unexpected[:5])
    model = model.to(device).eval()

    n_param = sum(p.numel() for p in model.parameters())
    print('params    : %.2f M' % (n_param / 1e6))

    if args.dataset == 'SL1':
        ds = SL1(train=False, dataset='SL1', crop_szie=args.crop_size)
    else:
        ds = SLSD(train=False, dataset='SLSD', crop_szie=args.crop_size)
    print('val images: %d' % len(ds))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print('batches   : %d' % len(loader))

    metric = SegmentationMetric(2)
    total_bf1 = torch.zeros(1, device=device)
    pred_dir = None
    if args.save_preds > 0:
        pred_dir = os.path.join(args.out_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

    t0 = time.time()
    seen = 0
    with torch.no_grad():
        for i, sample in enumerate(loader):
            image = sample['image'].to(device, non_blocking=True)
            label = sample['label'].to(device, non_blocking=True).long().squeeze(1)
            logit = model(image, label, train=False)

            total_bf1 += metric.BoundaryF1(logit, label, device)
            pred = logit.argmax(dim=1)
            metric.addBatch(pred.cpu().numpy(), label.cpu().numpy())
            seen += image.size(0)

            if pred_dir is not None and i < args.save_preds:
                names = sample.get('name', None)
                for b in range(pred.shape[0]):
                    nm = names[b] if names is not None else 'batch%d_%d' % (i, b)
                    np.save(os.path.join(pred_dir, '%s_pred.npy' % nm),
                            pred[b].cpu().numpy().astype(np.uint8))
                    np.save(os.path.join(pred_dir, '%s_gt.npy' % nm),
                            label[b].cpu().numpy().astype(np.uint8))

            if i % 20 == 0:
                print('  [%d/%d] miou_sofar=%.4f' %
                      (i, len(loader), metric.meanIntersectionOverUnion()))

    elapsed = time.time() - t0
    miou = float(metric.meanIntersectionOverUnion())
    acc = float(metric.Accuracy())
    f1 = metric.F1()
    precision = metric.Precision()
    recall = metric.Recall()
    class_iou = metric.IntersectionOverUnion()
    bf1 = float((total_bf1 / len(loader)).item())

    result = {
        'model': args.models,
        'dataset': args.dataset,
        'information': args.information,
        'checkpoint': os.path.abspath(args.checkpoint),
        'num_val_images': len(ds),
        'num_params_M': round(n_param / 1e6, 3),
        'miou': round(miou, 6),
        'acc': round(acc, 6),
        'f1': round(float(np.nanmean(f1)), 6),
        'precision': round(float(np.nanmean(precision)), 6),
        'recall': round(float(np.nanmean(recall)), 6),
        'boundary_f1': round(bf1, 6),
        'iou_sea': round(float(class_iou[0]), 6),
        'iou_land': round(float(class_iou[1]), 6),
        'f1_per_class': [round(float(x), 6) for x in f1],
        'eval_seconds': round(elapsed, 1),
        'evaluated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        '=' * 70,
        'Evaluation result',
        '=' * 70,
        'model        : %s' % args.models,
        'dataset      : %s' % args.dataset,
        'information  : %s' % args.information,
        'checkpoint   : %s' % result['checkpoint'],
        'val images   : %d' % result['num_val_images'],
        'params (M)   : %.3f' % result['num_params_M'],
        '-' * 70,
        'mIoU         : %.4f' % result['miou'],
        'Accuracy     : %.4f' % result['acc'],
        'F1           : %.4f' % result['f1'],
        'Precision    : %.4f' % result['precision'],
        'Recall       : %.4f' % result['recall'],
        'Boundary F1  : %.4f' % result['boundary_f1'],
        'IoU  sea     : %.4f' % result['iou_sea'],
        'IoU  land    : %.4f' % result['iou_land'],
        '-' * 70,
        'elapsed      : %.1f s' % result['eval_seconds'],
        'evaluated at : %s' % result['evaluated_at'],
        '=' * 70,
    ]
    with open(os.path.join(args.out_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print()
    print('\n'.join(lines))
    print('wrote: %s' % os.path.join(args.out_dir, 'metrics.json'))
    print('EVAL_DONE')


if __name__ == '__main__':
    main()
