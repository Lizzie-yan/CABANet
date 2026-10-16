#!/usr/bin/env python
"""汇总 work_dir 下所有实验的指标，输出对比表（Markdown + CSV）。

产出三份文件（都在 --out_dir 下）：
  comparison_table.md/csv     全部实验（含消融）
  comparison_methods.md/csv   只含「对比方法 + 本文方法」，可直接放进论文表格

两类数据来源，均递归扫描（新实验的 train.log 在 work_dir/CMLFormer/xxx/ 里）：
  1. <exp>/metrics.json —— eval_model.py 写出，字段最全（含参数量、BoundaryF1）
  2. <exp>/train.log    —— 训练日志结尾的 summary 行
       gen A: dataset:X information:TAG lr:L epoch:E batchsize:B best_miou:.. best_acc:.. best_f1:.. best_bf1:..
       gen B: dataset:X info:TAG     lr:L epoch:E batch: N   best_miou:.. best_acc:.. best_f1:.. best_bf1:..

用法:
  python collect_metrics.py                    # 打印并在 work_dir 下写出表格
  python collect_metrics.py --out_dir /tmp     # 指定输出目录
"""
import argparse
import csv
import glob
import json
import os
import re
import time

WORK = '/home/data/disk0/ningfan/CABA/WORK1'
BASE = os.path.join(WORK, 'work_dir')

SUMA = re.compile(r'dataset:(\S+)\s+information:(\S+)\s+lr:(\S+)\s+epoch:(\d+)\s+'
                  r'batchsize:(\d+)\s+best_miou:([\d.]+)\s+best_acc:([\d.]+)\s+'
                  r'best_f1:([\d.]+)\s+best_bf1:([-\d.eE]+)')
SUMB = re.compile(r'dataset:(\S+)\s+info:(\S+)\s+lr:(\S+)\s+epoch:(\d+)\s+'
                  r'batch:(\d+)\s+best_miou:([\d.]+)\s+best_acc:([\d.]+)\s+'
                  r'best_f1:([\d.]+)\s+best_bf1:([-\d.eE]+)')
NAME_RE = re.compile(r'_lr[\d.]+_epoch(\d+)_batchsize(\d+)_')

COLS = ['method', 'dataset', 'epochs', 'batch', 'mIoU', 'acc', 'f1',
        'boundary_f1', 'params_M', 'source', 'run_dir']

SKIP = ('_smoke', '__pycache__', 'mmseg_shim', 'pretrained', '.git')

NEW_RUNS = ('cmlformer', 'mcatunet')

# 论文对比表要用的方法集合
COMPARISON = {'CMLFormer', 'MCAT-UNet', 'CGGLNet', 'DCNet', 'ACUNet',
              'DBENet', 'HED-UNet'}
PROPOSED = {'DBBANet'}


def recipe_from_name(name):
    m = NAME_RE.search(name)
    return (m.group(1), m.group(2)) if m else ('?', '?')


def guess_method(name):
    low = name.lower()
    for key, label in [('cmlformer', 'CMLFormer'), ('mcatunet', 'MCAT-UNet'),
                       ('dbbanet', 'DBBANet'), ('cgglnet', 'CGGLNet'),
                       ('cglnet', 'CGGLNet'), ('dcnet', 'DCNet'),
                       ('acunet', 'ACUNet'), ('dbenet', 'DBENet'),
                       ('hed_unet', 'HED-UNet'), ('bisenetv2', 'BiSeNetV2'),
                       ('deeplabv3', 'DeepLabV3'), ('pspnet', 'PSPNet'),
                       ('danet', 'DANet'), ('swint', 'Swin-T'), ('rest', 'ResT'),
                       ('bs_', 'BSModel')]:
        if key in low:
            return label
    return name.split('_lr')[0]


def from_metrics_json(path, exp):
    d = json.load(open(path, encoding='utf-8'))
    ep, bs = recipe_from_name(exp)
    return {
        'method': guess_method(exp),
        'dataset': d.get('dataset', '?'),
        'epochs': ep,
        'batch': bs,
        'mIoU': d.get('miou'),
        'acc': d.get('acc'),
        'f1': d.get('f1'),
        'boundary_f1': d.get('boundary_f1'),
        'params_M': d.get('num_params_M'),
        'source': 'metrics.json',
        'run_dir': exp,
    }


def from_train_log(path, exp):
    txt = open(path, encoding='utf-8', errors='replace').read()
    ms = [m for m in (SUMA.search(l) or SUMB.search(l) for l in txt.split('\n')) if m]
    if not ms:
        return None
    m = ms[0]
    if float(m.group(7)) == 0.0 and len(ms) > 1:
        m = ms[-1]          # 续跑时首条 summary 的 acc/f1 会是 0
    return {
        'method': guess_method(exp),
        'dataset': m.group(1),
        'epochs': m.group(4),
        'batch': m.group(5),
        'mIoU': float(m.group(6)),
        'acc': float(m.group(7)),
        'f1': float(m.group(8)),
        'boundary_f1': float(m.group(9)),
        'params_M': None,
        'source': 'train.log',
        'run_dir': exp,
    }


def ok(path):
    return not any(s in path for s in SKIP)


def gpu_note(r):
    if not any(k in r['run_dir'].lower() for k in NEW_RUNS):
        return ''
    try:
        return ' [%d卡DDP]' % (int(r['batch']) // 2)
    except (TypeError, ValueError):
        return ''


def fmt(v, nd=4):
    return ('%.' + str(nd) + 'f') % v if v is not None else '-'


def write_pair(out_dir, stem, rows, title, header_note):
    csv_path = os.path.join(out_dir, stem + '.csv')
    md_path = os.path.join(out_dir, stem + '.md')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# %s\n\n生成时间: %s\n\n' % (title, time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write(header_note)
        f.write('\n| 方法 | 数据集 | 轮数 | 总batch | mIoU | Accuracy | F1 | BoundaryF1 | 参数量(M) |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            f.write('| %s%s | %s | %s | %s | %s | %s | %s | %s | %s |\n' % (
                r['method'], gpu_note(r), r['dataset'], r['epochs'], r['batch'],
                fmt(r['mIoU']), fmt(r['acc']), fmt(r['f1']),
                fmt(r['boundary_f1']), fmt(r['params_M'], 3)))
    return csv_path, md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=BASE)
    args = ap.parse_args()

    rows, seen = [], set()

    for p in sorted(glob.glob(os.path.join(BASE, '**', 'metrics.json'), recursive=True)):
        if not ok(p):
            continue
        exp = os.path.relpath(os.path.dirname(p), BASE)
        try:
            rows.append(from_metrics_json(p, exp))
            seen.add(exp)
        except Exception as e:
            print('  跳过 %s: %s' % (p, e))

    for p in sorted(glob.glob(os.path.join(BASE, '**', 'train.log'), recursive=True)):
        if not ok(p):
            continue
        exp = os.path.relpath(os.path.dirname(p), BASE)
        if exp in seen:
            continue
        r = from_train_log(p, exp)
        if r:
            rows.append(r)

    rows.sort(key=lambda r: (r['dataset'] or '', -(r['mIoU'] or 0)))

    print('=' * 130)
    print('%-56s %-6s %6s %6s %8s %8s %8s %8s %8s  %s' %
          ('run_dir', 'dataset', 'ep', 'bs', 'mIoU', 'acc', 'f1', 'bf1', 'params', 'src'))
    print('=' * 130)
    for r in rows:
        print('%-56s %-6s %6s %6s %8s %8s %8s %8s %8s  %s' % (
            r['run_dir'][:56], r['dataset'], r['epochs'], r['batch'],
            fmt(r['mIoU']), fmt(r['acc']), fmt(r['f1']), fmt(r['boundary_f1']),
            fmt(r['params_M'], 2), r['source']))
    print('=' * 130)
    print('共 %d 条记录' % len(rows))

    for ds in sorted({r['dataset'] for r in rows}):
        sub = [r for r in rows if r['dataset'] == ds]
        print()
        print('### %s 排行（%d 条）' % (ds, len(sub)))
        for i, r in enumerate(sub, 1):
            print('  %2d. %-56s mIoU=%s  f1=%s  bs=%s%s'
                  % (i, r['run_dir'][:56], fmt(r['mIoU']), fmt(r['f1']),
                     r['batch'], gpu_note(r)))

    note = ('> **训练配方**：`bs` 为**总 batch size**。旧实验（CGGLNet/DCNet/ACUNet/'
            'DBENet/HED-UNet/bs 消融）为**单卡 batchsize2**；\n'
            '> 新接入的 CMLFormer / MCAT-UNet 按用户指定用多卡 DDP（每卡 batch 2），'
            '总 batch 更大，横向比较请留意。\n'
            '>\n'
            '> **BoundaryF1 可用性**：`train_distributed.py` 早先的 BF1 聚合有 bug'
            '（除以 world_size 而非批次数），导致多卡实验的 bf1 被放大约 40 倍'
            '（出现 8~14 这类值）。\n'
            '> 该 bug 已修复，**CMLFormer / MCAT-UNet 的 BoundaryF1 有效（0~1）**；'
            '此前用多卡跑的旧实验该列**不可用**（mIoU / Accuracy / F1 不受影响，仍然有效），\n'
            '> 表格中凡是 `BoundaryF1 > 1` 的行都属于这一类，请勿直接引用。\n')

    p1 = write_pair(args.out_dir, 'comparison_table', rows,
                    'work_dir 全部实验结果汇总', note)

    comp = [r for r in rows if r['method'] in COMPARISON or r['method'] in PROPOSED]
    comp.sort(key=lambda r: (r['dataset'] or '', -(r['mIoU'] or 0)))
    p2 = write_pair(args.out_dir, 'comparison_methods', comp,
                    '对比方法结果表（论文用）',
                    note + '> 本表只含外部对比方法 + 本文方法 DBBANet，不含消融实验。\n')

    if comp:
        print()
        print('=' * 130)
        print('论文用对比表（对比方法 + DBBANet）')
        print('=' * 130)
        print('%-16s %-6s %6s %6s %9s %9s %9s' %
              ('method', 'dataset', 'ep', 'bs', 'mIoU', 'f1', 'bf1'))
        print('-' * 130)
        for r in comp:
            print('%-16s %-6s %6s %6s %9s %9s %9s' % (
                r['method'], r['dataset'], r['epochs'], r['batch'],
                fmt(r['mIoU']), fmt(r['f1']), fmt(r['boundary_f1'])))
        print('=' * 130)

    print()
    print('已写出:')
    for p in p1 + p2:
        print('  %s' % p)
    print('COLLECT_DONE')


if __name__ == '__main__':
    main()
