#!/bin/bash

# --- 新增：在这里指定你要使用的GPU物理序号 ---
# 例如，使用物理GPU 2, 3, 4, 5, 6
export CUDA_VISIBLE_DEVICES="4,5"

# 把 torch/torchvision 预训练权重的下载缓存限制在本项目目录内
# (默认会写到 ~/.cache/torch，这里显式改到 CABA 目录下)
export TORCH_HOME="/home/data/disk0/ningfan/CABA/.torch_home"

# 计算你指定的GPU数量，并覆盖原来的GPUS_PER_NODE
# 这使得脚本更加健壮，避免手动设置的数量和CUDA_VISIBLE_DEVICES不匹配
# 注意：这个计算方法要求序号是连续的逗号分隔列表
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# 切到脚本所在目录，保证 ./models、./work_dir 等相对路径正确
cd "$(dirname "$(readlink -f "$0")")"

# 你的Python脚本文件名
SCRIPT_NAME="train_distributed.py" 

# 随机化端口号，避免端口冲突
MASTER_PORT=$(shuf -i 29500-29600 -n 1)

echo "将使用以下物理GPU: ${CUDA_VISIBLE_DEVICES}"
echo "总共 ${GPUS_PER_NODE} 个GPU"
echo "启动端口: ${MASTER_PORT}"

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT_NAME} \
    --distributed \
    --dataset SLSD\
    --models dbbanet \
    --end_epoch 200 \
    --lr 0.0003 \
    --train_batchsize 8 \
    --information "25SLSD"

echo "训练脚本执行完毕。"








