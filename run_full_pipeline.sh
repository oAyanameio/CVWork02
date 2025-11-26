#!/bin/bash

# 使用 GPU 2（可修改）
export CUDA_VISIBLE_DEVICES=2

# 日志文件（可修改）
LOGFILE="full_pipeline.log"

echo "======================================="
echo " Launching Full Pipeline (VOC + Faster R-CNN)"
echo " Using GPU: $CUDA_VISIBLE_DEVICES"
echo " Log File: $LOGFILE"
echo "======================================="

# 后台运行完整流程：GT 可视化 → 训练 → mAP 验证 → 难例分析
nohup python -u assignment2_full.py \
    --mode full_pipeline \
    --samples_per_class 3 \
    --vis_n 50 \
    --save_n 100 \
    > "$LOGFILE" 2>&1 &

echo "Started! Monitor log with:"
echo "  tail -f $LOGFILE"
echo "======================================="
