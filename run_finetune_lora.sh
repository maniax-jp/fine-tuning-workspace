#!/bin/bash

echo "=============================================="
echo "Sarashina 0.5B LoRAファインチューニング"
echo "=============================================="
echo ""

# 依存パッケージのインストール
echo "ステップ 1: 依存パッケージのインストール..."
pip install -q transformers datasets accelerate sentencepiece protobuf peft bitsandbytes

echo ""
echo "ステップ 2: トレーニング実行..."
echo ""

# トレーニング実行
python /workspace/finetune_with_lora.py

echo ""
echo "=============================================="
echo "完了！"
echo "=============================================="
