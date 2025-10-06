#!/bin/bash
# Transformersを使ったファインチューニング（依存関係インストール付き）

set -e

echo "====================================="
echo "依存パッケージのインストール"
echo "====================================="

pip install -q transformers datasets accelerate sentencepiece protobuf

echo ""
echo "====================================="
echo "ファインチューニング開始"
echo "====================================="
echo ""

python /workspace/finetune_with_transformers.py
