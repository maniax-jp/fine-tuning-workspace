# Sarashina 2.2 0.5B Fine-tuning

日本語言語モデル「Sarashina 2.2 0.5B」のファインチューニングプロジェクト

## 📋 概要

このプロジェクトは、Hugging Face Transformersを使用してSarashina 2.2 0.5B Instructモデルをファインチューニングするためのスクリプトです。

- **ベースモデル**: [sbintuitions/sarashina2.2-0.5B-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5B-instruct-v0.1)
- **パラメータ数**: 0.79B
- **フレームワーク**: Hugging Face Transformers
- **精度**: BFloat16

## 🐳 使用するDockerイメージ

```bash
nvcr.io/nvidia/pytorch:25.01-py3
```

## 📦 必要な環境

- **GPU**: NVIDIA GPU（CUDA対応）
- **VRAM**: 最低16GB以上推奨（32GB推奨）
- **Docker**: NVIDIA Container Toolkit がインストール済み
- **ストレージ**: 約20GB以上の空き容量

## 🚀 起動方法

### 1. Dockerコンテナの起動

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.01-py3
```

**オプション説明:**
- `--gpus all`: 全てのGPUを使用
- `--shm-size=16g`: 共有メモリを16GBに設定
- `--ulimit memlock=-1`: メモリロック制限を解除
- `-v ./fine-tuning-workspace:/workspace`: ローカルディレクトリをマウント
- `-w /workspace`: 作業ディレクトリを設定

### 2. ファインチューニングの実行

#### 方法A: フル・ファインチューニング（全パラメータ）

コンテナ内で以下のコマンドを実行：

```bash
bash run_finetune_transformers.sh
```

または、手動で実行：

```bash
# 依存パッケージのインストール
pip install -q transformers datasets accelerate sentencepiece protobuf

# トレーニング実行
python finetune_with_transformers.py
```

#### 方法B: LoRAファインチューニング（推奨・メモリ効率的）

```bash
bash run_finetune_lora.sh
```

または、手動で実行：

```bash
# 依存パッケージのインストール
pip install -q transformers datasets accelerate sentencepiece protobuf peft bitsandbytes

# トレーニング実行
python finetune_with_lora.py
```

**LoRAの利点:**
- 💾 メモリ使用量が大幅に削減（約1/10のパラメータ）
- ⚡ 学習速度が高速
- 💰 計算コストが低い
- 🔄 複数のLoRAアダプターを切り替え可能

## 📁 ファイル構成

```
workspace/
├── README.md                          # このファイル
├── .gitignore                         # Git除外設定
├── finetune_with_transformers.py      # フル・ファインチューニング
├── finetune_with_lora.py              # LoRAファインチューニング（NEW!）
├── run_finetune_transformers.sh       # フル版実行スクリプト
├── run_finetune_lora.sh               # LoRA版実行スクリプト（NEW!）
├── inference_lora.py                  # LoRAモデル推論スクリプト（NEW!）
├── train.jsonl                        # 訓練データ（13,513サンプル）
├── valid.jsonl                        # 検証データ（1,502サンプル）
└── nemo_experiments/                  # 出力ディレクトリ（.gitignoreで除外）
    ├── sarashina_finetune_hf/         # フル・ファインチューニング出力
    │   ├── checkpoint-13500/          # ベストモデル（eval_loss: 1.962）
    │   ├── checkpoint-20200/
    │   └── checkpoint-20271/          # 最終チェックポイント
    └── sarashina_lora_finetune/       # LoRAファインチューニング出力
```

## 📊 データフォーマット

訓練・検証データは以下のJSON Lines形式：

```json
{"input": "指示文やプロンプト", "output": "期待される応答", "source": "データソース名"}
```

## ⚙️ トレーニング設定

### フル・ファインチューニング

```python
- エポック数: 3
- バッチサイズ: 1
- 勾配累積ステップ: 2（実質バッチサイズ: 2）
- 学習率: 1e-5
- 最大トークン長: 2048
- 精度: BFloat16
- 評価間隔: 100ステップ
- 保存間隔: 100ステップ
```

### LoRAファインチューニング

```python
- エポック数: 3
- バッチサイズ: 2
- 勾配累積ステップ: 4（実質バッチサイズ: 8）
- 学習率: 2e-4（LoRAは高めの学習率が効果的）
- LoRAランク (r): 8
- LoRA Alpha: 32
- LoRAドロップアウト: 0.1
- 対象モジュール: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 学習可能パラメータ: 約10%
- 最大トークン長: 2048
- 精度: BFloat16
```

## 📈 トレーニング結果

### フル・ファインチューニング

- **訓練損失**: 3.93 → 0.85
- **検証損失**: 2.33 → 1.98
- **総ステップ数**: 20,271
- **所要時間**: 約3時間20分
- **ベストモデル**: checkpoint-13500（eval_loss: 1.962）

### LoRAファインチューニング

- **訓練損失**: 平均 2.09
- **総ステップ数**: 5,070
- **所要時間**: 約2時間28分（8,885秒）
- **処理速度**:
  - 4.56 サンプル/秒
  - 0.57 ステップ/秒
- **エポック数**: 3
- **学習データ**: 13,513サンプル × 3エポック = 40,539回の学習

## 🔧 カスタマイズ

### 学習率の変更

`finetune_with_transformers.py`の`TrainingArguments`内で設定：

```python
learning_rate=1e-5  # お好みの値に変更
```

### エポック数の変更

```python
num_train_epochs=3  # お好みの値に変更
```

### バッチサイズの調整

```python
per_device_train_batch_size=1  # VRAM容量に応じて調整
gradient_accumulation_steps=2   # 実質的なバッチサイズを増やす
```

## 💾 モデルの使用方法

### フル・ファインチューニングモデルの使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# ベストモデルを読み込み
model_path = "./nemo_experiments/sarashina_finetune_hf/checkpoint-13500"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 推論
inputs = tokenizer("こんにちは、", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### LoRAモデルの使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ベースモデルとトークナイザーの読み込み
base_model_path = "sbintuitions/sarashina2.2-0.5B-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRAアダプターの適用
lora_path = "./sarashina_0.5B_lora"
model = PeftModel.from_pretrained(base_model, lora_path)

# 推論
inputs = tokenizer("こんにちは、", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

または、推論スクリプトを使用：

```bash
python inference_lora.py
```

## 📝 ライセンス

このプロジェクトで使用しているSarashina 2.2モデルのライセンスについては、[元のモデルページ](https://huggingface.co/sbintuitions/sarashina2.2-0.5B-instruct-v0.1)を参照してください。

## 🙏 謝辞

- ベースモデル: [sbintuitions/sarashina2.2-0.5B-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5B-instruct-v0.1)
- フレームワーク: Hugging Face Transformers
- Docker環境: NVIDIA PyTorch Container

## 📞 サポート

問題が発生した場合は、Issueを作成してください。
