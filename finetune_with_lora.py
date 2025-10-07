#!/usr/bin/env python3
"""
LoRAを使ったSarashina 0.5Bのファインチューニング
PEFTライブラリを使用してメモリ効率的にファインチューニングを実行
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
import json

def main():
    print("="*60)
    print("Sarashina 0.5B ファインチューニング (LoRA版)")
    print("="*60)

    # モデルとトークナイザーのロード
    model_name = "sbintuitions/sarashina2.2-0.5B-instruct-v0.1"

    print("\nステップ 1: モデルとトークナイザーのロード...")
    print(f"  Hugging Faceから {model_name} をダウンロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  ベースモデルロード完了: {model.num_parameters() / 1e9:.2f}B パラメータ")

    # LoRA設定
    print("\nステップ 2: LoRA設定...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,                          # LoRAのランク（低いほどメモリ効率的）
        lora_alpha=32,                # LoRAのスケーリング係数
        lora_dropout=0.1,             # ドロップアウト率
        target_modules=[              # LoRAを適用するモジュール
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # LoRAモデルの準備
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"  LoRA設定完了:")
    print(f"    - ランク (r): {lora_config.r}")
    print(f"    - Alpha: {lora_config.lora_alpha}")
    print(f"    - ドロップアウト: {lora_config.lora_dropout}")
    print(f"    - 学習可能パラメータ: {trainable_params:,} / {all_params:,}")
    print(f"    - 学習率: {100 * trainable_params / all_params:.2f}%")

    # データセットのロード
    print("\nステップ 3: データセットのロード...")
    dataset = load_dataset("json", data_files={
        "train": "/workspace/train.jsonl",
        "validation": "/workspace/valid.jsonl"
    })

    print(f"  訓練データ: {len(dataset['train'])} サンプル")
    print(f"  検証データ: {len(dataset['validation'])} サンプル")

    # データの前処理
    def preprocess_function(examples):
        # inputとoutputを結合
        texts = []
        for inp, out in zip(examples['input'], examples['output']):
            text = inp + out
            texts.append(text)

        # トークナイズ
        result = tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

        # ラベルの設定（入力と同じ）
        result["labels"] = [[label if label != tokenizer.pad_token_id else -100 for label in labels] for labels in result["input_ids"]]
        return result

    print("\nステップ 4: データの前処理...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="トークナイズ中",
    )

    # データコレクターの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LMなのでFalse
    )

    # トレーニング設定
    print("\nステップ 5: トレーニング設定...")
    training_args = TrainingArguments(
        output_dir="/workspace/nemo_experiments/sarashina_lora_finetune",
        num_train_epochs=3,
        per_device_train_batch_size=2,      # LoRAはメモリ効率的なので増やせる
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,      # 実質バッチサイズ: 8
        learning_rate=2e-4,                 # LoRAは高めの学習率が効果的
        warmup_steps=100,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",  # ロガーを無効化
        remove_unused_columns=False,
    )

    # Trainerの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("\nステップ 6: トレーニング開始...")
    print(f"  エポック数: 3")
    print(f"  バッチサイズ: 2 (勾配累積: 4 → 実質バッチサイズ: 8)")
    print(f"  学習率: 2e-4")
    print()

    # トレーニング実行
    trainer.train()

    # LoRAモデルの保存
    print("\nステップ 7: LoRAモデルの保存...")
    lora_output_dir = "/workspace/sarashina_0.5B_lora"
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)

    print("\n" + "="*60)
    print("トレーニング完了！")
    print(f"LoRAモデル保存先: {lora_output_dir}")
    print("\n使用方法:")
    print("from peft import PeftModel")
    print("from transformers import AutoModelForCausalLM")
    print(f"base_model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"model = PeftModel.from_pretrained(base_model, '{lora_output_dir}')")
    print("="*60)

if __name__ == "__main__":
    main()
