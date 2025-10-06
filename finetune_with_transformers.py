#!/usr/bin/env python3
"""
Hugging Face Transformersを使ったSarashina 0.5Bのファインチューニング
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import json

def main():
    print("="*60)
    print("Sarashina 0.5B ファインチューニング (Transformers版)")
    print("="*60)

    # モデルとトークナイザーのロード
    model_path = "/root/.cache/huggingface/hub/models--sbintuitions--sarashina2.2-0.5B-instruct-v0.1/snapshots/e4b9aacc3f644893d0179847946ef6c58d868f29"

    print("\nステップ 1: モデルとトークナイザーのロード...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  モデルロード完了: {model.num_parameters() / 1e9:.2f}B パラメータ")

    # データセットのロード
    print("\nステップ 2: データセットのロード...")
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
            padding=False,
        )

        # ラベルの設定（入力と同じ）
        result["labels"] = result["input_ids"].copy()
        return result

    print("\nステップ 3: データの前処理...")
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
    print("\nステップ 4: トレーニング設定...")
    training_args = TrainingArguments(
        output_dir="/workspace/nemo_experiments/sarashina_finetune_hf",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
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

    print("\nステップ 5: トレーニング開始...")
    print(f"  エポック数: 3")
    print(f"  バッチサイズ: 1 (勾配累積: 2)")
    print(f"  学習率: 1e-5")
    print()

    # トレーニング実行
    trainer.train()

    # モデルの保存
    print("\nステップ 6: モデルの保存...")
    output_dir = "/workspace/sarashina_0.5B_finetuned"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "="*60)
    print("トレーニング完了！")
    print(f"モデル保存先: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
