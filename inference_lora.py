#!/usr/bin/env python3
"""
LoRAファインチューニング済みモデルでの推論スクリプト
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("="*60)
    print("LoRAモデルでの推論")
    print("="*60)

    # ベースモデル名
    base_model_name = "sbintuitions/sarashina2.2-0.5B-instruct-v0.1"

    # LoRAアダプターのパス
    lora_path = "/workspace/sarashina_0.5B_lora"

    print("\nステップ 1: ベースモデルのロード...")
    print(f"  Hugging Faceから {base_model_name} をダウンロード中...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("ステップ 2: LoRAアダプターの適用...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    print("\n" + "="*60)
    print("推論の準備が完了しました！")
    print("="*60)

    # 推論例
    prompts = [
        "日本の首都は",
        "人工知能とは",
        "富士山について教えて",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- 例 {i} ---")
        print(f"入力: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"出力: {generated_text}")
        print()

    print("="*60)
    print("推論完了！")
    print("="*60)

if __name__ == "__main__":
    main()
