import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path

# 設定
MODEL_NAME = "sbintuitions/sarashina2.2-0.5B-instruct-v0.1"
DATASET_PATH = "dataset/databricks-dolly-15k-ja-zundamon.json"
OUTPUT_DIR = "output"
TRAIN_SPLIT = 0.9

def load_dataset_from_json(file_path, train_split=0.9):
    """JSONファイルからデータセットを読み込み、訓練用データを抽出"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 上から90%を訓練データとして使用
    train_size = int(len(data) * train_split)
    train_data = data[:train_size]

    print(f"Total data: {len(data)}")
    print(f"Training data: {len(train_data)}")

    return train_data

def format_prompt(sample):
    """データをプロンプト形式に整形"""
    instruction = sample['instruction']
    input_text = sample['input']
    output = sample['output']

    if input_text:
        prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 応答:\n{output}"
    else:
        prompt = f"### 指示:\n{instruction}\n\n### 応答:\n{output}"

    return prompt

def preprocess_data(data, tokenizer, max_length=512):
    """データの前処理とトークナイゼーション"""
    formatted_texts = [format_prompt(sample) for sample in data]

    # トークナイズ
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )

    return Dataset.from_dict(encodings)

def main():
    print("=" * 50)
    print("LLM Fine-tuning Script")
    print("=" * 50)

    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルとトークナイザーの読み込み
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # データセットの読み込み
    print(f"\nLoading dataset: {DATASET_PATH}")
    train_data = load_dataset_from_json(DATASET_PATH, TRAIN_SPLIT)

    # データの前処理
    print("\nPreprocessing data...")
    train_dataset = preprocess_data(train_data, tokenizer)

    # データコレーターの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果言語モデリング
    )

    # 学習パラメータの設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=device == "cuda",  # bf16に変更
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # ファインチューニング開始
    print("\n" + "=" * 50)
    print("Starting fine-tuning...")
    print("=" * 50)
    trainer.train()

    # モデルの保存
    print("\nSaving fine-tuned model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    print("\n" + "=" * 50)
    print("Fine-tuning completed!")
    print(f"Model saved to: {OUTPUT_DIR}/final_model")
    print("=" * 50)

if __name__ == "__main__":
    main()
