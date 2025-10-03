# LLM ファインチューニング

sarashina2.2-0.5B-instruct-v0.1モデルを使用したファインチューニングプログラムです。

## 概要

このプロジェクトは、日本語の対話データセット（databricks-dolly-15k-ja-zundamon）を使用して、言語モデルをファインチューニングします。

## 必要環境

- Python 3.8以上
- CUDA対応GPU（推奨）
- 8GB以上のVRAM（GPU使用時）

## セットアップ

1. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

## 使用方法

### ファインチューニングの実行

```bash
python train.py
```

### 設定のカスタマイズ

[train.py](train.py)の冒頭で以下の設定を変更できます:

```python
MODEL_NAME = "sbintuitions/sarashina2.2-0.5B-instruct-v0.1"  # ベースモデル
DATASET_PATH = "dataset/databricks-dolly-15k-ja-zundamon.json"  # データセットパス
OUTPUT_DIR = "output"  # 出力ディレクトリ
TRAIN_SPLIT = 0.9  # 訓練データの割合（0.9 = 90%）
```

## データセット形式

データセットは以下のJSON形式である必要があります:

```json
[
    {
        "instruction": "質問文",
        "input": "補足情報（オプション）",
        "output": "回答文",
        "category": "カテゴリ",
        "index": "インデックス"
    }
]
```

## 学習パラメータ

デフォルトの学習パラメータ（[train.py:99-113](train.py#L99-L113)）:

- **エポック数**: 3
- **バッチサイズ**: 4
- **勾配累積ステップ**: 4（実効バッチサイズ = 16）
- **学習率**: 2e-5
- **ウォームアップステップ**: 100
- **最大トークン長**: 512
- **精度**: BF16（GPU使用時）

## 出力

学習済みモデルは以下のディレクトリに保存されます:

```
output/
├── final_model/          # 最終モデル
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── checkpoint-500/       # 中間チェックポイント
└── logs/                 # 学習ログ
```

## プロンプト形式

学習データは以下の形式でフォーマットされます:

**入力情報がある場合:**
```
### 指示:
{instruction}

### 入力:
{input}

### 応答:
{output}
```

**入力情報がない場合:**
```
### 指示:
{instruction}

### 応答:
{output}
```

## トラブルシューティング

### メモリ不足エラー

バッチサイズを減らしてください:
```python
per_device_train_batch_size=2  # 4から2に変更
```

### CUDA out of memoryエラー

以下のいずれかを試してください:
- バッチサイズを減らす
- `max_length`を512から256に変更
- 勾配チェックポイントを有効化

### FP16エラー

現在の設定ではBF16を使用しています。GPU がBF16に対応していない場合は、[train.py:109](train.py#L109)を以下に変更:
```python
bf16=False,
fp16=False,  # FP32を使用
```

## ライセンス

使用するモデルとデータセットのライセンスに従ってください。
