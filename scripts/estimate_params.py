#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确统计 Mini-LLM 模型的参数量（训练参数量 + 激活参数量）。

在正式训练前运行，确认 JSON 配置产生的参数量是否符合预期。

用法（项目根目录执行）：

  # 使用 JSON 配置（推荐）
  python scripts/estimate_params.py --config_json configs/v4_300m_2xa100.json

  # 使用默认 CLI 参数
  python scripts/estimate_params.py --model_name mini_deepseekv4 --max_seq_len 2048

输出包含：
  - exact:  每类参数的具体数值（total / trainable / non_trainable）
  - approx: 按模块分组的参数分布（embedding / attention / ffn / moe 等）

预期用途：
  1. 验证 JSON 配置的参数量级是否正确（如 300M 配置应接近 300M）
  2. 对比不同配置的参数量差异
  3. 确认 MoE 模型的 activated params 比例
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_models import get_model_and_config, get_model_info


def main():
    parser = argparse.ArgumentParser(description="Estimate Mini-LLM model parameters")
    parser.add_argument("--model_name", type=str, default="mini_deepseekv4", help="Model name from mini_models")
    parser.add_argument("--config_json", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length (affects positional encoding params)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "mini_tokenizer"))
    Model, Config = get_model_and_config(args.model_name)

    # 从 JSON 加载配置（如果提供）
    cfg_kwargs = {}
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg_kwargs = json.load(f)

    cfg_kwargs.update({
        "vocab_size": len(tokenizer),
        "use_cache": False,
        "max_position_embeddings": args.max_seq_len,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })

    config = Config(**cfg_kwargs)
    model = Model(config)

    exact, info = get_model_info(model)

    print("=" * 60)
    print(f"Model: {args.model_name}")
    if args.config_json:
        print(f"Config: {args.config_json}")
    print("=" * 60)

    print("\n[config]")
    print(config)
    print()

    print("[exact parameter counts]")
    print(json.dumps(exact, ensure_ascii=False, indent=2))
    print()

    print("[approximate breakdown]")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print()

    total_params = exact.get("total", 0)
    trainable_params = exact.get("trainable", 0)
    print("-" * 40)
    print(f"  Total params:      {total_params / 1e6:.2f}M ({total_params:,})")
    print(f"  Trainable params:  {trainable_params / 1e6:.2f}M ({trainable_params:,})")
    print(f"  Non-trainable:     {(total_params - trainable_params) / 1e6:.2f}M")

    if info.get("activated_params", 0):
        print(f"  Activated (MoE):   {info['activated_params'] / 1e6:.2f}M")
    if info.get("total_params_per_token", 0):
        print(f"  Params/token:      {info['total_params_per_token']:,}")

    # 释放模型
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("-" * 40)
    print("Done.")


if __name__ == "__main__":
    main()
