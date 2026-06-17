#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单 batch CUDA sanity check — 在正式训练前验证 GPU 环境、模型前向/反向、混合精度数值稳定性。

用法（项目根目录执行）：

  # 使用 JSON 配置（推荐）
  python scripts/sanity_check.py --config_json configs/v4_300m_2xa100.json

  # 使用默认 CLI 参数
  python scripts/sanity_check.py --model_name mini_deepseekv4 --max_seq_len 512

预期输出：
  [model info] ...       # 模型参数量等信息
  forward OK             # 前向传播成功
  backward OK            # 反向传播成功
  loss = 10.xxx          # 初始 loss（未训练）≈ ln(vocab_size)
  sanity ok              # 全部通过

失败信号：
  - CUDA out of memory   # 显存不足，需要减小 batch_size 或 max_seq_len
  - NaN/Inf in loss      # 数值不稳定，检查精度设置或学习率
  - import error         # 环境问题，检查依赖安装
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.amp import autocast
from transformers import AutoTokenizer

# 将项目根目录加入 sys.path，确保能导入 mini_models
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_models import get_model_and_config, get_model_info


def main():
    parser = argparse.ArgumentParser(description="Single-batch CUDA sanity check for Mini-LLM")
    parser.add_argument("--model_name", type=str, default="mini_deepseekv4", help="Model name from mini_models")
    parser.add_argument("--config_json", type=str, default=None, help="Path to JSON config (optional, overrides model defaults)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Sequence length for the test batch")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the test (1 is safest)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Precision for the test. Should match your training config.")
    args = parser.parse_args()

    # ---- 环境检查 ----
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available. GPU is required for training.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"[env] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[env] CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- 加载模型 ----
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "mini_tokenizer"))
    Model, Config = get_model_and_config(args.model_name)

    # 从 JSON 加载配置（如果提供）
    cfg_kwargs = {}
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg_kwargs = json.load(f)
        print(f"[config] loaded from {args.config_json}")

    # 训练脚本对以下参数有最终决定权
    cfg_kwargs.update({
        "vocab_size": len(tokenizer),
        "use_cache": False,
        "max_position_embeddings": args.max_seq_len,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })

    config = Config(**cfg_kwargs)
    model = Model(config).to(device)
    model.train()

    # 打印模型信息
    _, info = get_model_info(model)
    print(f"[model] {json.dumps(info, ensure_ascii=False, indent=2)}")

    # ---- 生成随机输入（模拟一个 batch） ----
    input_ids = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(args.batch_size, args.max_seq_len),
        device=device,
        dtype=torch.long,
    )

    # ---- 前向传播 ----
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.precision]
    enable_amp = args.precision != "fp32"

    try:
        with autocast(device_type="cuda", enabled=enable_amp, dtype=amp_dtype):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
        print("[OK] forward")
    except Exception as e:
        print(f"[FAIL] forward: {e}")
        sys.exit(1)

    # ---- 反向传播 ----
    try:
        loss.backward()
        print("[OK] backward")
    except Exception as e:
        print(f"[FAIL] backward: {e}")
        sys.exit(1)

    # ---- 数值检查 ----
    loss_val = float(loss.detach().cpu())
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[FAIL] loss is NaN or Inf (loss={loss_val})")
        sys.exit(1)

    # 初始 loss 应 ≈ ln(vocab_size) ≈ ln(32004) ≈ 10.37
    expected_ln = math.log(len(tokenizer))
    print(f"[result] loss = {loss_val:.4f} (expected ~{expected_ln:.2f} = ln(vocab_size) for untrained model)")

    # ---- 清理 ----
    del model, input_ids
    torch.cuda.empty_cache()

    print("[PASS] sanity ok")
    print()
    print("提示：接下来可以运行正式训练。如果是多卡训练，建议再验证 DDP 通信：")
    print("  torchrun --nproc_per_node=2 scripts/sanity_check.py --config_json <config>")


if __name__ == "__main__":
    main()
