# Mini-DeepLLM — DeepSeek V4 迷你复现

<p align="center">
  <strong>从零实现的 DeepSeek-V4 架构，面向单卡 A100 80GB 优化，200M 参数量</strong>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.56.1-ff69b4)

</div>

---

## 一、项目简介

本项目是 DeepSeek-V4 架构的迷你复现版本。核心特点：

- **架构完整** — 完整实现 DeepSeek-V4 的核心组件：Multi-Head Latent Attention (MLA)、Compressed Sparse Attention (CSA)、Heavily Compressed Attention (HCA)、MoE with shared experts、Multi-Token Prediction (MTP)
- **单卡 A100 优化** — 200M 参数配置，适配 A100 80GB 单卡训练
- **完全 transformers 兼容** — 支持 `from_pretrained` / `save_pretrained` / `generate`
- **数据管线完整** — 从 tokenizer 训练到数据下载、预处理、预训练全套脚本

### 文件结构

```
Mini-DeepLLM/
├── mini_models/
│   └── mini_deepseekv4/       ← DeepSeek V4 模型定义
│       ├── configuration_mini_deepseekv4.py  ← 配置类
│       └── modeling_mini_deepseekv4.py       ← 模型实现
│   ├── attention/             ← 注意力机制（MLA / CSA / HCA / Flash Attn）
│   ├── ffn/                   ← FFN / SwiGLU / MoE
│   ├── norm.py / rope.py / cache.py / generator.py
│   └── __init__.py            ← transformers AutoModel 注册
├── mini_tokenizer/            ← 预训练 tokenizer（词表 32004）
├── data_loader/               ← 预训练数据加载器
├── train/
│   ├── pretrain.py            ← 预训练脚本
│   └── utils.py               ← 训练工具函数
├── scripts/                   ← 环境配置、数据下载、预处理
├── configs/
│   ├── v4_200m_single_a100.json   ← 200M 配置（推荐）
│   └── v4_500m_single_a100.json   ← 500M 配置
└── data/tokenizer_data/       ← tokenizer 数据
```

---

## 二、模型架构

### DeepSeek-V4 核心组件

| 组件 | 说明 |
|------|------|
| **MLA** (Multi-Head Latent Attention) | 低秩压缩注意力，大幅降低 KV Cache 占用 |
| **CSA / HCA** (Compressed / Heavily Compressed Attention) | 分层压缩注意力机制，按 compress_ratio 将连续的 KV 压缩为单个 entry |
| **Sliding Window Attention** | 前 3 层 + 最后 1 层使用滑动窗口，捕获局部上下文 |
| **MoE + Shared Experts** | 路由专家 + 共享专家组合，提升参数效率 |
| **NoAux Load Balance** | 无辅助损失的路由负载均衡（通过动态偏置调整） |
| **MTP** (Multi-Token Prediction) | 多 token 预测辅助训练头，预训练时启用，推理时移除 |

### 200M 配置详情

```json
{
  "hidden_size": 768,
  "num_hidden_layers": 10,
  "num_attention_heads": 12,
  "head_dim": 64,
  "q_lora_rank": 192,
  "o_lora_rank": 96,
  "moe_intermediate_size": 768,
  "n_routed_experts": 8,
  "n_activated_experts": 2,
  "n_shared_experts": 1,
  "use_mtp": true
}
```

| 指标 | 值 |
|------|-----|
| 总参数量 | ~197M（含 MTP） |
| 推理参数量 | ~193M（去掉 MTP） |
| 每 token 激活参数量 | ~62M（top-2 专家 + 共享专家） |

### 各层注意力类型分布（10 层）

| 层索引 | 注意力类型 | 说明 |
|--------|----------|------|
| 0-2 | Sliding Attention | 滑动窗口，窗口大小 128 |
| 3 | Compressed Sparse Attention | CSA，compress_ratio=4 |
| 4 | Heavily Compressed Attention | HCA，compress_ratio=128 |
| 5-7 | CSA / HCA 交替 | 交替压缩注意力 |
| 9 | Sliding Attention | 末层滑动窗口 |

---

## 三、环境配置

### 3.1 克隆项目

```shell
git clone https://github.com/rokyky/Mini-DeepLLM.git
cd Mini-DeepLLM
```

### 3.2 初始化环境

```shell
# Linux
bash ./scripts/setup.sh

# Windows
.\scripts\setup.ps1
```

### 3.3 下载数据

```shell
bash ./scripts/download_data.sh
```

建议选择：
- **[4]** 20% 采样的 Fineweb-Edu-Chinese-V2.1 `.bin` 格式数据
- **[6]** 匠数科技大模型数据集 `.bin` 格式数据

---

## 四、训练

### 4.1 参数量估算

```shell
python scripts/estimate_params.py \
  --model_name mini_deepseekv4 \
  --config_json configs/v4_200m_single_a100.json \
  --max_seq_len 2048
```

### 4.2 Sanity Check

```shell
python scripts/sanity_check.py \
  --model_name mini_deepseekv4 \
  --config_json configs/v4_200m_single_a100.json \
  --max_seq_len 512 \
  --batch_size 1 \
  --precision bf16
```

### 4.3 正式训练

```shell
python ./train/pretrain.py \
  --model_name mini_deepseekv4 \
  --config_json configs/v4_200m_single_a100.json \
  --max_seq_len 2048 \
  --max_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --precision bf16 \
  --target_tokens 6_000_000_000 \
  --log_interval 20 \
  --save_interval 1000
```

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--max_batch_size` | 32 | 单卡 A100 80GB 可容纳 32 条 2048 序列 |
| `--gradient_accumulation_steps` | 2 | 等效 batch = 32 × 2 = 64 条 |
| `--target_tokens` | 6_000_000_000 | 训练 6B token 后自动停止 |
| `--precision` | bf16 | bfloat16 混合精度，A100 原生支持 |
| `--save_interval` | 1000 | 每 1000 step 保存一次 checkpoint |

### 4.4 训练时间估算

以下估算基于单卡 **A100 80GB**，bf16 混合精度：

| 配置 | batch | seq_len | 等效 batch | tokens/sec | 1B tokens | 6B tokens | 20B tokens |
|------|-------|---------|-----------|-----------|-----------|-----------|-----------|
| 200M (experts=8, top_k=2) | 32 | 2048 | 64 | ~80-100K | **~3 小时** | **~18 小时** | **~3 天** |
| 200M (experts=8, top_k=2) | 64 | 1024 | 128 | ~120-150K | **~2 小时** | **~12 小时** | **~1.5 天** |
| 500M (experts=16, top_k=4) | 16 | 2048 | 32 | ~30-50K | **~6 小时** | **~1.5 天** | **~5 天** |

> **估算方法**：
> - 200M MoE 每 token 约激活 62M 参数（top-2 + 共享专家）
> - A100 BF16 理论峰值 312 TFLOPS，预估利用率 25-35%
> - batch_size=32, seq_len=2048 时每步约 65K tokens
> - 真实吞吐受数据加载、MoE 路由计算、压缩注意力等影响

**推荐训练方案**：

| 目标 | 方案 | 单卡 A100 80GB 耗时 |
|------|------|-------------------|
| 快速验证 | 200M / 1B tokens | **~3 小时** |
| 基线模型 | 200M / 6B tokens | **~18 小时（1 天）** |
| 完整训练 | 200M / 20B tokens | **~3 天** |
| 更大规模 | 500M / 6B tokens | **~1.5 天** |

### 4.5 TensorBoard 监控

```shell
tensorboard --logdir=output/ --port=8080 --bind_all
```

训练会记录以下指标：
- `loss` / `ppl` — 交叉熵损失和困惑度
- `learning_rate` — 学习率调度曲线
- `Seq Aux Loss` — 序列级辅助损失（负载均衡）
- `MTP Loss` — 多 token 预测损失
- `ExpertBalance` — 各层专家负载 max/min 比值

### 4.6 断点续训

```shell
python ./train/pretrain.py \
  --model_name mini_deepseekv4 \
  --config_json configs/v4_200m_single_a100.json \
  --resume_from_checkpoint ./output/pretrained_mini_deepseekv4/checkpoints/checkpoint-XXXXX \
  --target_tokens 6_000_000_000
```

---

## 五、推理

### 终端交互

```shell
python ./example/test_terminal.py --model_name=mini_deepseekv4
```

### OpenAI 兼容 API

```shell
python ./example/test_api.py --model_name=mini_deepseekv4
```

搭配 [CherryStudio](https://www.cherry-ai.com/) 等前端使用。

---

## 六、自定义架构调整

### 调整参数量

编辑 `configs/v4_200m_single_a100.json`：

| 参数 | 增大 → 更大模型 | 减小 → 更快/更省显存 |
|------|---------------|-------------------|
| `hidden_size` | 768 → 1024 | 768 → 512 |
| `num_hidden_layers` | 10 → 16 | 10 → 8 |
| `n_routed_experts` | 8 → 16 | 8 → 4 |
| `n_activated_experts` | 2 → 4 | 2 → 2 |
| `moe_intermediate_size` | 768 → 1024 | 768 → 512 |

### 关闭 MTP 节省显存

```json
{ "use_mtp": false }
```

关闭后减少约 50M 参数（训练时），推理不受影响。

---

## 七、有问有答

### Q: 为什么是 200M？
A: 200M 参数是单卡 A100 80GB 的「甜蜜点」——显存充裕（~12GB 占用），训练速度快（1 天跑完 6B token），同时保持 MoE 架构的完整性。

### Q: 能否在更小的 GPU 上跑？
A: 可以。降低 `max_batch_size`、`max_seq_len`（如 1024），并开启梯度检查点（自行在 train/pretrain.py 中添加 `model.gradient_checkpointing_enable()`），可在 24GB 显存的 GPU（如 RTX 4090）上运行，但 batch_size 需降到 2-4。

### Q: 和原始 DeepSeek-V4 有什么区别？
A:
- **参数量大幅缩小**：原始 V4 百亿级 → 本项目 200M
- **专家数量减少**：原始 256 专家 → 本项目 8 专家
- **层数减少**：原始 60 层 → 本项目 10 层
- **核心架构保持一致**：MLA / CSA / HCA / MoE / MTP 逻辑相同

### Q: 如何在新数据上训练 tokenizer？
A: 运行 `python ./train/train_tokenizer.py`（需将对应脚本从原 Mini-LLM 复制）。
