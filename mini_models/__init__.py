from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import PreTrainedModel
from typing import Tuple
from .generator import Generator

from .mini_deepseekv4.modeling_mini_deepseekv4 import MiniDeepSeekV4ForCausalLM, MiniDeepSeekV4Model
from .mini_deepseekv4.configuration_mini_deepseekv4 import MiniDeepSeekV4Config


# 注册到 transformers Auto 体系
AutoConfig.register("mini_deepseekv4", MiniDeepSeekV4Config)
AutoModel.register(MiniDeepSeekV4Config, MiniDeepSeekV4Model)
AutoModelForCausalLM.register(MiniDeepSeekV4Config, MiniDeepSeekV4ForCausalLM)


def get_model_and_config(model_name: str = "mini_deepseekv4", **kwargs):
    """
    获取模型类和配置类（当前项目仅支持 mini_deepseekv4）
    """
    if model_name != "mini_deepseekv4":
        raise ValueError(f"Unsupported model: {model_name}. This project only supports mini_deepseekv4.")
    return MiniDeepSeekV4ForCausalLM, MiniDeepSeekV4Config


def get_model_info(model: PreTrainedModel) -> Tuple[int, dict]:
    """
    计算模型可训练参数量和激活参数量信息
    """
    config = model.config
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_approx(params: int) -> str:
        return f"{params / 1_000_000 if params < 1_000_000_000 else params / 1_000_000_000:.1f}{' M' if params < 1_000_000_000 else ' B'}"

    # MTP 参数量（非推理时使用）
    mtp_params = 0
    if hasattr(model, "mtp") and model.mtp is not None:
        for name, param in model.mtp.named_parameters():
            if param.requires_grad:
                is_shared = False
                if name == "embed_tokens.weight" and model.mtp.embed_tokens is model.model.embed_tokens:
                    is_shared = True
                elif name == "lm_head.weight" and model.mtp.lm_head is model.lm_head:
                    is_shared = True
                if not is_shared:
                    mtp_params += param.numel()

    # MoE 激活比例
    activation_ratio = config.n_activated_experts / config.n_routed_experts

    # 所有增强专家的参数
    routed_experts_total = 0
    for i in range(config.num_hidden_layers):
        layer = model.model.layers[i]
        if hasattr(layer, 'ffn'):
            from .mini_deepseekv4.modeling_mini_deepseekv4 import MiniDeepSeekV4MoE
            if isinstance(layer.ffn, MiniDeepSeekV4MoE):
                for expert in layer.ffn.experts:
                    routed_experts_total += sum(p.numel() for p in expert.parameters() if p.requires_grad)

    activated = trainable_params - mtp_params - routed_experts_total + routed_experts_total * activation_ratio

    architecture_type = getattr(model, "architecture_type", "MoE")
    approx_params_info = {
        "architecture": architecture_type,
        "trainable_params": get_approx(trainable_params) + f" (including MTP: {get_approx(mtp_params)})",
        "total_params": get_approx(trainable_params - mtp_params) + f" (activated: {get_approx(activated)})",
    }
    return {"specific_params": trainable_params}, approx_params_info


__all__ = ['get_model_and_config', 'get_model_info', 'Generator']
