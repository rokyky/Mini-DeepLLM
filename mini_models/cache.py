from typing import Optional
import torch

from transformers.cache_utils import DynamicSlidingWindowLayer

from mini_models.mini_deepseekv4.configuration_mini_deepseekv4 import MiniDeepSeekV4Config


class MiniDeepSeekV4CacheLayer(DynamicSlidingWindowLayer):
    """
    MiniDeepSeekV4 的 CacheLayer, 统一 Sliding、CSA 和 HCA 的缓存管理
    其中的 compressor 字段用于存储进行 core attention 计算的 kv entry, indexer 字段用于存储 indexer 内部 compressor 的 kv entry
    """
    def __init__(self, config: MiniDeepSeekV4Config):
        super().__init__(sliding_window=config.window_size)
        self.buffer_kv: dict[str, Optional[torch.Tensor]] = {"compressor": None, "indexer": None}
        self.buffer_gate: dict[str, Optional[torch.Tensor]] = {"compressor": None, "indexer": None}
        self.overlap_kv: dict[str, Optional[torch.Tensor]] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, Optional[torch.Tensor]] = {"compressor": None, "indexer": None}
        self.compressed_kv: dict[str, Optional[torch.Tensor]] = {"compressor": None, "indexer": None}
        self.compressed_valid: dict[str, Optional[torch.Tensor]] = {"compressor": None, "indexer": None}
        self.entry_count: dict[str, int] = {"compressor": 0, "indexer": 0}

    def _cache_is_initialized(self) -> bool:
        if hasattr(self, "is_initialized"):
            return bool(self.is_initialized)
        return self.keys is not None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor | None = None,
    ) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = self.keys
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self._cache_is_initialized():
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def update_compressor_buffer(
        self,
        name: str,
        compress_ratio: int,
        kv: torch.Tensor,
        gate: torch.Tensor,
        overlap: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        batch_size = kv.shape[0]
        buffered_kv = self.buffer_kv[name]
        buffered_gate = self.buffer_gate[name]
        context_kv = self.overlap_kv[name] if overlap else None
        context_gate = self.overlap_gate[name] if overlap else None

        if buffered_kv is not None and buffered_kv.shape[1] > 0:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)

        usable = (kv.shape[1] // compress_ratio) * compress_ratio
        n_blocks = usable // compress_ratio
        first_block_position = self.entry_count[name] * compress_ratio
        self.buffer_kv[name] = kv[:, usable:]
        self.buffer_gate[name] = gate[:, usable:]

        if n_blocks == 0:
            block_positions = torch.empty(batch_size, 0, dtype=torch.long, device=kv.device)
            block_valid = torch.empty(batch_size, 0, dtype=torch.bool, device=kv.device)
            return kv[:, :usable], gate[:, :usable], block_positions, block_valid

        new_kv = kv[:, :usable]
        new_gate = gate[:, :usable]

        if overlap:
            self.overlap_kv[name] = new_kv[:, -compress_ratio:]
            self.overlap_gate[name] = new_gate[:, -compress_ratio:]

        if overlap and context_kv is not None and context_kv.shape[1] > 0:
            flat_kv = torch.cat([context_kv, new_kv], dim=1)
            flat_gate = torch.cat([context_gate, new_gate], dim=1)
            positions = torch.arange(
                first_block_position - compress_ratio,
                first_block_position + usable,
                compress_ratio,
                dtype=torch.long,
                device=kv.device,
            )
            block_valid = torch.ones(batch_size, n_blocks + 1, dtype=torch.bool, device=kv.device)
            block_valid[:, 0] = False
        else:
            flat_kv = new_kv
            flat_gate = new_gate
            positions = torch.arange(
                first_block_position,
                first_block_position + usable,
                compress_ratio,
                dtype=torch.long,
                device=kv.device,
            )
            block_valid = torch.ones(batch_size, n_blocks, dtype=torch.bool, device=kv.device)

        block_positions = positions.unsqueeze(0).expand(batch_size, -1)
        return flat_kv, flat_gate, block_positions, block_valid

    def update_compressor_states(
        self,
        name: str,
        compressed: torch.Tensor,
        block_valid: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        batch_size, n_new, head_dim = compressed.shape
        if block_valid is None:
            block_valid = torch.ones(batch_size, n_new, dtype=torch.bool, device=compressed.device)

        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
            self.compressed_valid[name] = block_valid
        elif n_new > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
            self.compressed_valid[name] = torch.cat([self.compressed_valid[name], block_valid], dim=1)

        self.entry_count[name] += n_new
        return self.compressed_kv[name], self.compressed_valid[name]
