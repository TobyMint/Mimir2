# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MimirConnector: a minimal CPU-memory KV cache connector for vLLM v1.

Purpose
-------
This connector stores a request's KV cache in pinned CPU memory and reloads it
into the GPU paged buffer on a cache hit. It exists to let us measure the
*end-to-end* cost of the reload path (CPU->GPU) versus recomputing the prefix
(prefill), including all connector per-step overhead.

Design notes
------------
- Mirrors SharedStorageConnector's scheduler/worker interface (same hooks), but
  replaces per-layer safetensors disk files with an in-process dict of pinned
  CPU tensors. Disk I/O in the debug SharedStorageConnector is far slower than
  prefill and would mislead the reload-vs-prefill comparison; CPU memory keeps
  the measurement fair.
- Cache key: the block-aligned prefix of the request's prompt tokens. A hit
  occurs when a previously stored prefix is a prefix of the current request,
  enabling reuse across multi-turn agent steps (the accumulating-prefix case
  that matters for this project), not just exact prompt repeats.
- First version is deliberately synchronous (blocking store/load, no layer-wise
  pipelining). Correctness and a fair baseline first; async overlap is a later
  optimization that must be justified by end-to-end gains, not assumed.
"""
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Module-level CPU KV store, shared across connector instances within a
# process. The v1 connector framework instantiates the connector twice per
# process (once for the SCHEDULER role, once for the WORKER role); the
# scheduler-side instance answers cache-hit queries while the worker-side
# instance performs the actual GPU<->CPU copies. Sharing one dict lets the
# scheduler observe what the worker stored. This is a single-process
# development arrangement (VLLM_ENABLE_V1_MULTIPROCESSING=0); cross-process
# sharing is a later concern for multi-worker deployments.
_CPU_KV_STORE: dict[str, dict[str, torch.Tensor]] = {}


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Floor-align token count down to a multiple of block_size."""
    return (num_tokens // block_size) * block_size


@dataclass
class ReqMeta:
    # Token ids of the prefix whose KV is being stored / loaded.
    token_ids: torch.Tensor
    # Slot mappings into the paged KV buffer, same length as token_ids.
    slot_mapping: torch.Tensor
    # Whether this step should store (True) or load (False) for this request.
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(token_ids: list[int], block_ids: list[int],
                  block_size: int, is_store: bool,
                  mm_hashes: list[str]) -> "ReqMeta":
        valid_num_tokens = _align_to_block_size(len(token_ids), block_size)
        if valid_num_tokens == 0:
            # Nothing block-aligned to transfer; build an empty meta.
            return ReqMeta(
                token_ids=torch.empty(0, dtype=torch.long),
                slot_mapping=torch.empty(0, dtype=torch.long),
                is_store=is_store,
                mm_hashes=mm_hashes,
            )
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
            block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class MimirConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(self, token_ids: list[int], block_ids: list[int],
                    block_size: int, is_store: bool,
                    mm_hashes: list[str]) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store,
                              mm_hashes))


class MimirConnector(KVConnectorBase_V1):
    """CPU-memory KV cache connector.

    State is split by role (enforced by the v1 connector framework):
      * SCHEDULER role: tracks which requests need load/store this step and
        answers cache-hit queries. Holds only token metadata, no tensors.
      * WORKER role: holds the CPU KV store and performs the actual
        GPU<->CPU copies on each forward.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size

        # Scheduler-side state: requests that need a load this step.
        self._requests_need_load: dict[str, Request] = {}

        # Worker-side state: CPU KV store (module-level, shared with the
        # scheduler-side instance of this connector in the same process).
        self._cpu_kv_store = _CPU_KV_STORE

    # ==============================
    # Cache key helpers
    # ==============================
    @staticmethod
    def _prefix_hash(token_ids: torch.Tensor) -> str:
        """Hash a block-aligned token prefix to a stable cache key."""
        return hashlib.md5(token_ids.numpy().tobytes(),
                           usedforsecurity=False).hexdigest()

    def _lookup_prefix_hit(
        self,
        request: "Request",
    ) -> Optional[tuple[str, int]]:
        """Find the longest stored prefix of this request's prompt.

        Returns (cache_key, num_hit_tokens) or None. We scan candidate
        block-aligned prefix lengths from the longest down so the first hit
        is the longest prefix. This supports the accumulating-prefix case:
        a multi-turn agent whose later requests extend an earlier stored
        prefix will hit on the shared prefix.
        """
        total = len(request.prompt_token_ids)
        aligned_total = _align_to_block_size(total, self._block_size)
        if aligned_total == 0:
            return None
        # Scan from longest aligned prefix down to one block.
        for n in range(aligned_total, 0, -self._block_size):
            prefix = torch.tensor(request.prompt_token_ids)[:n]
            key = self._prefix_hash(prefix)
            if key in self._cpu_kv_store:
                return key, n
        return None

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Copy cached KV from CPU store into vLLM's paged KV buffer."""
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        assert isinstance(metadata, MimirConnectorMetadata)
        if not isinstance(metadata, MimirConnectorMetadata):
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for request in metadata.requests:
            if request.is_store:
                continue
            if len(request.slot_mapping) == 0:
                continue
            key = self._prefix_hash(request.token_ids)
            layer_store = self._cpu_kv_store.get(key)
            if layer_store is None:
                logger.warning(
                    "MimirConnector load miss for key %s (tokens=%d)", key,
                    len(request.token_ids))
                continue

            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_attr = getattr(layer, 'kv_cache', None)
                if kv_cache_attr is None:
                    continue
                kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                src = layer_store.get(layer_name)
                if src is None:
                    continue
                self._inject_kv(kv_cache_layer, src.to(kv_cache_layer.device),
                                request.slot_mapping, attn_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Synchronous implementation: nothing to wait for.
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Copy this layer's KV from the paged buffer into the CPU store."""
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        assert isinstance(metadata, MimirConnectorMetadata)

        for request in metadata.requests:
            if not request.is_store:
                continue
            if len(request.slot_mapping) == 0:
                continue
            key = self._prefix_hash(request.token_ids)
            extracted = self._extract_kv(kv_layer, request.slot_mapping,
                                         attn_metadata)
            # Detach + move to pinned CPU. non_blocking=False for correctness
            # in this first version; overlap is a later optimization.
            cpu_tensor = extracted.detach().to("cpu", non_blocking=False)
            self._cpu_kv_store.setdefault(key, {})[layer_name] = cpu_tensor

    def wait_for_save(self):
        # Synchronous implementation: nothing to wait for.
        return

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """Return how many new tokens can be loaded from CPU beyond what is
        already computed locally."""
        hit = self._lookup_prefix_hit(request)
        if hit is None:
            return 0, False
        _, num_hit_tokens = hit
        new_tokens = num_hit_tokens - num_computed_tokens
        if new_tokens <= 0:
            return 0, False
        logger.info("MimirConnector cache hit: %d tokens (computed=%d)",
                    new_tokens, num_computed_tokens)
        return new_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build per-step metadata: which requests load vs store this step."""
        meta = MimirConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                # Load path: we have a cached prefix for this request.
                meta.add_request(token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size,
                                 is_store=False,
                                 mm_hashes=new_req.mm_hashes)
            else:
                # Store path: cache this request's prefix for future reuse.
                # Only store if there is at least one full block.
                aligned = _align_to_block_size(
                    len(new_req.prompt_token_ids), self._block_size)
                if aligned >= self._block_size:
                    meta.add_request(token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids[0],
                                     block_size=self._block_size,
                                     is_store=True,
                                     mm_hashes=new_req.mm_hashes)

        # Handle resumed-from-preemption requests that need a reload.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                num_computed_tokens = cached_reqs.num_computed_tokens[i]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                new_block_ids = cached_reqs.new_block_ids[i]
                request = self._requests_need_load[req_id]
                total_tokens = num_computed_tokens + num_new_tokens
                token_ids = request.all_token_ids[:total_tokens]
                meta.add_request(token_ids=token_ids,
                                 block_ids=new_block_ids[0],
                                 block_size=self._block_size,
                                 is_store=False,
                                 mm_hashes=request.mm_hashes)

        self._requests_need_load.clear()
        return meta

    # ==============================
    # KV extract / inject helpers
    # ==============================
    @staticmethod
    def _inject_kv(dst_kv_cache_layer: torch.Tensor,
                   src_kv_cache: torch.Tensor, slot_mapping: torch.Tensor,
                   attn_metadata: "AttentionMetadata") -> None:
        """Write src KV (per-token) into the paged KV buffer at slot_mapping."""
        dst_shape = dst_kv_cache_layer.shape
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages = dst_shape[0]
            page_size = dst_shape[1]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                num_pages * page_size, -1)
            dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            dst_kv_cache_layer.reshape(dst_shape)
        else:
            num_pages = dst_shape[1]
            page_size = dst_shape[2]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                2, num_pages * page_size, -1)
            dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
            dst_kv_cache_layer.reshape(dst_shape)

    @staticmethod
    def _extract_kv(layer: torch.Tensor, slot_mapping: torch.Tensor,
                    attn_metadata: "AttentionMetadata") -> torch.Tensor:
        """Read per-token KV from the paged KV buffer at slot_mapping."""
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = layer.shape[0], layer.shape[1]
            return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        num_pages, page_size = layer.shape[1], layer.shape[2]
        return layer.reshape(2, num_pages * page_size,
                             -1)[:, slot_mapping, ...]
