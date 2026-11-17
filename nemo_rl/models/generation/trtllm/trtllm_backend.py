# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRT-LLM WorkerExtension for NCCL / IPC weight synchronisation.

Injected into TRT-LLM's RayGPUWorker via ``ray_worker_extension_cls``.

- ``update_weights_from_collective`` — NCCL broadcast via
  ``packed_broadcast_consumer``, used in non-colocated mode.
- ``update_weights_via_ipc_zmq`` — CUDA IPC handles streamed over a
  per-GPU ZMQ socket, used in colocated mode (NCCL can't form a group
  when train and inference processes share the same physical GPU).
"""

import gc
import inspect
import math
import os
import re
import time
from collections import OrderedDict
from typing import Any, Optional

import torch
import zmq
from tensorrt_llm.executor.ray.utils import control_action_decorator
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension

from nemo_rl.models.generation.trtllm.quantization import fp8 as fp8_quantization
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

# Disable TRT-LLM weight loader's ThreadPoolExecutor: serial loading keeps
# all copies on the caller's stream (same as NCCL writes), so the existing
# stream-level sync in packed_broadcast_consumer covers them without us
# needing defensive cross-stream synchronize() calls. Also lower peak memory.
os.environ.setdefault("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")


def _require_fp8_refit_hooks(model_loader: Any) -> None:
    """Require TRT-LLM hooks for transactional FP8 refits."""
    required_hooks = (
        "begin_update_weights",
        "finalize_update_weights",
        "abort_update_weights",
    )
    missing_hooks = [
        hook_name
        for hook_name in required_hooks
        if not callable(getattr(model_loader, hook_name, None))
    ]
    if missing_hooks:
        raise RuntimeError(
            "FP8 refit requires TRT-LLM weight-update hooks. "
            f"Missing APIs: {missing_hooks}."
        )


class NcclExtension(WorkerExtension):
    """NCCL-based weight update extension for TRT-LLM Ray workers.

    Attributes set by TRT-LLM's mixin injection (from ``RayGPUWorker``):
        self.engine    – ``PyExecutor`` instance
        self.device_id – int GPU ordinal

    The extension has no constructor of its own, so the refit state below is
    declared here and populated by the refit setup calls.
    """

    # Routed-expert prefix -> expert ids this rank stores (see
    # fp8_quantization.build_local_expert_lookup); None converts every expert.
    _local_expert_lookup: Optional[fp8_quantization.LocalExpertLookup] = None
    # nccl_reshard: per-PP-stage bulk communicators, the transfer plan, the misc
    # broadcast's (shape, dtype) map, and the bulk param -> local slot specs.
    pp_comm_groups: Optional[dict[int, Any]] = None
    nccl_reshard_refit_info: Optional[dict[str, Any]] = None
    misc_state_dict_info: Optional[dict[str, Any]] = None
    hf_to_local_param_map: Optional[Any] = None
    # nccl_reshard: converted expert stacks waiting for one batched reload, and
    # the expert prefix -> MoE weight-owner module map (built once).
    _pending_bulk_reload: Optional[dict[str, Any]] = None
    _moe_weight_owner_map: Optional[dict[str, Any]] = None
    # Models whose load_weights takes only a complete set of weights (GLM, DeepSeek V3):
    # non-expert tensors wait here, in host memory, for one load at the end of the refit.
    _staged_misc_weights: Optional[dict[str, Any]] = None
    _direct_expert_tensors: int = 0
    _expert_slot_checked: bool = False
    _checkpoint_fp8_names: Optional[frozenset] = None
    _refits_done: int = 0
    _addresses_before: Optional[dict] = None
    _delta_before: Optional[dict] = None
    _memory_before: Optional[tuple] = None
    _sent_now: Optional[dict] = None
    _sent_prev: Optional[dict] = None
    # nccl_reshard env knobs, parsed in prepare_nccl_reshard_refit_info.
    # Converted bytes to accumulate before one ``model_loader.reload`` call:
    # every reload walks the whole module tree, so batching keeps that host cost
    # off the transfer's critical path; 0 means one reload per received stack.
    _reload_batch_bytes: int = 2 << 30
    # Receive streams across PP stages.
    _refit_num_streams: int = 2

    # ------------------------------------------------------------------ #
    #  Collective initialisation (called once during setup)
    # ------------------------------------------------------------------ #

    # Park the executor loop while the refit group is built: its NCCL collectives
    # deadlock against the blocking ncclCommInitRank (seen as a 600 s watchdog kill).
    @control_action_decorator
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        assert torch.distributed.is_initialized(), (
            "TRT-LLM backend requires torch.distributed to be initialized before init_collective"
        )
        local_rank = torch.distributed.get_rank()
        rank = train_world_size + rank_prefix + local_rank

        pg = StatelessProcessGroup(
            master_address=ip,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        pg.init_nccl_communicator(device=self.device_id)
        self.model_update_group = pg

    # ------------------------------------------------------------------ #
    #  GPU profiling (runs inside each nsys-wrapped GPU worker)
    # ------------------------------------------------------------------ #

    def start_gpu_profiling(self) -> None:
        """Start CUDA profiler on this GPU worker (nsys capture-range trigger)."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop CUDA profiler on this GPU worker."""
        torch.cuda.profiler.stop()

    # ------------------------------------------------------------------ #
    #  Refit metadata (weight name → (shape, dtype) mapping)
    # ------------------------------------------------------------------ #

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self.state_dict_info = state_dict_info
        model = self.engine.model_engine.model
        self._local_expert_lookup = None
        if fp8_quantization.is_quantized_expert_refit(model.model_config.quant_config):
            fp8_quantization.validate_routed_experts(state_dict_info)
            _require_fp8_refit_hooks(self.engine.model_engine.model_loader)
            # Every rank receives the full expert stacks but loads only its
            # own slots; convert just those (EP16: 32 of 512 experts).
            self._local_expert_lookup = fp8_quantization.build_local_expert_lookup(
                model
            )

    # ------------------------------------------------------------------ #
    #  NCCL weight receive + reload
    # ------------------------------------------------------------------ #

    def update_weights_from_collective(
        self,
        *,
        drain: bool = True,
        recompute_kv: bool = False,
    ) -> bool:
        """Receive weights via NCCL broadcast and update model parameters.

        Args:
            drain: If True (default), wait for all in-flight requests to
                drain before applying weights — exclusive engine access.
                If False, the swap happens at a scheduler step boundary
                with in-flight requests still in the engine (in-flight
                weight update).
            recompute_kv: If True, call ``recompute_active_requests()`` after
                the refit: in-flight requests are re-prefilled under the new
                weights and the prefix cache is reset. If False, in-flight
                requests keep decoding with their current KV cache. With
                ``drain=True`` there are no in-flight requests, so True reduces
                to a prefix-cache reset.
        """
        assert hasattr(self, "state_dict_info") and self.state_dict_info is not None, (
            "state_dict_info not set — call prepare_refit_info first"
        )
        with self.engine.control_action(drain=drain):
            try:
                # Delegate the whole refit lifecycle to WorkerExtension: begin
                # unwraps a torch.compile'd model (without which load_weights
                # matches nothing and allow_partial_loading silently keeps the
                # old weights), finish re-wraps it and resets the prefix cache.
                # The device is already quiesced -- py_executor's
                # _handle_control_request synchronizes before releasing the
                # barrier control_action waits on.
                if not self._partial_loading_supported():
                    # The receive buffers (5 GiB each) need free device memory:
                    # hand cached blocks back before the transfer starts.
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Before begin_weight_update: it re-creates parameters.
                    self._addresses_before = self._tensor_addresses()
                    self._delta_before = self._delta_snapshot()
                    self._memory_before = self._memory_mark()
                self.begin_weight_update()
                packed_broadcast_consumer(
                    iterator=iter(self.state_dict_info.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self._reload_bucket,
                )
                self._finish_refit(recompute_kv)
            except Exception:
                self._staged_misc_weights = None
                self._direct_expert_tensors = 0
                self._expert_slot_checked = False
                self._delta_before = None
                self._sent_now = None
                self.engine.model_engine.model_loader.abort_update_weights()
                raise

        return True

    # ------------------------------------------------------------------ #
    #  IPC weight receive + reload (colocated mode)
    # ------------------------------------------------------------------ #

    def get_zmq_address(self) -> str:
        # Trainer side binds the same path (per-GPU UUID) so workers sharing
        # the same physical GPU meet on one socket.
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self) -> None:
        if hasattr(self, "zmq_socket"):
            return
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 120000)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 120000)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.connect(self.get_zmq_address())

    @control_action_decorator
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive weights via CUDA-IPC + ZMQ, reload model.

        Trainer sends ``(ipc_handle, list_keys, used_bytes)`` chunks; end of
        refit is signalled by ``IPCProtocol.COMPLETE``.
        """
        assert hasattr(self, "state_dict_info") and self.state_dict_info is not None, (
            "state_dict_info not set — call prepare_refit_info first"
        )
        model_engine = self.engine.model_engine
        model = model_engine.model

        buffer = None
        weights = None
        try:
            self.maybe_init_zmq()
            self.begin_weight_update()

            while True:
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device_id)

                weights = {}
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]
                    if isinstance(shape, list):
                        shape = torch.Size(shape)
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weights[key] = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    offset += calculate_aligned_size(size_in_bytes)

                assert offset == used_bytes, (
                    f"IPC payload offset mismatch: computed={offset}, sent={used_bytes}. "
                    "Likely stale state_dict_info (wrong shape/dtype for some key)."
                )

                if fp8_quantization.is_quantized_expert_refit(
                    model.model_config.quant_config
                ):
                    weights = fp8_quantization.load_weights(
                        weights.items(),
                        is_mx=fp8_quantization.is_mxfp8_model(
                            model.model_config.quant_config
                        ),
                        local_experts=self._local_expert_lookup,
                    )
                    # The mapper may retain split QKVZ/BA views across IPC chunks;
                    # detach them before ACK lets the trainer reuse the buffer.
                    weights = fp8_quantization.clone_mapper_staging_weights(weights)

                model_engine.model_loader.reload(
                    model,
                    weights,
                    allow_partial_loading=True,
                )
                torch.cuda.current_stream().synchronize()

                # Drop views before ACK — trainer reuses the buffer on the
                # next chunk, lingering views would read corrupted data.
                del weights, buffer
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            self.finalize_weight_update()
            self.finish_weight_update()
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception:
            model_engine.model_loader.abort_update_weights()
            raise

    def cleanup_zmq(self) -> None:
        """Close ZMQ socket if open — called from worker shutdown."""
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            del self.zmq_socket
        if hasattr(self, "zmq_context"):
            self.zmq_context.destroy()
            del self.zmq_context

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def report_device_id(self) -> str:
        from tensorrt_llm._torch.utils import get_device_uuid

        return get_device_uuid(self.device_id)

    # ------------------------------------------------------------------ #
    #  Per-bucket reload shared by the collective and nccl_reshard paths
    # ------------------------------------------------------------------ #
    def _finish_refit(self, recompute_kv: bool) -> None:
        """Finalize a refit inside control_action."""
        self._flush_staged_weights()
        self.finalize_weight_update()
        stale_graphs = self._release_graphs_if_tensors_moved()
        self._report_refit_delta()
        self._report_refit_memory()
        if recompute_kv:
            # Re-prefill in-flight requests under the new weights; this
            # also resets the prefix cache (NVIDIA/TensorRT-LLM#17937).
            self.engine.recompute_active_requests()
        self.finish_weight_update()
        if stale_graphs:
            self._recapture_graphs()

    def _reload_bucket(self, weight_list) -> None:
        """Convert routed experts if the engine is FP8/MXFP8, then reload."""
        model_engine = self.engine.model_engine
        model = model_engine.model
        quant_config = model.model_config.quant_config
        if fp8_quantization.is_quantized_expert_refit(quant_config):
            weights = fp8_quantization.load_weights(
                weight_list,
                is_mx=fp8_quantization.is_mxfp8_model(quant_config),
                local_experts=self._local_expert_lookup,
            )
        else:
            weights = dict(weight_list)
        if not self._partial_loading_supported():
            self._reload_bucket_staged(weights)
            return
        model_engine.model_loader.reload(
            model,
            weights,
            allow_partial_loading=True,
        )

    # ------------------------------------------------------------------ #
    #  Models without partial loading (GLM / DeepSeek V3 load path)
    # ------------------------------------------------------------------ #
    # TRT-LLM loads these models through DeepseekV3WeightLoader, which takes a
    # complete set of weights only: ``model_loader.reload(...,
    # allow_partial_loading=True)`` asserts for them. Routed experts (per-expert
    # HF names) go straight into their MoE module, which does support partial
    # loads; everything else is copied to host memory and loaded once, at the
    # end of the refit, by the model's own loader with the expert modules skipped.
    _SPLIT_EXPERT_KEY_RE = re.compile(
        r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
        r"(?P<projection>gate_proj|up_proj|down_proj)\."
        r"(?P<leaf>weight|weight_scale_inv)$"
    )
    # HF projection name -> the name TRT-LLM's MoE modules load (the rename
    # DeepseekV3WeightLoader applies before it calls the MoE module).
    _EXPERT_PROJECTION_TO_MOE = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

    def _partial_loading_supported(self) -> bool:
        model = self.engine.model_engine.model
        return (
            "allow_partial_loading" in inspect.getfullargspec(model.load_weights).args
        )

    def _reload_bucket_staged(self, weights: dict) -> None:
        """Load a bucket's routed experts now and keep the rest for the end."""
        staged = self._staged_misc_weights
        if staged is None:
            staged = self._staged_misc_weights = {}
        by_prefix: dict[str, dict[str, Any]] = {}
        for name, tensor in weights.items():
            match = self._SPLIT_EXPERT_KEY_RE.fullmatch(name)
            if match is None:
                self._probe_received(name, tensor, None)
                if name in self._checkpoint_fp8_weights():
                    # The engine is built in the released checkpoint's layout:
                    # give the loader what that checkpoint would (FP8 + block scale).
                    fp8_data, scale_inv = fp8_quantization.cast_tensor_to_fp8_blockwise(
                        tensor
                    )
                    self._note_sent(name, tensor, fp8_data)
                    staged[name] = fp8_data.to("cpu")
                    staged[name + "_scale_inv"] = scale_inv.to("cpu")
                    continue
                # A host copy: the received view would pin its GPU receive buffer.
                staged[name] = tensor.detach().to("cpu", copy=True)
                continue
            if match["leaf"] == "weight":
                self._probe_received(
                    name, tensor, weights.get(name + "_scale_inv")
                )
            key = "{}.{}.{}".format(
                match["expert"],
                self._EXPERT_PROJECTION_TO_MOE[match["projection"]],
                match["leaf"],
            )
            by_prefix.setdefault(match["prefix"], {})[key] = tensor
        for prefix, module_weights in by_prefix.items():
            module = self._moe_weight_owner(prefix)
            module.load_weights(weights=[module_weights], allow_partial_loading=True)
            if not self._expert_slot_checked:
                self._expert_slot_checked = self._check_expert_slot(
                    prefix, module, module_weights
                )
            self._direct_expert_tensors += len(module_weights)

    # Tensors up to this size are kept alive across a refit so that a re-created
    # one can be written back into the memory the CUDA graphs were captured on.
    _PIN_MAX_ELEMENTS = 64 * 1024 * 1024

    def _tensor_slots(self):
        """Every parameter, buffer and tensor attribute: (key, module, kind, name, tensor)."""
        for module_name, module in self.engine.model_engine.model.named_modules():
            slots = [("p", name, value) for name, value in module._parameters.items()]
            slots += [("b", name, value) for name, value in module._buffers.items()]
            slots += [
                ("a", name, value)
                for name, value in vars(module).items()
                if isinstance(value, torch.Tensor)
            ]
            for kind, name, tensor in slots:
                if tensor is not None and tensor.numel() > 0:
                    yield f"{module_name}.{name}", module, kind, name, tensor

    def _tensor_addresses(self) -> dict:
        """key -> (address, the tensor itself when it is small enough to keep)."""
        return {
            key: (
                tensor.data_ptr(),
                tensor if tensor.numel() <= self._PIN_MAX_ELEMENTS else None,
            )
            for key, _module, _kind, _name, tensor in self._tensor_slots()
        }

    def _release_graphs_if_tensors_moved(self) -> bool:
        """Keep the CUDA graphs captured before the refit valid, or drop them.

        TRT-LLM keeps its captured graphs across a refit on the assumption that a
        refit moves no tensor. The DeepSeek V3 load path re-creates some (the
        indexer's fused projection, the absorbed MLA value projection), and a
        graph captured earlier keeps reading the old memory. A re-created tensor
        of the same shape is copied back into the old one, which stays the live
        tensor; if that is not possible for any of them, the graphs are released
        and decoding runs without them.
        """
        before, self._addresses_before = self._addresses_before, None
        if before is None:
            return False
        moved, kept_in_place, not_restorable = [], 0, []
        total = 0
        for key, module, kind, name, tensor in list(self._tensor_slots()):
            total += 1
            entry = before.get(key)
            if entry is None or entry[0] == tensor.data_ptr():
                continue
            moved.append(key)
            old = entry[1]
            if (
                old is None
                or old.shape != tensor.shape
                or old.dtype != tensor.dtype
                or old.device != tensor.device
            ):
                not_restorable.append(key)
                continue
            with torch.no_grad():
                old.copy_(tensor)
            if kind == "p":
                module._parameters[name] = old
            elif kind == "b":
                module._buffers[name] = old
            else:
                setattr(module, name, old)
            kept_in_place += 1
        print(
            f"[refit] {len(moved)} of {total} model tensors were re-created by the "
            f"refit; {kept_in_place} written back into their previous memory"
            + (f", e.g. {moved[:4]}" if moved else "")
            + (f"; NOT restorable: {not_restorable[:4]}" if not_restorable else ""),
            flush=True,
        )
        if not not_restorable:
            return False
        model_engine = self.engine.model_engine
        runner = model_engine.cuda_graph_runner
        freed = 0
        if runner is not None:
            # clear() below only forgets the padding dummies; free them in the KV
            # cache manager too, or a later capture cannot allocate them again.
            from tensorrt_llm._torch.pyexecutor.resource_manager import (
                ResourceManagerType,
            )

            kv_cache_manager = self.engine.resource_manager.get_resource_manager(
                ResourceManagerType.KV_CACHE_MANAGER
            )
            for cache in runner._padding_dummy_caches():
                for request in list(cache.values()):
                    kv_cache_manager.free_resources(request)
                    freed += 1
        model_engine._release_model_owned_cuda_graphs()
        model_engine._release_cuda_graphs()
        print(
            "[refit] released the CUDA graphs captured before the refit "
            f"({freed} padding dummies freed); decoding runs without graphs",
            flush=True,
        )
        return True

    def _recapture_graphs(self) -> None:
        """Run the engine warm-up again so decoding gets CUDA graphs back.

        Off unless NRL_TRTLLM_REFIT_RECAPTURE=1: the warm-up zero-fills the KV
        pools, which is only safe when no request is kept across the refit.
        """
        if os.environ.get("NRL_TRTLLM_REFIT_RECAPTURE", "0") != "1":
            return
        try:
            start = time.perf_counter()
            self.engine.model_engine.warmup(self.engine.resource_manager)
            print(
                f"[refit] CUDA graphs recaptured in {time.perf_counter() - start:.1f} s",
                flush=True,
            )
        except Exception as error:  # without graphs decoding is slower, not wrong
            print(
                "[refit] could not recapture CUDA graphs; decoding runs without "
                f"them: {error!r}",
                flush=True,
            )

    # What a refit needs on the engine's GPU. nvidia-smi shows the card full during
    # a refit, which may be PyTorch's cache rather than real demand, so report both:
    # PyTorch's own peak and the memory nobody held when the refit started.
    # NRL_TRTLLM_REFIT_MEMORY=0 turns it off.
    def _memory_mark(self) -> Optional[tuple]:
        """Device memory at the start of a refit; resets PyTorch's peak counters."""
        if os.environ.get("NRL_TRTLLM_REFIT_MEMORY", "1") != "1":
            return None
        try:
            free, total = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            return (free, total, torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        except Exception as error:  # a statistic must never break a refit
            print(f"[refit memory] mark failed: {error!r}", flush=True)
            return None

    def _report_refit_memory(self) -> None:
        """Print this refit's peak PyTorch memory next to what was available for it."""
        before, self._memory_before = self._memory_before, None
        if before is None:
            return
        try:
            import json

            gib = float(1024**3)
            free_before, total, allocated_before, reserved_before = before
            free_after, _total = torch.cuda.mem_get_info()
            peak_allocated = torch.cuda.max_memory_allocated()
            # What the refit could have used without failing: memory nobody held, plus
            # the blocks PyTorch had reserved but not handed out.
            available = free_before + (reserved_before - allocated_before)
            needed = peak_allocated - allocated_before
            print(
                "[refit memory] "
                + json.dumps(
                    {
                        "refit": self._refits_done,
                        "device": torch.cuda.current_device(),
                        "total_gib": round(total / gib, 1),
                        "free_before_gib": round(free_before / gib, 1),
                        "free_after_gib": round(free_after / gib, 1),
                        "allocated_before_gib": round(allocated_before / gib, 1),
                        "reserved_before_gib": round(reserved_before / gib, 1),
                        "peak_allocated_gib": round(peak_allocated / gib, 1),
                        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 1),
                        "allocated_after_gib": round(torch.cuda.memory_allocated() / gib, 1),
                        "reserved_after_gib": round(torch.cuda.memory_reserved() / gib, 1),
                        "needed_gib": round(needed / gib, 1),
                        "available_gib": round(available / gib, 1),
                        "margin_gib": round((available - needed) / gib, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:  # a statistic must never break a refit
            print(f"[refit memory] report failed: {error!r}", flush=True)

    # How much of the engine a refit really changes: an evenly spaced sample of
    # every model tensor is taken before the refit and compared, byte for byte,
    # with the same elements after it. NRL_TRTLLM_REFIT_DELTA=0 turns it off.
    _DELTA_SAMPLE = 32768

    def _delta_sample(self, tensor):
        """Evenly spaced elements of a tensor as raw bytes on the host: [n, itemsize]."""
        if not tensor.is_contiguous():
            return None
        raw = (
            tensor.detach()
            .reshape(-1)
            .view(torch.uint8)
            .view(-1, tensor.element_size())
        )
        step = max(1, raw.shape[0] // self._DELTA_SAMPLE)
        return raw[::step][: self._DELTA_SAMPLE].to("cpu", copy=True)

    def _delta_snapshot(self) -> Optional[dict]:
        """Samples of every model tensor, taken before a refit."""
        if os.environ.get("NRL_TRTLLM_REFIT_DELTA", "1") != "1":
            return None
        try:
            snapshot = {}
            for key, _module, _kind, _name, tensor in self._tensor_slots():
                sample = self._delta_sample(tensor)
                if sample is not None:
                    snapshot[key] = (str(tensor.dtype), tuple(tensor.shape), sample)
            self._sent_now = {}
            return snapshot
        except Exception as error:  # a statistic must never break a refit
            print(f"[refit delta] snapshot failed: {error!r}", flush=True)
            return None

    def _note_sent(self, name: str, tensor, fp8_data) -> None:
        """Keep a sample of a weight as the trainer sent it, and of its FP8 codes.

        Same element positions in both, so the report can say how many of the
        values the trainer changed since the previous refit also change the
        FP8 code the engine gets.
        """
        if self._sent_now is None:
            return
        try:
            sample = self._delta_sample(tensor)
            codes = None
            if tuple(fp8_data.shape) == tuple(tensor.shape):
                codes = self._delta_sample(fp8_data)
            if sample is not None:
                self._sent_now[name] = (sample, codes)
        except Exception as error:  # a statistic must never break a refit
            print(f"[refit delta] sent sample failed: {error!r}", flush=True)
            self._sent_now = None

    def _sent_delta(self) -> dict:
        """label -> [tensors, elements, sent values changed, FP8 codes changed, both]."""
        now, self._sent_now = self._sent_now, None
        previous, self._sent_prev = self._sent_prev, now
        groups: dict[str, list] = {}
        if not now or not previous:
            return groups
        for name, (sample, codes) in now.items():
            old = previous.get(name)
            if old is None or old[0].shape != sample.shape:
                continue
            sent_changed = (sample != old[0]).any(dim=1)
            label = re.sub("[.][0-9]+(?=[.]|$)", ".N", name)
            group = groups.setdefault(label, [0, 0, 0, 0, 0])
            group[0] += 1
            group[1] += int(sample.shape[0])
            group[2] += int(sent_changed.sum())
            if codes is not None and old[1] is not None and old[1].shape == codes.shape:
                code_changed = (codes != old[1]).any(dim=1)
                group[3] += int(code_changed.sum())
                group[4] += int((code_changed & sent_changed).sum())
        return groups

    def _report_refit_delta(self) -> None:
        """Print how many sampled elements this refit changed, per kind of tensor."""
        before, self._delta_before = self._delta_before, None
        if before is None:
            return
        try:
            import json

            start = time.perf_counter()
            groups: dict[str, list] = {}
            other_layout = 0
            for key, _module, _kind, _name, tensor in self._tensor_slots():
                entry = before.get(key)
                if entry is None:
                    continue
                dtype, shape, old = entry
                new = None
                if dtype == str(tensor.dtype) and shape == tuple(tensor.shape):
                    new = self._delta_sample(tensor)
                if new is None or new.shape != old.shape:
                    other_layout += 1
                    continue
                changed = int((new != old).any(dim=1).sum())
                # One group per dtype and module path with the indices removed.
                label = dtype.replace("torch.", "") + " " + re.sub(
                    "[.][0-9]+(?=[.]|$)", ".N", key
                )
                group = groups.setdefault(label, [0, 0, 0, 0])
                group[0] += 1
                group[1] += 1 if changed else 0
                group[2] += int(old.shape[0])
                group[3] += changed
            print(
                "[refit delta] "
                + json.dumps(
                    {
                        "refit": self._refits_done,
                        "other_layout": other_layout,
                        "seconds": round(time.perf_counter() - start, 2),
                        # label -> [tensors, tensors changed, elements sampled, elements changed]
                        "groups": groups,
                        # Weights the engine holds in FP8, as the trainer sent them (BF16):
                        # label -> [tensors, elements, sent changed, FP8 code changed, both]
                        "sent": self._sent_delta(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:  # a statistic must never break a refit
            print(f"[refit delta] comparison failed: {error!r}", flush=True)

    def _checkpoint_dir(self) -> Optional[str]:
        llm_args = getattr(self.engine.model_engine.model_loader, "llm_args", None)
        model_dir = getattr(llm_args, "model", None)
        return None if model_dir is None else str(model_dir)

    def _checkpoint_fp8_weights(self) -> frozenset:
        """Names the released checkpoint stores in block FP8 (they have a scale).

        Empty unless the engine was built in that checkpoint's own layout (see
        fp8.checkpoint_quant_config): then non-expert weights are quantized the
        same way before they reach the loader.
        """
        names = self._checkpoint_fp8_names
        if names is not None:
            return names
        names = frozenset()
        model = self.engine.model_engine.model
        model_dir = self._checkpoint_dir()
        if (
            model_dir is not None
            and fp8_quantization.is_fp8_model(model.model_config.quant_config)
            and fp8_quantization.checkpoint_quant_config(model_dir) is not None
        ):
            import json

            index_path = os.path.join(model_dir, "model.safetensors.index.json")
            with open(index_path, encoding="utf-8") as handle:
                weight_map = json.load(handle)["weight_map"]
            suffix = "_scale_inv"
            names = frozenset(
                name[: -len(suffix)] for name in weight_map if name.endswith(suffix)
            )
            print(
                f"[refit] engine uses the checkpoint's FP8 layout: {len(names)} "
                "tensors are block FP8 there",
                flush=True,
            )
        self._checkpoint_fp8_names = names
        return names

    # First refit only: the trainer still holds the checkpoint's weights, so what
    # arrives can be compared with the checkpoint itself (dequantized).
    _PROBE_RE = re.compile(
        r"^(model\.embed_tokens\.weight|model\.norm\.weight|lm_head\.weight"
        r"|model\.layers\.(0|3|38|77)\.(self_attn\.[a-z_.]+\.(weight|bias)"
        r"|(input|post_attention)_layernorm\.weight"
        r"|mlp\.(gate\.weight|gate\.e_score_correction_bias"
        r"|(gate|up|down)_proj\.weight|shared_experts\.(gate|up|down)_proj\.weight"
        r"|experts\.(0|1|64|65|128|129|192|193)\.(gate|up|down)_proj\.weight)))$"
    )

    def _probe_received(self, name: str, tensor: Any, scale_inv: Any) -> None:
        if self._refits_done > 0 or self._PROBE_RE.fullmatch(name) is None:
            return
        if os.environ.get("NRL_TRTLLM_REFIT_PROBE", "1") == "0":
            return
        try:
            from safetensors import safe_open

            model_dir = self._checkpoint_dir()
            index = getattr(self, "_probe_index", None)
            if index is None:
                import json

                with open(
                    os.path.join(model_dir, "model.safetensors.index.json"),
                    encoding="utf-8",
                ) as handle:
                    index = self._probe_index = json.load(handle)["weight_map"]

            def read(key):
                with safe_open(os.path.join(model_dir, index[key]), framework="pt") as f:
                    return f.get_tensor(key)

            def dequantized(data, scale):
                data = data.to(torch.float32)
                if scale is None or data.dim() != 2:
                    return data
                rows, cols = data.shape
                scale = scale.to(torch.float32)
                return data * scale.repeat_interleave(128, 0)[:rows].repeat_interleave(
                    128, 1
                )[:, :cols]

            reference = dequantized(
                read(name), read(name + "_scale_inv") if name + "_scale_inv" in index else None
            )
            received = dequantized(
                tensor.detach().cpu(), None if scale_inv is None else scale_inv.detach().cpu()
            )
            if tuple(reference.shape) != tuple(received.shape):
                print(
                    f"[refit probe] {name}: SHAPE received {tuple(received.shape)} "
                    f"checkpoint {tuple(reference.shape)}",
                    flush=True,
                )
                return
            error = float((received - reference).norm() / (reference.norm() + 1e-30))
            print(
                f"[refit probe] {name}: rel err vs checkpoint {error:.3e} "
                f"(|received| {float(received.abs().mean()):.3e}, "
                f"|checkpoint| {float(reference.abs().mean()):.3e})",
                flush=True,
            )
        except Exception as error:  # the probe must never break a refit
            print(f"[refit probe] {name}: not compared: {error!r}", flush=True)

    def _check_expert_slot(self, prefix: str, module: Any, module_weights: dict) -> bool:
        """Compare one loaded expert slot with the tensor that was sent.

        The MoE loader skips keys it does not know when loading partially, so a
        naming mismatch would otherwise leave the old weights in place silently.
        Returns True once a slot has been compared.
        """
        local_ids = getattr(module, "initial_local_expert_ids", None)
        loaded_all = getattr(module, "w2_weight", None)
        if local_ids is None or loaded_all is None:
            return False
        local_ids = [int(i) for i in local_ids]
        for key, tensor in module_weights.items():
            expert, projection, leaf = key.split(".")
            if projection != "w2" or leaf != "weight" or int(expert) not in local_ids:
                continue
            slot = local_ids.index(int(expert))
            torch.cuda.current_stream().synchronize()
            loaded = loaded_all.data[slot]
            if (
                loaded.numel() != tensor.numel()
                or loaded.element_size() != tensor.element_size()
            ):
                print(
                    f"[refit] expert slot check skipped for {prefix}: loaded "
                    f"{tuple(loaded.shape)} {loaded.dtype} vs sent "
                    f"{tuple(tensor.shape)} {tensor.dtype}",
                    flush=True,
                )
                return True
            same = torch.equal(
                loaded.contiguous().view(torch.uint8).reshape(-1),
                tensor.to(loaded.device).contiguous().view(torch.uint8).reshape(-1),
            )
            if not same:
                raise RuntimeError(
                    f"refit: {prefix} expert {expert} (slot {slot}) does not hold "
                    "the weight that was just loaded into it"
                )
            print(
                f"[refit] expert slot check passed: {prefix} expert {expert} -> "
                f"slot {slot}, {tuple(loaded.shape)} {loaded.dtype}",
                flush=True,
            )
            return True
        return False

    def _flush_staged_weights(self) -> None:
        """Load the staged non-expert weights with the model's own loader."""
        staged, self._staged_misc_weights = self._staged_misc_weights, None
        expert_tensors, self._direct_expert_tensors = self._direct_expert_tensors, 0
        self._expert_slot_checked = False
        if staged is None and expert_tensors == 0:
            return
        self._refits_done += 1
        if not staged or expert_tensors == 0:
            raise RuntimeError(
                "refit for a model without partial loading needs both routed-expert "
                f"and other weights; got {expert_tensors} expert tensors and "
                f"{len(staged or {})} others"
            )
        from tensorrt_llm._torch.models.modeling_deepseekv3 import (
            DeepseekV3ForCausalLM,
            DeepseekV3WeightLoader,
        )

        model = self.engine.model_engine.model
        if not isinstance(model, DeepseekV3ForCausalLM):
            raise NotImplementedError(
                f"refit without partial loading is implemented for the DeepSeek V3 "
                f"load path only, not {type(model).__name__}"
            )
        if fp8_quantization.is_fp8_model(model.model_config.quant_config):
            # Under block FP8 the loader always dequantizes kv_b_proj and reads a
            # block scale next to it. The trainer sends BF16, so give it unit
            # scales: the weight passes through unchanged.
            for name in [n for n in staged if n.endswith(".kv_b_proj.weight")]:
                scale_name = name + "_scale_inv"
                if scale_name not in staged:
                    rows, cols = staged[name].shape
                    staged[scale_name] = torch.ones(
                        (math.ceil(rows / 128), math.ceil(cols / 128)),
                        dtype=torch.float32,
                    )
        print(
            f"[refit] loading {len(staged)} staged non-expert tensors with "
            f"DeepseekV3WeightLoader; {expert_tensors} routed-expert tensors were "
            "loaded directly",
            flush=True,
        )
        # After the first load, setup_aliases() registers every layer's input norm
        # (and the final norm) a second time, as the previous layer's
        # ``next_layer_layernorm``. named_modules() then reports them under that
        # alias only, and the loader skips that name, so the norms would keep
        # their old weights. Drop the aliases for the load; set them up again after.
        for layer in model.model.layers:
            if getattr(layer, "next_layer_layernorm", None) is not None:
                layer.next_layer_layernorm = None
        # Every module gets fresh, untransformed weights in this refit: clear the
        # "already transformed" guards, as model_loader.reload does before a full
        # reload. The loader writes some weights (kv_a/q_a, kv_b) straight into
        # the parameters, which does not clear the guard by itself.
        self.engine.model_engine.model_loader._reset_weights_transformed(model)
        seen: set[int] = set()
        second_names = []
        for name, module in model.named_modules(remove_duplicate=False):
            if id(module) in seen and len(module._parameters) > 0:
                second_names.append(name)
            seen.add(id(module))
        if second_names:
            print(
                f"[refit] {len(second_names)} parameter-holding modules are still "
                f"registered under a second name, e.g. {second_names[:4]}",
                flush=True,
            )
        DeepseekV3WeightLoader(model).load_weights(staged, skip_modules=[".mlp.experts"])
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        # Compare what the engine now holds with what was sent, for every tensor
        # the loader stores unchanged (same name, shape and dtype).
        compared, differing = 0, []
        for name, param in model.named_parameters():
            sent = staged.get(name)
            if (
                sent is None
                or tuple(sent.shape) != tuple(param.shape)
                or sent.dtype != param.dtype
            ):
                continue
            compared += 1
            if not torch.equal(param.detach().cpu(), sent):
                differing.append(name)
        print(
            f"[refit] parameter check: {compared} tensors compared with what was "
            f"sent, {len(differing)} differ"
            + (f", e.g. {differing[:6]}" if differing else ""),
            flush=True,
        )
        model.setup_aliases()
        if compared == 0 or "model.norm.weight" in differing or len(differing) * 2 > compared:
            raise RuntimeError(
                f"refit: the engine does not hold the weights that were sent "
                f"({len(differing)} of {compared} compared tensors differ)"
            )

    # ------------------------------------------------------------------ #
    #  nccl_reshard (shard-to-shard) refit
    # ------------------------------------------------------------------ #
    # The trainer reshards the routed experts straight into each engine's
    # EP-local slots over one communicator per PP stage (that stage's train
    # ranks + this engine layout's ranks); everything else rides the misc
    # packed broadcast on model_update_group, loaded by _reload_bucket.
    @control_action_decorator
    def init_nccl_reshard_comm_group(
        self,
        rank_prefix: int,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Join the bulk-path communicator of every PP stage.

        Parked at a step boundary for the same reason as ``init_collective``:
        ncclCommInitRank across train + inference ranks deadlocks against the
        executor loop's own NCCL object collectives.
        """
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        assert torch.distributed.is_initialized(), (
            "TRT-LLM backend requires torch.distributed to be initialized before "
            "init_nccl_reshard_comm_group"
        )
        gen_rank_in_group = (
            train_ranks_per_stage + rank_prefix + torch.distributed.get_rank()
        )
        stale_groups = list((self.pp_comm_groups or {}).values())
        self.pp_comm_groups = {}
        torch.cuda.empty_cache()
        for stage in range(pp_size):
            print(
                f"  refit: reshard rendezvous [gen] stage={stage} "
                f"addr={pp_ips[stage]}:{pp_ports[stage]} "
                f"rank={gen_rank_in_group} world_size={sub_world_size}",
                flush=True,
            )
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=gen_rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=self.device_id)
            self.pp_comm_groups[stage] = group
        for previous in stale_groups:
            release_within(
                previous.abort, RELEASE_GRACE_S, "a previous reshard bulk communicator"
            )
        refit_info = self.nccl_reshard_refit_info
        if refit_info is not None:
            self.hf_to_local_param_map = self._build_expert_local_param_map(refit_info)

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Keep the per-layer transfer plan and map its bulk params to local slots."""
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            _STR_TO_DTYPE,
            restore_refit_info_placements,
        )

        # Parsed here, before any refit: a bad value raised mid-receive would
        # leave the trainers blocked on sends that are never received.
        self._reload_batch_bytes = int(
            os.environ.get("NRL_TRTLLM_REFIT_RELOAD_BATCH_BYTES", str(2 << 30))
        )
        self._refit_num_streams = int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2"))
        self.nccl_reshard_refit_info = restore_refit_info_placements(refit_info)
        model = self.engine.model_engine.model
        quant_config = model.model_config.quant_config
        # The local-expert lookup doubles as the slot oracle for the bulk
        # specs, so build it for bf16 engines too.
        self._local_expert_lookup = fp8_quantization.build_local_expert_lookup(model)
        if self._local_expert_lookup is None:
            raise RuntimeError(
                "nccl_reshard refit needs the MoE modules' initial_local_expert_ids "
                "(none found, or an expert load balancer is active)"
            )
        if fp8_quantization.is_quantized_expert_refit(quant_config):
            _require_fp8_refit_hooks(self.engine.model_engine.model_loader)
        misc_meta = self.nccl_reshard_refit_info.get("misc_meta", {})
        self.misc_state_dict_info = {
            name: (torch.Size(meta["shape"]), _STR_TO_DTYPE[meta["dtype"]])
            for name, meta in misc_meta.items()
        }
        if self.pp_comm_groups:
            self.hf_to_local_param_map = self._build_expert_local_param_map(
                self.nccl_reshard_refit_info
            )

    def _build_expert_local_param_map(self, refit_info: dict):
        """One LocalParamSpec per grouped routed-expert projection.

        ``pre`` allocates a canonical ``[E_local, out, in]`` bf16 staging
        buffer for this rank's EP slice, ``post`` turns it into per-expert HF
        entries (FP8/MXFP8 for quantized engines) and hands them to TRT-LLM's
        reload. The EP slice must be the slots TRT-LLM loads
        (``initial_local_expert_ids``); a mismatch is an error, never a
        silent partial update.
        """
        from torch.distributed._tensor import Shard

        from nemo_rl.weight_sync.nccl_reshard_utils import (
            _STR_TO_DTYPE,
            HFToLocalParamMap,
            LocalParamSpec,
            RefitCtx,
        )
        from nemo_rl.weight_sync.xferdtensor_python import local_shard_slices

        pp_comm_groups = self.pp_comm_groups
        if not pp_comm_groups:
            raise RuntimeError(
                "nccl_reshard refit mapping needs the per-PP-stage communicators"
            )
        lookup = self._local_expert_lookup
        assert lookup is not None, "prepare_nccl_reshard_refit_info must run first"
        device = torch.device("cuda", self.device_id)
        specs = {}
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                name = param_info["name"]
                projection = param_info.get("grouped_expert_proj")
                if projection is None:
                    raise NotImplementedError(
                        "TRT-LLM nccl_reshard refit moves grouped routed experts only; "
                        f"bulk param {name!r} would need this engine's TP layout"
                    )
                bad_dims = [
                    placement.dim
                    for placement in param_info["dst_placements"]
                    if isinstance(placement, Shard) and placement.dim != 0
                ]
                if bad_dims:
                    raise ValueError(
                        f"{name!r}: TRT-LLM experts shard on EP only (dst Shard dims "
                        f"{bad_dims} unsupported)"
                    )
                stage = param_info.get("pp_stage", 0)
                if stage not in pp_comm_groups:
                    raise RuntimeError(f"no reshard communicator for PP stage {stage}")
                rank = pp_comm_groups[stage].rank
                slices = local_shard_slices(param_info, rank)
                global_shape = tuple(param_info["global_shape"])
                local_shape = []
                for size, sl in zip(global_shape, slices):
                    start = 0 if sl.start is None else sl.start
                    stop = size if sl.stop is None else sl.stop
                    local_shape.append(stop - start)
                local_shape = tuple(local_shape)
                expert_start = 0 if slices[0].start is None else slices[0].start
                expert_ids = list(range(expert_start, expert_start + local_shape[0]))
                prefix = name.rsplit(f".{projection}.weight", 1)[0]
                expected = lookup(prefix)
                if expected is None:
                    raise RuntimeError(
                        f"{name!r}: no MoE module for expert prefix {prefix!r} on this rank"
                    )
                if list(expected) != expert_ids:
                    raise RuntimeError(
                        f"{name!r}: reshard EP slice {expert_ids[0]}..{expert_ids[-1]} "
                        f"differs from TRT-LLM's local slots "
                        f"{list(expected)[0]}..{list(expected)[-1]}"
                    )
                dtype_value = param_info.get("dtype")
                dtype = (
                    dtype_value
                    if isinstance(dtype_value, torch.dtype)
                    else _STR_TO_DTYPE.get(str(dtype_value))
                )
                if dtype is None:
                    raise ValueError(
                        f"{name!r}: unsupported wire dtype {dtype_value!r}"
                    )

                def pre(_base, shape=local_shape, dtype=dtype):
                    return RefitCtx(buf=torch.empty(shape, dtype=dtype, device=device))

                def post(
                    ctx, prefix=prefix, projection=projection, expert_ids=expert_ids
                ):
                    self._queue_received_experts(
                        prefix, projection, expert_ids, ctx.buf
                    )

                specs[name] = LocalParamSpec(
                    base=None, pre=pre, post=post, wire_dtype=dtype
                )
        return HFToLocalParamMap(specs=specs)

    def _convert_received_experts(
        self, prefix: str, projection: str, expert_ids: list[int], stack: torch.Tensor
    ) -> dict:
        """Turn one received ``[E_local, out, in]`` stack into loader-ready tensors."""
        model = self.engine.model_engine.model
        quant_config = model.model_config.quant_config
        if fp8_quantization.is_quantized_expert_refit(quant_config):
            weights: dict = {}
            fp8_quantization.convert_expert_projection_stack(
                weights,
                prefix=prefix,
                projection=projection,
                tensor=stack,
                expert_ids=expert_ids,
                is_mx=fp8_quantization.is_mxfp8_model(quant_config),
            )
            return weights
        # bf16 engines load the staging views directly; keep them alive until
        # the batched reload has copied them (record_stream is on the caller).
        return {
            f"{prefix}.{expert_id}.{projection}.weight": expert
            for expert_id, expert in zip(expert_ids, stack.unbind(0))
        }

    def _queue_received_experts(
        self, prefix: str, projection: str, expert_ids: list[int], stack: torch.Tensor
    ) -> None:
        """Convert one received stack and reload it, batched under the byte budget."""
        weights = self._convert_received_experts(prefix, projection, expert_ids, stack)
        pending = self._pending_bulk_reload
        if pending is None:
            pending = self._pending_bulk_reload = {"weights": {}, "bytes": 0}
        pending["weights"].update(weights)
        pending["bytes"] += sum(t.numel() * t.element_size() for t in weights.values())
        if pending["bytes"] >= self._reload_batch_bytes:
            self._flush_received_experts()

    @staticmethod
    def _direct_expert_load_enabled() -> bool:
        """Load received expert stacks straight into their MoE modules.

        ``model_loader.reload`` walks every module of the model per call and
        ends with a stream synchronize, so each call stalls the receive loop
        until every transfer enqueued so far has landed. The direct path replays
        the loader's MoE branch for just the modules the queued stacks belong
        to and leaves the copies enqueued. NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD=0
        falls back to the loader.
        """
        return os.environ.get("NRL_TRTLLM_REFIT_DIRECT_EXPERT_LOAD", "1") != "0"

    def _moe_weight_owners(self) -> dict:
        """``expert prefix -> module`` for every MoE weight owner, built once."""
        owners = self._moe_weight_owner_map
        if owners is not None:
            return owners
        from tensorrt_llm._torch.moe.fused_moe.weight_owner import (
            is_moe_weight_owner,
        )

        owners = {}
        for name, module in self.engine.model_engine.model.named_modules():
            if not is_moe_weight_owner(module):
                continue
            # The loader addresses ConfigurableMoE's weight-owning backend by
            # its parent's name (see _load_weights_impl_v2).
            if name.endswith(".backend"):
                name = name[: -len(".backend")]
            owners[name] = module
        self._moe_weight_owner_map = owners
        return owners

    def _moe_weight_owner(self, prefix: str) -> Any:
        """The MoE weight owner of an expert prefix, matched as a dotted suffix.

        Owners are named from the engine's root module, while weight prefixes
        are named from the module the mapper loads into: a VLM wrapper
        (Qwen3.5) nests its decoder under ``llm.``, and the mapper is bound to
        that decoder only once ``load_weights`` has run.
        """
        owners = self._moe_weight_owners()
        module = owners.get(prefix)
        if module is not None:
            return module
        matches = [
            owner for name, owner in owners.items() if name.endswith("." + prefix)
        ]
        if len(matches) != 1:
            # Every queued weight is a routed expert: no single match means the
            # owner names no longer line up with the weight names.
            raise RuntimeError(
                f"expected one TRT-LLM MoE weight owner for expert prefix "
                f"{prefix!r}, found {len(matches)} (known owners: "
                f"{sorted(owners)[:4]}...)"
            )
        owners[prefix] = matches[0]
        return matches[0]

    def _load_expert_weights_direct(self, weights: dict) -> dict:
        """Replay the loader's per-module MoE branch for the queued stacks.

        This copies upstream TRT-LLM code at the pinned ``[tool.trtllm].ref``:
        the per-module branch of ``_load_weights_impl_v2``
        (tensorrt_llm/_torch/models/modeling_utils.py) and, for mappers that
        take ``allow_partial_loading`` (Qwen3.5), the ``preprocess_weights``
        call of the model's ``load_weights``. Other mappers (Qwen3 MoE) inherit
        the base no-op that returns None, and their models never call it, so
        it is skipped. It skips the
        ``params_map`` rename, ``should_skip_module``,
        ``does_require_special_handling`` and ``mark_consumed`` steps, which do
        not apply to routed-expert stacks; re-check this copy on every TRT-LLM
        bump.

        Returns the weights that no MoE module claimed: only entries the mapper
        filtered out of every owner's prefix (loaded by the generic loader
        afterwards).
        """
        model_loader = self.engine.model_engine.model_loader
        mapper = model_loader.weight_mapper
        if mapper is None:
            return weights
        preprocess = mapper.preprocess_weights
        if "allow_partial_loading" in inspect.signature(preprocess).parameters:
            weights = preprocess(weights, allow_partial_loading=True)
        # "<prefix>.<expert>.<projection>.<leaf>" -> "<prefix>"
        prefixes = {name.rsplit(".", 3)[0] for name in weights if name.count(".") >= 3}
        leftover = dict(weights)
        for prefix in sorted(prefixes):
            module = self._moe_weight_owner(prefix)
            module_weights = mapper.filter_weights(prefix, weights)
            if not module_weights:
                continue
            module_name = prefix.rsplit(".", 1)[-1]
            if mapper.is_special_instance_module(module):
                mapper.handle_special_instance_module(
                    module, module_name, module_weights, allow_partial_loading=True
                )
            else:
                module.load_weights(
                    weights=[module_weights], allow_partial_loading=True
                )
            for key in list(leftover):
                if key.startswith(prefix + "."):
                    del leftover[key]
        return leftover

    def _flush_received_experts(self) -> None:
        """Reload every queued converted stack in one loader call."""
        pending = self._pending_bulk_reload
        if not pending or not pending["weights"]:
            return
        weights = pending["weights"]
        self._pending_bulk_reload = {"weights": {}, "bytes": 0}
        model_engine = self.engine.model_engine
        if self._direct_expert_load_enabled():
            weights = self._load_expert_weights_direct(weights)
        if weights:
            model_engine.model_loader.reload(
                model_engine.model, weights, allow_partial_loading=True
            )

    def _recv_bulk_params(self) -> None:
        """Receive every bulk param of every PP stage into its local slots."""
        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        refit_info = self.nccl_reshard_refit_info
        pp_comm_groups = self.pp_comm_groups
        param_map = self.hf_to_local_param_map
        stage_params = OrderedDict()
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                stage_params.setdefault(param_info.get("pp_stage", 0), []).append(
                    param_info
                )
        num_streams = max(1, min(self._refit_num_streams, len(stage_params)))
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events = {}
        for idx, (stage, params) in enumerate(stage_params.items()):
            if (idx - num_streams) in events:
                events[idx - num_streams].synchronize()
            stream = streams[idx % num_streams]
            with torch.cuda.stream(stream):
                group = pp_comm_groups[stage]
                for param_info in params:
                    spec = param_map.get(param_info["name"])
                    assert spec is not None, (
                        f"nccl_reshard_refit: {param_info['name']!r} has no local spec "
                        "(its weights would be discarded)"
                    )
                    ctx = spec.materialize()
                    xferdtensor(
                        None,
                        param_info["src_mesh_info"],
                        param_info["src_placements"],
                        DTensorRef(ctx.buf, param_info["global_shape"]),
                        param_info["dst_mesh_info"],
                        param_info["dst_placements"],
                        group,
                        stream,
                    )
                    if spec.post is not None:
                        spec.post(ctx)
                    del ctx
                # Whatever is still queued belongs to this stage's stream.
                self._flush_received_experts()
                event = torch.cuda.Event()
                event.record()
                events[idx] = event
        self._flush_received_experts()
        torch.cuda.synchronize()

    def _receive_and_load_misc_params(self) -> None:
        """Misc params: the collective path's packed broadcast + per-bucket reload."""
        misc_state_dict_info = self.misc_state_dict_info or {}
        if not misc_state_dict_info:
            return
        packed_broadcast_consumer(
            iterator=iter(misc_state_dict_info.items()),
            group=self.model_update_group,
            src=0,
            post_unpack_func=self._reload_bucket,
        )

    def nccl_reshard_refit(
        self, refit_timeout_s=None, *, drain: bool = True, recompute_kv: bool = False
    ) -> bool:
        """Receive a refit shard-to-shard, then the misc broadcast, then finalize.

        Same engine bracket as ``update_weights_from_collective``: ``drain``
        picks exclusive access at a step boundary (True) or the in-flight
        weight update (False, ``recompute_kv`` as there); the refit lifecycle
        (torch.compile unwrap, deferred weight processing, prefix-cache reset)
        is delegated to ``WorkerExtension`` via ``begin_weight_update`` /
        ``finalize_weight_update`` / ``finish_weight_update``.
        """
        refit_info = self.nccl_reshard_refit_info
        if refit_info is None:
            raise RuntimeError(
                "prepare_nccl_reshard_refit_info must run before the refit"
            )
        if not self.pp_comm_groups:
            raise RuntimeError("init_nccl_reshard_comm_group must run before the refit")
        if self.hf_to_local_param_map is None:
            self.hf_to_local_param_map = self._build_expert_local_param_map(refit_info)
        with self.engine.control_action(drain=drain):
            try:
                self.begin_weight_update()
                self._recv_bulk_params()
                torch.cuda.empty_cache()
                self._receive_and_load_misc_params()
                self._finish_refit(recompute_kv)
            except Exception:
                # Drop converted stacks of the failed refit; never reload them later.
                self._pending_bulk_reload = None
                self.engine.model_engine.model_loader.abort_update_weights()
                raise
        return True
