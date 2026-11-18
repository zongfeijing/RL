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
"""OpenAI-compatible HTTP server wrapping ``tensorrt_llm.LLM``, serving /v1/chat/completions.

Returns prompt and generated token ids alongside per-token logprobs, and supports
Qwen3 tool calling, DeepSeekR1Parser reasoning, and prefix token splicing.

Under PD disaggregation this endpoint is *leg-aware*. A replica's
``OpenAIDisaggServer`` drives it twice per request:

* ``context_only`` -- prefill only. Returns the handshake and the prompt token
  ids, skipping all post-processing; see :func:`_context_leg_response`.
* ``generation_only`` -- decodes and post-processes as usual, but takes the
  prompt token ids the orchestrator relays from the context leg rather than
  rebuilding them, so the sequence matches the KV that was transferred.
"""

import asyncio
import functools
import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

from nemo_rl.models.generation.openai_server_utils import (
    splice_prefix_tokens,
)

logger = logging.getLogger(__name__)


@functools.cache
def _choice_declares_token_ids() -> bool:
    """Whether the installed TRT-LLM's chat choice model declares ``token_ids``.

    Cached because it is consulted once per generated response, and imported
    lazily because this module is also used on the aggregated path where
    tensorrt_llm need not be importable.
    """
    try:
        from tensorrt_llm.serve.openai_protocol import ChatCompletionResponseChoice
    except ImportError:
        return False
    return "token_ids" in ChatCompletionResponseChoice.model_fields


@functools.cache
def _chat_completion_request_cls() -> type:
    """``ChatCompletionRequest`` whose assistant ``tool_calls`` survive dumping.

    The OpenAI message types declare ``tool_calls`` as an ``Iterable``, which
    validates into a lazy ``ValidatorIterator``. For an assistant message that
    carries both ``reasoning`` and ``reasoning_content`` -- what NeMo-Gym sends
    -- the union resolves to TRT-LLM's ``ReasoningAssistantMessage`` and
    ``model_dump`` then emits ``tool_calls: []``, so the prompt is rendered
    without the conversation's earlier tool calls. Materializing the iterator
    right after validation keeps them. Deferred so importing this module does
    not require tensorrt_llm.
    """
    from pydantic import model_validator
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    class _ChatCompletionRequest(ChatCompletionRequest):
        @model_validator(mode="after")
        def _materialize_tool_calls(self) -> "_ChatCompletionRequest":
            for message in self.messages:
                tool_calls = message.get("tool_calls")
                if tool_calls is not None and not isinstance(tool_calls, list):
                    message["tool_calls"] = list(tool_calls)
            return self

    return _ChatCompletionRequest


def normalize_chat_inputs(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """``messages`` and ``tools`` as TRT-LLM's ``ChatCompletionRequest`` dumps them.

    Every path that renders the chat template goes through here, so the engine
    answering Gym directly and the disagg route render the same prompt. Key
    order matters: templates serialize each tool verbatim (``tool | tojson``),
    and validation reorders a tool's keys to the model's declaration order,
    ``{"type", "function"}`` -- what the disagg server hands its engines and
    what vLLM renders -- while Gym sends ``{"function", "type"}``.
    """
    dump = _chat_completion_request_cls()(
        # Required by the model but never rendered.
        model="",
        messages=messages,
        tools=tools,
    ).model_dump(mode="json", exclude_unset=True)
    return dump["messages"], dump["tools"]


async def build_spliced_prompt_ids(
    messages: list[dict],
    tools: "list[dict] | None",
    tokenizer: Any,
    model_config: Any,
    template_kwargs: dict[str, Any],
    eos_token_ids: Collection[int] | None = None,
    chat_template: str | None = None,
    required_prefix_token_ids: list[int] | None = None,
) -> list[int]:
    """Render the chat template and splice in the on-policy prefix.

    ``eos_token_ids`` is every token that can end a turn (see
    :func:`resolve_eos_token_ids`); ``None`` means only ``tokenizer.eos_token_id``.

    ``required_prefix_token_ids`` is the request's top-level field of that
    name: the exact prompt + generation ids of the previous call in the
    rollout. NeMo-Gym's model server sends it when the agent harness does not
    echo token ids on its messages (``supply_prefix_token_ids``) and then
    checks that the returned prompt starts with it. When given and non-empty
    it wins over the ids found on the messages, as in the vLLM server. It must
    be the call whose answer is the conversation's last assistant message: an
    assistant message a harness writes after the model's turn cannot be told
    from the model's own.

    The single source of truth for turning a conversation into engine prompt
    ids -- shared by the engine adapter's route and the disagg frontend's
    tokenizer (phase-2 ``frontend_tokenize``) so both produce IDENTICAL ids;
    a divergence between them would silently train on off-policy tokens.

    Raises ``ValueError`` for unparseable or multimodal messages. The
    tokenizer-heavy part runs in a thread: the HF fast tokenizer releases the
    GIL, so a 30k-token render stops serializing the caller's event loop.
    """
    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    conversation, mm_coroutine, *_ = parse_chat_messages_coroutines(
        messages, model_config
    )
    mm_data, mm_embeddings = await mm_coroutine
    if mm_data is not None or mm_embeddings is not None:
        raise ValueError(
            "NeMo-RL's TRT-LLM HTTP adapter does not support multimodal chat inputs"
        )

    def _sync() -> list[int]:
        return _spliced_prompt_ids(
            messages,
            conversation,
            tokenizer,
            tools,
            template_kwargs,
            eos_token_ids,
            chat_template=chat_template,
            required_prefix_token_ids=required_prefix_token_ids,
        )

    return await asyncio.to_thread(_sync)


# Sampling rate for shadow-validating frontend-supplied prompt ids on the
# context leg: the ctx adapter recomputes the ids from the messages it also
# received and logs a mismatch. Cheap insurance while frontend_tokenize is
# young; 0 disables.
_TOKENIZE_SHADOW_RATE = float(
    os.environ.get("NRL_TRTLLM_TOKENIZE_SHADOW_RATE", "0") or 0
)


def _parse_disaggregated_params(raw: Optional[dict[str, Any]]) -> tuple[Any, bool]:
    """The engine's ``DisaggregatedParams`` plus the orchestrator's b64 request.

    The body carries TRT-LLM's *wire* ``DisaggregatedParams``; the engine takes
    the LLM-level one (opaque_state is bytes there, base64 on the wire), so it
    is converted with TRT-LLM's own helper. The conversion copies only what the
    engine consumes, which drops ``return_prompt_token_ids_b64`` -- an
    orchestrator-to-context-worker instruction (``gen_tokids_ctxbytes``) that
    must therefore be read off the wire model before converting.

    Returns:
        ``(None, False)`` without disaggregated params, else the LLM-level
        params and whether the context leg should return
        ``prompt_token_ids_b64`` instead of the int list.
    """
    if raw is None:
        return None, False
    from tensorrt_llm.serve.openai_protocol import (
        DisaggregatedParams as WireDisaggregatedParams,
    )
    from tensorrt_llm.serve.openai_protocol import to_llm_disaggregated_params

    wire = WireDisaggregatedParams(**raw)
    return_b64 = bool(getattr(wire, "return_prompt_token_ids_b64", False))
    return to_llm_disaggregated_params(wire), return_b64


def _context_leg_response(
    model_name: str,
    prompt_token_ids: list[int],
    gen: Any,
    return_prompt_token_ids_b64: bool,
) -> Any:
    """Reply to a ``context_only`` request.

    Prefill materialised KV and at most one token; the disagg server only reads
    the handshake back off this response (plus the prompt token ids, so the
    generation server need not re-tokenize). Nothing here is user-visible, so
    the reasoning/tool/stop-token post-processing is skipped entirely.

    ``return_prompt_token_ids_b64`` comes from :func:`_parse_disaggregated_params`.
    """
    from fastapi.responses import JSONResponse
    from tensorrt_llm.serve.openai_protocol import to_disaggregated_params

    ctx_out = getattr(gen, "disaggregated_params", None)
    if ctx_out is None:
        raise RuntimeError(
            "context leg returned no disaggregated_params; the engine is most "
            "likely missing cache_transceiver_config"
        )

    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": gen.finish_reason,
                "disaggregated_params": to_disaggregated_params(ctx_out).model_dump(),
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": 0,
            "total_tokens": len(prompt_token_ids),
        },
    }

    # The orchestrator asks for the base64 int32 buffer when it wants to relay a
    # string instead of materialising the int list on its event loop.
    if return_prompt_token_ids_b64:
        import base64

        import numpy as np

        response["prompt_token_ids_b64"] = base64.b64encode(
            np.asarray(prompt_token_ids, dtype=np.int32).tobytes()
        ).decode("ascii")
    else:
        response["prompt_token_ids"] = prompt_token_ids

    return JSONResponse(content=response)


def resolve_eos_token_ids(
    tokenizer: Any,
    stop_token_ids: Collection[int] | None = None,
    generation_eos_token_ids: Any = None,
) -> set[int]:
    """Every token that ends a turn: configured stops, tokenizer EOS and generation-config EOS."""
    eos_token_ids: set[int] = set(stop_token_ids or ())
    for token_ids in (tokenizer.eos_token_id, generation_eos_token_ids):
        if isinstance(token_ids, int):
            eos_token_ids.add(token_ids)
        elif token_ids is not None:
            eos_token_ids.update(t for t in token_ids if isinstance(t, int))
    return eos_token_ids


def _request_matches_profile(body: dict[str, Any], profile: dict[str, Any]) -> bool:
    """Whether every sampling param the request pins equals *profile*'s value.

    Params the request leaves unset are not evidence against a profile. This
    server samples from the profile it selects and never from the request, so an
    omitted field simply takes that profile's value -- unlike vLLM, where an
    unset ``top_p`` is resolved from the model's ``generation_config.json`` and
    therefore has to be rejected outright (see ``vllm_worker_async.py``).

    Args:
        body: Decoded chat-completions request body.
        profile: Sampling profile to test, keyed ``temperature``/``top_p``/``top_k``.

    Returns:
        True when the request is compatible with *profile*.
    """
    for key in ("temperature", "top_p", "top_k"):
        requested = body.get(key)
        if requested is not None and requested != profile.get(key):
            return False
    return True


def _tokens_for_response_text(
    token_ids: list[int], stop_token_ids: Collection[int]
) -> list[int]:
    """``token_ids`` without trailing stop tokens, as a new list."""
    text_token_end = len(token_ids)
    while text_token_end and token_ids[text_token_end - 1] in stop_token_ids:
        text_token_end -= 1
    return token_ids[:text_token_end]


def _prompt_opens_reasoning(tokenizer: Any, token_ids: list[int]) -> bool:
    """Whether the prompt ends inside an open ``<think>`` block."""
    tail = tokenizer.decode(token_ids[-16:], skip_special_tokens=False)
    return tail.rstrip().endswith("<think>")


def _build_reasoning_parser(
    name: str,
    chat_template_kwargs: dict[str, Any],
    *,
    reasoning_at_start: bool = False,
) -> Any:
    from tensorrt_llm.llmapi.reasoning_parser import (
        DeepSeekR1Parser,
        ReasoningParserFactory,
    )

    parser = ReasoningParserFactory.create_reasoning_parser(name, chat_template_kwargs)
    if isinstance(parser, DeepSeekR1Parser):
        parser.reasoning_at_start = reasoning_at_start
        parser.in_reasoning = reasoning_at_start
    return parser


def _build_sampling_params(
    sampling_params_cls: Any,
    *,
    sampling_config: dict[str, Any],
    stop_token_ids: list[int] | None,
    max_tokens: int,
) -> Any:
    """Build the TRT-LLM sampling params for one HTTP rollout request.

    Mirrors the direct generate() path
    (``TrtllmAsyncGenerationWorkerImpl._build_sampling_params``) so both paths
    sample from the same distribution for a given generation config.

    Args:
        sampling_params_cls: ``tensorrt_llm.SamplingParams``, injected so this
            helper stays importable and testable without the TRT-LLM runtime.
        sampling_config: NeMo-RL generation config (temperature / top_p / top_k).
        stop_token_ids: Extra stop tokens from the generation config, if any.
        max_tokens: Output cap for this request, already clamped to the context window.

    Returns:
        A ``SamplingParams`` instance to hand to ``llm.generate_async``.
    """
    # TRT-LLM spells "no top-k restriction" as 0, the generation config as null.
    top_k_cfg = sampling_config["top_k"]
    stop_ids = list(stop_token_ids or [])
    return sampling_params_cls(
        temperature=float(sampling_config["temperature"]),
        top_p=float(sampling_config["top_p"]),
        top_k=int(top_k_cfg) if top_k_cfg is not None else 0,
        max_tokens=int(max_tokens),
        stop_token_ids=stop_ids or None,
        # Include generated stop tokens so generation_token_ids stays contiguous with the next turn's prompt.
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def create_app(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    *,
    max_seq_len: int,
    sampling_config: dict[str, Any],
    val_sampling_config: dict[str, Any],
    stop_token_ids: list[int] | None = None,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
) -> "FastAPI":
    """Build a FastAPI application backed by *llm* (``tensorrt_llm.LLM``).

    Args:
        llm: The ``tensorrt_llm.LLM`` engine to serve.
        tokenizer: Tokenizer matching *llm*, used for prompt construction.
        model_name: Model identifier echoed back on responses.
        max_seq_len: Context window of the engine being fronted.
        sampling_config: Train sampling profile, keyed
            ``temperature``/``top_p``/``top_k``.
        val_sampling_config: Validation sampling profile, same keys.
        stop_token_ids: Extra stop tokens to trim from generations.
        default_chat_template_kwargs: Server-side chat template defaults.
        tool_parser: Registered TRT-LLM tool parser name, or None to infer.
        reasoning_parser: Registered TRT-LLM reasoning parser name, or None.

    Returns:
        The configured FastAPI application.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # Per-request template kwargs override these defaults. Only what the config
    # sets, like the vLLM server: an unset key falls back to the chat template's
    # own default, and the disagg frontend's tokenizer renders the same way.
    _server_template_kwargs: dict[str, Any] = dict(default_chat_template_kwargs or {})

    # Use the configured parser or infer one from the model config.
    _tool_parser_name = _resolve_tool_parser_name(tool_parser, model_name)
    _tool_parser_instance = _build_tool_parser(_tool_parser_name)
    _parse_tool_calls = _make_parse_tool_calls(_tool_parser_instance)

    model_config = getattr(llm, "_hf_model_config", None)
    if model_config is None:
        raise RuntimeError(
            "TRT-LLM HTTP server requires the LLM's loaded Hugging Face model config"
        )

    if reasoning_parser is not None:
        _build_reasoning_parser(reasoning_parser, _server_template_kwargs)

    # Match TRT-LLM's effective EOS set. Stops are kept in the returned token
    # metadata but excluded from the parsed response text.
    generation_config = getattr(llm, "_generation_config", None)
    _eos_token_ids = resolve_eos_token_ids(
        tokenizer,
        stop_token_ids,
        generation_config.get("eos_token_id")
        if isinstance(generation_config, dict)
        else getattr(generation_config, "eos_token_id", None),
    )

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body: dict = await _decode_request_body(request)
        try:
            messages, tools = normalize_chat_inputs(
                body.get("messages", []), body.get("tools")
            )
        except ValueError as e:  # pydantic.ValidationError is a ValueError
            return JSONResponse(status_code=400, content={"error": str(e)})
        logprobs_requested = body.get("logprobs", False)

        # Conversation affinity for attention-DP routing; affects perf only.
        # Needs NVIDIA-NeMo/Gym#3582 to send conversation_params.
        _conv_id = (body.get("conversation_params") or {}).get("conversation_id")

        # The disagg server drives this endpoint twice per request -- once
        # context_only, once generation_only -- carrying the handshake between
        # them. opaque_state is bytes in the engine but base64 on the wire, so
        # convert with TRT-LLM's own helper rather than reproducing it.
        disagg_params, return_prompt_token_ids_b64 = _parse_disaggregated_params(
            body.get("disaggregated_params")
        )
        is_context_leg = getattr(disagg_params, "request_type", None) == "context_only"

        # The NeMo-RL generation config, not the request, sets sampling. Serve the
        # train profile, or the validation profile grpo.validate() stamps.
        if _request_matches_profile(body, sampling_config):
            active_sampling_config = sampling_config
        elif _request_matches_profile(body, val_sampling_config):
            active_sampling_config = val_sampling_config
        else:
            raise AssertionError(
                f"request sampling (temperature={body.get('temperature')!r}, "
                f"top_p={body.get('top_p')!r}, top_k={body.get('top_k')!r}) "
                f"matches neither the train sampling params ({sampling_config}) "
                f"nor the validation sampling params ({val_sampling_config})"
            )

        # Request kwargs override server defaults.
        per_request_kwargs: dict[str, Any] = body.get("chat_template_kwargs") or {}
        effective_template_kwargs = {**_server_template_kwargs, **per_request_kwargs}

        # NeMo-Gym's prefix supply: the previous call's prompt + generation
        # ids, for harnesses whose messages carry no token ids.
        required_prefix_token_ids = body.get("required_prefix_token_ids")
        if required_prefix_token_ids is not None and not (
            isinstance(required_prefix_token_ids, list)
            and all(
                isinstance(_t, int) and not isinstance(_t, bool)
                for _t in required_prefix_token_ids
            )
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "required_prefix_token_ids must be a list of integers"
                },
            )

        # b64 when gen_tokids_ctxbytes is set, an int array otherwise.
        supplied = body.get("prompt_token_ids") or body.get("prompt_token_ids_b64")
        if isinstance(supplied, str):
            # Same int32 buffer encoding openai_server.py uses on this hop.
            import base64

            import numpy as np

            supplied = np.frombuffer(
                base64.b64decode(supplied), dtype=np.int32
            ).tolist()

        if supplied:
            adj_prompt = list(supplied)
            # Shadow validation for frontend-tokenized ids (context leg only:
            # the generation leg's ids legitimately extend past the messages).
            if (
                is_context_leg
                and messages
                and _TOKENIZE_SHADOW_RATE > 0
                and random.random() < _TOKENIZE_SHADOW_RATE
            ):
                try:
                    recomputed = await build_spliced_prompt_ids(
                        messages,
                        tools,
                        tokenizer,
                        model_config,
                        effective_template_kwargs,
                        _eos_token_ids,
                        chat_template=chat_template,
                        required_prefix_token_ids=required_prefix_token_ids,
                    )
                    if recomputed != adj_prompt:
                        diff = next(
                            (
                                i
                                for i, (a, b) in enumerate(zip(adj_prompt, recomputed))
                                if a != b
                            ),
                            min(len(adj_prompt), len(recomputed)),
                        )
                        lo, hi = max(0, diff - 8), diff + 8
                        logger.error(
                            "frontend-tokenize SHADOW MISMATCH: supplied %d ids, "
                            "recomputed %d, first divergence at %d; "
                            "supplied[%d:%d]=%s (%r) vs recomputed=%s (%r); "
                            "template_kwargs=%r first_msg_role=%r",
                            len(adj_prompt),
                            len(recomputed),
                            diff,
                            lo,
                            hi,
                            adj_prompt[lo:hi],
                            tokenizer.decode(adj_prompt[lo:hi]),
                            recomputed[lo:hi],
                            tokenizer.decode(recomputed[lo:hi]),
                            effective_template_kwargs,
                            (messages[0] or {}).get("role") if messages else None,
                        )
                except Exception as e:  # noqa: BLE001 - shadow must never fail serving
                    logger.error("frontend-tokenize shadow recompute failed: %s", e)
        else:
            try:
                adj_prompt = await build_spliced_prompt_ids(
                    messages,
                    tools,
                    tokenizer,
                    model_config,
                    effective_template_kwargs,
                    _eos_token_ids,
                    chat_template=chat_template,
                    required_prefix_token_ids=required_prefix_token_ids,
                )
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        # Infer parser state from the exact engine prompt.
        _active_reasoning_parser = (
            _build_reasoning_parser(
                reasoning_parser,
                effective_template_kwargs,
                reasoning_at_start=_prompt_opens_reasoning(tokenizer, adj_prompt),
            )
            if reasoning_parser is not None
            else None
        )

        max_tokens_requested = (
            body.get("max_tokens") or body.get("max_completion_tokens") or max_seq_len
        )
        remaining_ctx = max(0, max_seq_len - len(adj_prompt))

        # Return HTTP 400 on context exhaustion.
        if remaining_ctx == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"context length exceeded: prompt ({len(adj_prompt)} tokens) exhausted context window ({max_seq_len})"
                },
            )

        max_tokens = min(int(max_tokens_requested), remaining_ctx)

        from tensorrt_llm import SamplingParams as TrtSamplingParams
        from tensorrt_llm.executor.utils import RequestError

        # Build sampling parameters from the selected train or validation profile.
        sampling = _build_sampling_params(
            TrtSamplingParams,
            sampling_config=active_sampling_config,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
        )
        try:
            _conv_params = None
            if _conv_id:
                from tensorrt_llm.conversation_params import ConversationParams

                _conv_params = ConversationParams(conversation_id=str(_conv_id))
            output = await llm.generate_async(
                {"prompt_token_ids": adj_prompt},
                sampling_params=sampling,
                disaggregated_params=disagg_params,
                conversation_params=_conv_params,
            )
        except RequestError as e:
            err = str(e)
            if "max_seq_len" in err or "max_num_tokens" in err:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"context length exceeded: {err}"},
                )
            raise

        gen = output.outputs[0]

        if is_context_leg:
            # Prefill produced KV and at most one token. Everything downstream
            # -- reasoning parsing, tool parsing, stop-token trimming -- is for
            # the completed generation, so skip it and hand the disagg server
            # just what it needs to build the generation leg.
            return _context_leg_response(
                model_name, adj_prompt, gen, return_prompt_token_ids_b64
            )

        gen_token_ids = list(gen.token_ids)

        gen_logprobs: list[float] = []
        if gen.logprobs:
            # TRT-LLM returns floats in simple format and token-indexed dicts otherwise.
            for token_id, lp in zip(gen_token_ids, gen.logprobs, strict=True):
                if isinstance(lp, (int, float)):
                    gen_logprobs.append(float(lp))
                elif isinstance(lp, dict):
                    gen_logprobs.append(float(lp[token_id].logprob))
                else:
                    raise TypeError(f"Unsupported TRT-LLM logprob type: {type(lp)}")

        # Parse without trailing stops; the returned token ids and logprobs keep them.
        text_token_ids = _tokens_for_response_text(gen_token_ids, _eos_token_ids)
        gen_text = tokenizer.decode(text_token_ids, skip_special_tokens=False)

        finish_reason = "stop"
        if gen.finish_reason is not None:
            fr = str(gen.finish_reason).lower()
            if "length" in fr:
                finish_reason = "length"

        # Split reasoning from answer, if a parser is configured.
        if _active_reasoning_parser is not None:
            parsed = _active_reasoning_parser.parse(gen_text)
            reasoning_content: str = parsed.reasoning_content
            answer_text: str = parsed.content
        else:
            reasoning_content = ""
            answer_text = gen_text

        if tools:
            content_text, parsed_tool_calls = _parse_tool_calls(answer_text, tools)
        else:
            content_text, parsed_tool_calls = answer_text, []

        if parsed_tool_calls:
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": content_text or None,
                "reasoning_content": reasoning_content,
                "tool_calls": parsed_tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            msg_dict = {
                "role": "assistant",
                "content": answer_text,
                "reasoning_content": reasoning_content,
            }

        # NeMo-Gym reads the rollout fields off the *message*
        # (nemo_rl/environments/nemo_gym.py: a message without
        # generation_token_ids is skipped outright, so a miss loses the whole
        # turn's training data silently). Aggregated serving answers Gym
        # directly, so attach them here.
        #
        # Not under disaggregation: there the reply is re-validated by the
        # disagg server against ChatMessage, which is extra="forbid" and would
        # 400 on these. They ride the declared fields instead
        # (choices[].token_ids, prompt_token_ids, logprobs) and the disagg
        # server's outbound adaptor re-attaches them to the message before Gym
        # ever sees it -- see trtllm_disagg_server.attach_rollout_fields.
        if disagg_params is None:
            msg_dict["prompt_token_ids"] = adj_prompt
            msg_dict["generation_token_ids"] = gen_token_ids
            msg_dict["generation_log_probs"] = gen_logprobs

        response: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": msg_dict,
                    "finish_reason": finish_reason,
                }
            ],
            # Declared on ChatCompletionResponse precisely so a generation
            # server need not re-tokenize the prompt.
            "prompt_token_ids": adj_prompt,
            "usage": {
                "prompt_tokens": len(adj_prompt),
                "completion_tokens": len(gen_token_ids),
                "total_tokens": len(adj_prompt) + len(gen_token_ids),
            },
        }

        # Generated token ids get their own declared field only on a TRT-LLM
        # build whose ChatCompletionResponseChoice carries one
        # (CompletionResponseChoice already does). Emitting it unconditionally
        # would 400 at the disagg server's extra="forbid" re-validation on every
        # other build -- which is precisely where the token_id:N logprobs
        # fallback below is supposed to take over, so the write has to be gated
        # the same way the read is.
        if _choice_declares_token_ids():
            response["choices"][0]["token_ids"] = gen_token_ids

        if logprobs_requested and gen_logprobs:
            # `token` carries the id rather than the decoded text when asked.
            # On TRT-LLM builds whose ChatCompletionResponseChoice lacks
            # token_ids (declared since f94eeb37), this declared string field is
            # how the ids survive the disagg server's strict re-validation. Same
            # encoding vLLM uses, which is what NeMo-Gym already parses.
            # Under disaggregation the flag never survives the frontend's
            # gym-field strip, yet ids are exactly what rides this channel
            # (the disagg adaptor parses token_id:N back out) -- and decoding
            # each generated token individually is per-token tokenizer work
            # nobody reads. Force the id encoding on any disagg request.
            as_ids = (
                bool(body.get("return_tokens_as_token_ids"))
                or disagg_params is not None
            )
            response["choices"][0]["logprobs"] = {
                "content": [
                    {
                        "token": (
                            f"token_id:{tid}" if as_ids else tokenizer.decode([tid])
                        ),
                        "logprob": lp,
                        "bytes": None,
                        "top_logprobs": [],
                    }
                    for tid, lp in zip(gen_token_ids, gen_logprobs)
                ]
            }

        return JSONResponse(content=response)

    return app


# ---------------------------------------------------------------------------
#  Tool-call parser factory — delegates to TRT-LLM's registered parsers

# ---------------------------------------------------------------------------


def _resolve_tool_parser_name(configured_name: str | None, model_name: str) -> str:
    """Resolve the configured parser or infer it from the model."""
    if configured_name:
        return configured_name

    from tensorrt_llm.serve.tool_parser.tool_parser_factory import (
        resolve_auto_tool_parser,
    )

    resolved_name = resolve_auto_tool_parser(model_name)
    if resolved_name:
        return resolved_name

    raise ValueError(
        f"Could not infer a tool parser from {model_name!r}; "
        "set trtllm_cfg.tool_parser explicitly."
    )


# TRT-LLM's disagg orchestrator encodes the forwarded worker request with
# msgspec msgpack and flags it with this header, deliberately leaving
# Content-Type as application/json so FastAPI still routes the body
# (tensorrt_llm/serve/openai_client.py: MSGPACK_HEADERS). A plain
# ``await request.json()`` therefore dies on the first byte of every
# orchestrator-forwarded request -- 0x8e is a msgpack fixmap header, not UTF-8.
# Only the request leg is encoded this way: the client reads Content-Type off
# the response solely to reject an unexpected event-stream, so responses stay
# ordinary JSON.
_MSGPACK_REQUEST_HEADER = "x-trtllm-msgpack"


async def _decode_request_body(request: "Request") -> dict[str, Any]:
    """Body as a dict, honoring the orchestrator's msgpack framing."""
    if request.headers.get(_MSGPACK_REQUEST_HEADER) != "1":
        return await request.json()

    import msgspec

    return msgspec.msgpack.decode(await request.body())


def _build_tool_parser(name: str) -> Any:
    """Instantiate a TRT-LLM tool parser by registered name."""
    # Import lazily and preserve import errors.
    from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

    return ToolParserFactory.create_tool_parser(name)


def _make_parse_tool_calls(tool_parser_instance: Any) -> Any:
    """Return a tool-call parser bound to a specific parser instance."""

    def _parse(text: str, tools: list[dict] | None) -> tuple[str, list[dict[str, Any]]]:
        if not text or not tool_parser_instance.has_tool_call(text):
            return text, []

        # Preserve argument types with TRT-LLM typed tool schemas.
        typed_tools: list[Any] = []
        if tools:
            from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam

            typed_tools = [ChatCompletionToolsParam(**tool) for tool in tools]

        result = tool_parser_instance.detect_and_parse(text, typed_tools)
        calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": item.parameters,
                },
            }
            for item in (result.calls or [])
        ]
        if not calls:
            return text, []
        return result.normal_text.strip(), calls

    return _parse


# ---------------------------------------------------------------------------
#  Prompt construction

# ---------------------------------------------------------------------------


def _to_int_ids(enc: Any) -> list[int]:
    """Coerce chat-template output to a flat list[int]."""
    if hasattr(enc, "input_ids"):  # transformers v5 BatchEncoding
        enc = enc.input_ids
    if len(enc) and isinstance(enc[0], (list, tuple)):  # batch-of-one nesting
        enc = enc[0]
    return [int(t) for t in enc]


def _build_prompt_token_ids(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
    default_template_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
) -> list[int]:
    """Convert chat messages to token IDs via apply_chat_template (full retokenisation each turn).

    Full retokenisation avoids the gen_token_ids double-counting bug (~4000 tok/turn explosion
    that exhausted context at turn ~32 when prefix accumulation was used).
    """
    template_kwargs: dict[str, Any] = {
        **(default_template_kwargs or {}),
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if tools:
        template_kwargs["tools"] = tools
    if chat_template is not None:
        template_kwargs["chat_template"] = chat_template
    return _to_int_ids(tokenizer.apply_chat_template(messages, **template_kwargs))


def _compute_splice_inputs(
    raw_messages: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    default_template_kwargs: dict[str, Any],
    chat_template: str | None = None,
    required_prefix_token_ids: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Return preserved and rendered token IDs for the on-policy prefix splice.

    An explicit, non-empty ``required_prefix_token_ids`` (the request's
    top-level field) is used as is; otherwise the prefix is rebuilt from the
    last assistant message that carries token ids.
    """
    required_prefix_ids: list[int] = []
    if required_prefix_token_ids:
        required_prefix_ids = _to_int_ids(required_prefix_token_ids)
    else:
        for _m in reversed(raw_messages):
            if _m.get("role") == "assistant" and "prompt_token_ids" in _m:
                required_prefix_ids = list(_m["prompt_token_ids"]) + list(
                    _m.get("generation_token_ids") or []
                )
                break

    _last_asst_idx = next(
        (
            i
            for i in reversed(range(len(conversation)))
            if conversation[i].get("role") == "assistant"
        ),
        None,
    )
    _msgs_to_last_asst = (
        conversation[: _last_asst_idx + 1]
        if _last_asst_idx is not None
        else conversation
    )
    _prefix_tkw: dict[str, Any] = {
        **default_template_kwargs,
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if tools:
        _prefix_tkw["tools"] = tools
    if chat_template is not None:
        _prefix_tkw["chat_template"] = chat_template
    template_prefix_ids = _to_int_ids(
        tokenizer.apply_chat_template(_msgs_to_last_asst, **_prefix_tkw)
    )
    return required_prefix_ids, template_prefix_ids


def _spliced_prompt_ids(
    raw_messages: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any],
    eos_token_ids: Collection[int] | None,
    chat_template: str | None = None,
    required_prefix_token_ids: list[int] | None = None,
) -> list[int]:
    """Render ``conversation`` and splice in the on-policy prefix (blocking)."""
    # Full retokenization avoids accumulating generation token IDs twice.
    prompt_token_ids = _build_prompt_token_ids(
        conversation,
        tokenizer,
        tools=tools,
        default_template_kwargs=template_kwargs,
        chat_template=chat_template,
    )
    # Empty required_prefix_ids on turn one returns the template unchanged.
    required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
        raw_messages,
        conversation,
        tokenizer,
        tools,
        template_kwargs,
        chat_template=chat_template,
        required_prefix_token_ids=required_prefix_token_ids,
    )
    closing_marker = None
    if required_prefix_ids and _prefix_ends_at_last_assistant(
        raw_messages, required_prefix_token_ids
    ):
        closing_marker = _turn_closing_marker(
            required_prefix_ids,
            template_prefix_ids,
            prompt_token_ids,
            _end_of_turn_ids(eos_token_ids, tokenizer),
            tokenizer,
        )
    if closing_marker is not None:
        template_prefix_ids = template_prefix_ids + [closing_marker]
    splice = splice_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=required_prefix_ids,
        template_prefix_token_ids=template_prefix_ids,
        template_token_ids=prompt_token_ids,
        eos_token_id=eos_token_ids or None,
    )
    token_ids = splice.token_ids
    boundary = splice.model_cut_end
    if closing_marker is not None and token_ids[boundary] != closing_marker:
        # The model ended its turn on another end-of-turn token than the one
        # that opens the next message (say <|user|> or <|endoftext|> before a
        # tool block). Its token stays, the prefix is verbatim, and the
        # message keeps its own marker instead of sitting under the model's.
        token_ids = (
            token_ids[: boundary + 1] + [closing_marker] + token_ids[boundary + 1 :]
        )
    return token_ids


# A longer run of tokens after the last end-of-turn token is not whitespace;
# it bounds the decode in _assistant_turn_left_open.
_MAX_TRAILING_WHITESPACE_TOKENS = 8


def _end_of_turn_ids(
    eos_token_ids: Collection[int] | int | None, tokenizer: Any
) -> frozenset[int]:
    """The end-of-turn set the way ``splice_prefix_tokens`` reads its ``eos_token_id``.

    ``None`` or an empty collection means the tokenizer's own EOS, which may
    be one id, several, or missing.
    """
    if eos_token_ids is None or (
        not isinstance(eos_token_ids, int) and not eos_token_ids
    ):
        eos_token_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_token_ids is None:
        return frozenset()
    if isinstance(eos_token_ids, int):
        return frozenset((eos_token_ids,))
    return frozenset(eos_token_ids)


def _prefix_ends_at_last_assistant(
    raw_messages: list[dict[str, Any]],
    required_prefix_token_ids: list[int] | None,
) -> bool:
    """Whether the on-policy prefix ends with the conversation's last assistant turn.

    The request's top-level prefix is the previous call, whose answer is that
    turn. Ids echoed on the messages may sit on an earlier assistant message
    than the last one; the rendered prefix then covers more than the model's
    tokens, and the splice is left as it was. That includes one turn echoed
    as several assistant messages with the ids not on the last of them: the
    messages after the ids cannot be told from ones the harness wrote.
    """
    if required_prefix_token_ids:
        return True
    last_assistant = next(
        (m for m in reversed(raw_messages) if m.get("role") == "assistant"), None
    )
    return last_assistant is not None and "prompt_token_ids" in last_assistant


def _assistant_turn_left_open(
    template_prefix_ids: list[int], eos_token_ids: frozenset[int], tokenizer: Any
) -> bool:
    """Whether the render through the last assistant message stops inside that turn.

    ChatML-style templates close every message with an end-of-turn token, so
    the render ends on one, at most followed by whitespace. GLM-style templates
    have no such token: the next message's role marker closes the turn, and
    the render ends on the assistant's own text.
    """
    last_eos = next(
        (
            i
            for i in range(len(template_prefix_ids) - 1, -1, -1)
            if template_prefix_ids[i] in eos_token_ids
        ),
        -1,
    )
    tail = template_prefix_ids[last_eos + 1 :]
    if not tail:
        return False
    if len(tail) > _MAX_TRAILING_WHITESPACE_TOKENS:
        return True
    return bool(tokenizer.decode(tail, skip_special_tokens=False).strip())


def _turn_closing_marker(
    model_prefix_ids: list[int],
    template_prefix_ids: list[int],
    prompt_token_ids: list[int],
    eos_token_ids: frozenset[int],
    tokenizer: Any,
) -> int | None:
    """The token that closes the last assistant turn, if the template leaves it open.

    ``splice_prefix_tokens`` finds the end of the model's last turn in the
    full render by counting the end-of-turn tokens of the rendered prefix. That
    lands on the right token when the template closes the assistant message
    itself (ChatML's ``<|im_end|>``). GLM's template does not: an assistant
    turn ends at the next message's ``<|user|>`` or ``<|observation|>``, which
    are also the tokens the model emits to stop. The rendered prefix then ends
    inside the turn, its last end-of-turn token is the marker *before* the
    turn, and the splice appends the preceding user/tool block and a re-render
    of the assistant turn after the model's own tokens: every call after the
    first carries the previous exchange twice.

    When the rendered prefix stops inside the turn, the token that closes it
    is the next end-of-turn token of the full render; counted into the
    rendered prefix it moves the cut to the real boundary. It is found by
    count as well, not by position: templates may render earlier turns
    differently once more messages follow (GLM can drop the reasoning of turns
    before the last user message).

    Two checks keep the marker from cutting messages out of the prompt.

    The model's tokens must show every turn the rendered prefix shows. Both
    hold one end-of-turn token per turn before the last assistant message
    when the prefix ends with that message; the model's may hold more (a stop
    token kept next to a marker). Fewer means messages lie between the
    model's turn and the last assistant message, a tool result followed by an
    assistant message the harness wrote, say. (So does the text of an
    end-of-turn token in what the model wrote, which the template renders as
    that token.) The model's final token is left out of the count: an
    end-of-turn token there ends the very turn in question.

    And the open turn must be all that precedes the marker in the full
    render. Anything more means it ends a later message: a message whose role
    opens with no end-of-turn token came in between (a system message), or
    the template closed the assistant message with a token outside the
    end-of-turn set, or rendered a generation prompt after it.

    ``None`` keeps the splice as it was: templates that close the assistant
    message, an assistant message that is last, and the cases the two checks
    turn down (the previous exchange is then still replayed; nothing is
    lost). So does an answer whose own text ends with the text of an
    end-of-turn token, which renders like a closed turn.
    """
    if not eos_token_ids:
        return None
    if not _assistant_turn_left_open(template_prefix_ids, eos_token_ids, tokenizer):
        return None

    rendered_turn_ends = sum(tid in eos_token_ids for tid in template_prefix_ids)
    prefix_turn_ends = sum(tid in eos_token_ids for tid in model_prefix_ids[:-1])
    if prefix_turn_ends < rendered_turn_ends:
        _report_splice_fallback_once(
            "turns",
            "the render through it shows %d end-of-turn tokens and the on-policy "
            "prefix holds %d: messages lie between the model's turn and the last "
            "assistant message, or the model wrote the text of such a token.",
            rendered_turn_ends,
            prefix_turn_ends,
        )
        return None

    seen = 0
    segment_start = 0
    for position, tid in enumerate(prompt_token_ids):
        if tid in eos_token_ids:
            if seen == rendered_turn_ends:
                break
            seen += 1
            segment_start = position + 1
    else:
        return None

    # Since the last end-of-turn token the full render may show less than the
    # rendered prefix does (reasoning dropped from history), never more.
    tail_start = len(template_prefix_ids)
    while tail_start and template_prefix_ids[tail_start - 1] not in eos_token_ids:
        tail_start -= 1
    open_turn = iter(template_prefix_ids[tail_start:])
    if not all(t in open_turn for t in prompt_token_ids[segment_start:position]):
        _report_splice_fallback_once(
            "position",
            "the next end-of-turn token of the full render does not directly "
            "follow that turn (%d tokens before it, %d in the turn as rendered "
            "alone).",
            position - segment_start,
            len(template_prefix_ids) - tail_start,
        )
        return None
    return tid


# Reasons already reported by _report_splice_fallback_once.
_reported_splice_fallbacks: set[str] = set()


def _report_splice_fallback_once(reason: str, finding: str, *args: Any) -> None:
    """Log, once per reason, that an open assistant turn keeps the count-based cut.

    Nothing else shows it: the prompt still starts with the previous call.
    """
    if reason in _reported_splice_fallbacks:
        return
    _reported_splice_fallbacks.add(reason)
    logger.warning(
        "Prefix splice: the chat template leaves the last assistant turn open, "
        "but " + finding + " Keeping the count-based cut, which repeats the "
        "previous exchange in the prompt. Logged once per process.",
        *args,
    )


# ---------------------------------------------------------------------------
#  Server lifecycle

# ---------------------------------------------------------------------------


def start_server(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    *,
    max_seq_len: int,
    sampling_config: dict[str, Any],
    val_sampling_config: dict[str, Any],
    stop_token_ids: list[int] | None = None,
    host: str = "0.0.0.0",
    port: int = 0,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
) -> "tuple[threading.Thread, str, Any]":
    """Start the HTTP server in a daemon thread and return (thread, base_url, server).

    Args:
        llm: The ``tensorrt_llm.LLM`` engine to serve.
        tokenizer: Tokenizer matching *llm*.
        model_name: Model identifier echoed back on responses.
        max_seq_len: Context window of the engine being fronted.
        sampling_config: Train sampling profile.
        val_sampling_config: Validation sampling profile.
        stop_token_ids: Extra stop tokens to trim from generations.
        host: Bind address.
        port: Bind port; 0 picks a free one.
        default_chat_template_kwargs: Server-side chat template defaults.
        tool_parser: Registered TRT-LLM tool parser name, or None to infer.
        reasoning_parser: Registered TRT-LLM reasoning parser name, or None.

    Returns:
        Tuple of (server thread, base URL, uvicorn server).
    """
    import uvicorn

    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local,
        _get_node_ip_local,
    )

    if port == 0:
        port = _get_free_port_local()

    node_ip = _get_node_ip_local()
    base_url = f"http://{node_ip}:{port}/v1"

    app = create_app(
        llm,
        tokenizer,
        model_name,
        max_seq_len=max_seq_len,
        sampling_config=sampling_config,
        val_sampling_config=val_sampling_config,
        stop_token_ids=stop_token_ids,
        default_chat_template_kwargs=default_chat_template_kwargs,
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        chat_template=chat_template,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info("TRT-LLM HTTP server starting on %s", base_url)

    return thread, base_url, server
