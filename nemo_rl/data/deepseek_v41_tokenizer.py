# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import copy
from collections.abc import Mapping
from typing import Any

from nemo_rl.data.deepseek_v41_encoding import (
    OPENAI_REASONING_EFFORTS,
    _name_calls_as_declared,
    encode_messages,
)

DEEPSEEK_V41_CHAT_TEMPLATE = "deepseek_v41"


def should_use_deepseek_v41_chat_template(tokenizer_config: Mapping[str, Any]) -> bool:
    """Return whether the config explicitly requests the DeepSeek V4.1 renderer."""
    return tokenizer_config.get("chat_template") == DEEPSEEK_V41_CHAT_TEMPLATE


def get_deepseek_v41_tokenizer(
    tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None
) -> Any:
    """Wrap a fast tokenizer with the DeepSeek V4.1 chat renderer.

    The checkpoint ships no Jinja template: its prompt format is the Python
    encoder in its ``encoding/`` directory. ``apply_chat_template`` here follows
    TensorRT-LLM's ``DeepseekV41Tokenizer.apply_chat_template`` step for step, so
    a prompt rendered by NeMo RL is the one ``trtllm-serve`` would render.
    """
    dsv41_tokenizer = copy.copy(tokenizer)
    default_chat_template_kwargs = dict(chat_template_kwargs or {})
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            conversation: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str | list[int]:
            kwargs = {**default_chat_template_kwargs, **kwargs}
            # A Jinja override has no meaning here; callers that thread the
            # recipe's chat_template through pass the selecting sentinel.
            kwargs.pop("chat_template", None)
            thinking = kwargs.get("thinking", False) or kwargs.get(
                "enable_thinking", False
            )
            thinking_mode = "thinking" if thinking else "chat"
            # Any budget the checkpoint accepts is passed through; OpenAI's
            # level names are translated to the numeric budget.
            reasoning_effort = kwargs.get("reasoning_effort")
            if (
                isinstance(reasoning_effort, str)
                and reasoning_effort in OPENAI_REASONING_EFFORTS
            ):
                reasoning_effort = OPENAI_REASONING_EFFORTS[reasoning_effort]

            # V4.1 has no developer role; a developer message is a system message.
            messages = [
                {**message, "role": "system"}
                if message.get("role") == "developer"
                else message
                for message in conversation
            ]
            if tools:
                messages = _name_calls_as_declared(messages, tools)
                # Tools render from the system message that leads the
                # conversation, after its content: one system block, not a
                # tools-only block ahead of the caller's system prompt.
                if (
                    messages
                    and messages[0].get("role") == "system"
                    and not messages[0].get("tools")
                ):
                    messages[0] = {**messages[0], "tools": tools}
                else:
                    messages.insert(0, {"role": "system", "tools": tools})

            prompt = encode_messages(
                messages=messages,
                thinking_mode=thinking_mode,
                context=kwargs.get("context"),
                drop_thinking=kwargs.get("drop_thinking", True),
                add_default_bos_token=kwargs.get("add_default_bos_token", True),
                reasoning_effort=reasoning_effort,
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
            )
            if not kwargs.get("tokenize", True):
                return prompt

            tokenizer_kwargs = {
                key: kwargs[key]
                for key in ("truncation", "max_length")
                if key in kwargs
            }
            return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)

        def num_special_tokens_to_add(self, *args: Any, **kwargs: Any) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self) -> tuple[Any, tuple[Any, dict[str, Any]]]:
            return get_deepseek_v41_tokenizer, (
                tokenizer,
                default_chat_template_kwargs,
            )

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    dsv41_tokenizer.__class__ = _DeepseekV41Tokenizer
    return dsv41_tokenizer


def maybe_wrap_deepseek_v41_tokenizer(
    tokenizer: Any,
    chat_template: str | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Return ``tokenizer`` rendering DeepSeek V4.1 prompts when ``chat_template`` selects it."""
    if chat_template != DEEPSEEK_V41_CHAT_TEMPLATE:
        return tokenizer
    return get_deepseek_v41_tokenizer(tokenizer, chat_template_kwargs)
