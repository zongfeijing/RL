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

import json
import pickle

import pytest

from nemo_rl.data.deepseek_v41_tokenizer import (
    get_deepseek_v41_tokenizer,
    maybe_wrap_deepseek_v41_tokenizer,
    should_use_deepseek_v41_chat_template,
)

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
EFFORT = (
    "<｜System｜>Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
USER = [{"role": "user", "content": "Solve 1+1."}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }
]
# Two calls in one assistant turn; the tool results arrive in the opposite order.
AGENT_CONVERSATION = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "List files."},
    {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "arguments": json.dumps({"command": "ls", "timeout": 5}),
                },
            },
            {
                "id": "b",
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "arguments": json.dumps({"command": "pwd"}),
                },
            },
        ],
    },
    {"role": "tool", "tool_call_id": "b", "content": "/repo"},
    {"role": "tool", "tool_call_id": "a", "content": "a.py"},
]


class DummyTokenizer:
    """Minimal stand-in for a fast tokenizer, picklable for __reduce__ tests."""

    vocab_size = 100

    def get_added_vocab(self):
        return {"<|extra_0|>": 100, "<|extra_1|>": 101}

    def encode(self, text, add_special_tokens=False, **kwargs):
        assert add_special_tokens is False
        max_length = kwargs.get("max_length")
        token_ids = [ord(character) % 256 for character in text]
        if max_length is not None and kwargs.get("truncation"):
            token_ids = token_ids[:max_length]
        return token_ids


@pytest.mark.parametrize(
    ("tokenizer_config", "expected"),
    [
        ({"chat_template": "deepseek_v41"}, True),
        ({"chat_template": "deepseek_v4"}, False),
        ({"chat_template": None}, False),
        ({}, False),
    ],
)
def test_should_use_deepseek_v41_chat_template(tokenizer_config, expected):
    assert should_use_deepseek_v41_chat_template(tokenizer_config) is expected


def test_maybe_wrap_leaves_other_templates_alone():
    tokenizer = DummyTokenizer()

    assert maybe_wrap_deepseek_v41_tokenizer(tokenizer, None) is tokenizer
    assert maybe_wrap_deepseek_v41_tokenizer(tokenizer, "deepseek_v4") is tokenizer
    wrapped = maybe_wrap_deepseek_v41_tokenizer(tokenizer, "deepseek_v41")
    assert wrapped is not tokenizer
    assert wrapped.apply_chat_template(USER, tokenize=False).startswith(BOS)


def test_chat_mode_closes_thinking_block():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(USER, tokenize=False)

    assert prompt == f"{BOS}<｜User｜>Solve 1+1.<｜Assistant｜></think>"


def test_thinking_mode_emits_default_reasoning_effort_and_opens_thinking_block():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(USER, tokenize=False, enable_thinking=True)

    assert prompt == (
        f"{BOS}{EFFORT.format(budget=75)}<｜User｜>Solve 1+1.<｜Assistant｜><think>"
    )


@pytest.mark.parametrize(
    ("reasoning_effort", "budget"),
    [("low", 50), ("medium", 62), ("high", 75), ("xhigh", 87), ("max", 100), (40, 40)],
)
def test_reasoning_effort_is_a_numeric_budget(reasoning_effort, budget):
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        USER, tokenize=False, thinking=True, reasoning_effort=reasoning_effort
    )

    assert prompt.startswith(BOS + EFFORT.format(budget=budget))


def test_invalid_reasoning_effort_is_rejected():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    with pytest.raises(ValueError, match="Invalid reasoning effort"):
        tokenizer.apply_chat_template(
            USER, tokenize=False, thinking=True, reasoning_effort="extreme"
        )


def test_developer_message_renders_as_system():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "developer", "content": "Be terse."}, *USER], tokenize=False
    )

    assert prompt == (
        f"{BOS}<｜System｜>Be terse.<｜User｜>Solve 1+1.<｜Assistant｜></think>"
    )


def test_chat_mode_history_matches_checkpoint_golden_output():
    """The checkpoint's encoding/tests/test_input_2.json and its golden output."""
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "reasoning_content": "The user said hello, I should greet back.",
            "content": "Hi there! How can I help you?",
        },
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "reasoning_content": "The user asks about the capital of France. It is Paris.",
            "content": "The capital of France is Paris.",
        },
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

    assert prompt == (
        f"{BOS}<｜System｜>You are a helpful assistant.<｜User｜>Hello"
        f"<｜Assistant｜></think>Hi there! How can I help you?{EOS}"
        "<｜User｜>What is the capital of France?"
        f"<｜Assistant｜></think>The capital of France is Paris.{EOS}"
    )


def test_tools_render_inside_the_leading_system_message():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        AGENT_CONVERSATION, tools=TOOLS, tokenize=False
    )

    # One system block: the caller's prompt, then the tool section.
    assert prompt.startswith(f"{BOS}<｜System｜>You are an agent.\n\n## Tools\n\n")
    assert prompt.count("<｜System｜>") == 1
    assert (
        '### Available Tool Schemas\n\n{"name": "execute_bash", "description": '
        '"Run a bash command.", "parameters": {"type": "object", "properties": '
        '{"command": {"type": "string"}, "timeout": {"type": "integer"}}, '
        '"required": ["command"]}}\n\n'
    ) in prompt


def test_tools_without_system_message_get_their_own_system_block():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(USER, tools=TOOLS, tokenize=False)

    assert prompt.startswith(f"{BOS}<｜System｜>\n\n## Tools\n\n")
    assert prompt.endswith("<｜User｜>Solve 1+1.<｜Assistant｜></think>")


def test_tool_calls_and_results_use_v41_dsml_and_call_order():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        AGENT_CONVERSATION, tools=TOOLS, tokenize=False
    )

    # The DSML tag names carry a leading space in V4.1; tool results fold into a
    # user turn, ordered by the calls rather than by arrival.
    assert prompt.endswith(
        "<｜User｜>List files.<｜Assistant｜></think>Looking.\n\n"
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="execute_bash">\n'
        '<｜DSML｜ parameter name="command" string="true">ls</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="timeout" string="false">5</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        '<｜DSML｜ invoke name="execute_bash">\n'
        '<｜DSML｜ parameter name="command" string="true">pwd</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        f"</｜DSML｜ calls>{EOS}"
        "<｜User｜><tool_result>a.py</tool_result>\n\n<tool_result>/repo</tool_result>"
        "<｜Assistant｜></think>"
    )


@pytest.mark.parametrize("thinking", [False, True])
def test_history_render_is_a_prefix_ending_on_the_same_eos(thinking):
    """What the on-policy prefix splice relies on: it locates turns by EOS count."""
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())
    kwargs = {"tools": TOOLS, "tokenize": False, "thinking": thinking}

    full = tokenizer.apply_chat_template(
        AGENT_CONVERSATION, add_generation_prompt=True, **kwargs
    )
    history = tokenizer.apply_chat_template(
        AGENT_CONVERSATION[:3], add_generation_prompt=False, **kwargs
    )

    assert full.count(EOS) == history.count(EOS) == 1
    assert history.endswith(EOS)
    assert full.startswith(history)


def test_tokenize_encodes_without_special_tokens_and_ignores_chat_template():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer())

    prompt = tokenizer.apply_chat_template(USER, tokenize=False)
    token_ids = tokenizer.apply_chat_template(USER, chat_template="deepseek_v41")
    truncated = tokenizer.apply_chat_template(USER, truncation=True, max_length=4)

    assert token_ids == [ord(character) % 256 for character in prompt]
    assert truncated == token_ids[:4]


def test_default_chat_template_kwargs_apply_and_can_be_overridden():
    tokenizer = get_deepseek_v41_tokenizer(
        DummyTokenizer(), {"enable_thinking": True, "reasoning_effort": "max"}
    )

    default = tokenizer.apply_chat_template(USER, tokenize=False)
    overridden = tokenizer.apply_chat_template(
        USER, tokenize=False, enable_thinking=False
    )

    assert default.startswith(BOS + EFFORT.format(budget=100))
    assert overridden == f"{BOS}<｜User｜>Solve 1+1.<｜Assistant｜></think>"


def test_wrapped_tokenizer_survives_pickling_and_reports_vocab():
    tokenizer = get_deepseek_v41_tokenizer(DummyTokenizer(), {"enable_thinking": True})

    restored = pickle.loads(pickle.dumps(tokenizer))

    assert restored.apply_chat_template(
        USER, tokenize=False
    ) == tokenizer.apply_chat_template(USER, tokenize=False)
    assert len(tokenizer) == 102
    assert tokenizer.get_added_vocab() == {"<|extra_0|>": 100, "<|extra_1|>": 101}
