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

import asyncio
import base64
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens
from nemo_rl.models.generation.trtllm.trtllm_http_server import (
    _build_prompt_token_ids,
    _build_sampling_params,
    _compute_splice_inputs,
    _context_leg_response,
    _decode_request_body,
    _make_parse_tool_calls,
    _parse_disaggregated_params,
    _prompt_opens_reasoning,
    _resolve_tool_parser_name,
    _spliced_prompt_ids,
    _tokens_for_response_text,
    _turn_closing_marker,
)

pytestmark = pytest.mark.trtllm


class _FakeToolParser:
    def __init__(self, *, calls):
        self.calls = calls

    def has_tool_call(self, text: str) -> bool:
        return "<tool_call>" in text

    def detect_and_parse(self, text: str, tools: list):
        del text, tools
        return SimpleNamespace(
            normal_text="visible before visible after",
            calls=self.calls,
        )


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]


def test_http_sampling_params_match_the_direct_path():
    """The HTTP rollout path must sample exactly like the direct generate() path."""
    sampling_params_cls = MagicMock()

    _build_sampling_params(
        sampling_params_cls,
        sampling_config={"temperature": 0.8, "top_p": 0.9, "top_k": 20},
        stop_token_ids=[9],
        max_tokens=4,
    )

    sampling_params_cls.assert_called_once_with(
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        max_tokens=4,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def test_http_sampling_params_map_null_top_k_and_empty_stop_tokens():
    """null top_k reaches TRT-LLM as 0, and unset stop tokens as None, not []."""
    sampling_params_cls = MagicMock()

    _build_sampling_params(
        sampling_params_cls,
        sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        stop_token_ids=None,
        max_tokens=8,
    )

    sampling_params_cls.assert_called_once_with(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_tokens=8,
        stop_token_ids=None,
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def test_response_text_excludes_stops_without_mutating_training_tokens():
    generation_token_ids = [10, 20, 2, 3]

    text_token_ids = _tokens_for_response_text(generation_token_ids, {2, 3})

    assert text_token_ids == [10, 20]
    assert generation_token_ids == [10, 20, 2, 3]


def test_response_text_returns_all_tokens_when_unterminated():
    generation_token_ids = [10, 20]

    assert _tokens_for_response_text(generation_token_ids, {2, 3}) == [10, 20]


def test_response_text_is_empty_when_generation_is_all_stops():
    assert _tokens_for_response_text([2, 3], {2, 3}) == []


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(
            {1: "assistant\n", 2: "<think>\n", 3: "\n\n</think>\n\n"}[t]
            for t in token_ids
        )


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([1], False),
        ([1, 2], True),
        ([1, 2, 3], False),
    ],
)
def test_prompt_opens_reasoning(token_ids, expected):
    assert _prompt_opens_reasoning(_FakeTokenizer(), token_ids) is expected


def test_explicit_tool_parser_overrides_model_auto_resolution():
    assert _resolve_tool_parser_name("qwen3_coder", "/missing/model") == "qwen3_coder"


def test_tool_parser_is_inferred_from_local_hf_config(tmp_path):
    pytest.importorskip("tensorrt_llm.serve.tool_parser.tool_parser_factory")
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_moe"}')

    assert _resolve_tool_parser_name(None, str(tmp_path)) == "qwen3"


def test_tool_parser_resolution_fails_loudly(monkeypatch):
    factory = pytest.importorskip("tensorrt_llm.serve.tool_parser.tool_parser_factory")
    monkeypatch.setattr(factory, "resolve_auto_tool_parser", lambda _: None)

    with pytest.raises(ValueError, match="set trtllm_cfg.tool_parser explicitly"):
        _resolve_tool_parser_name(None, "/missing/model")


def test_parse_tool_calls_uses_parser_normal_text():
    parser = _FakeToolParser(
        calls=[
            SimpleNamespace(
                name="run_command",
                parameters='{"command": "pwd"}',
            )
        ]
    )
    parse_tool_calls = _make_parse_tool_calls(parser)

    content, calls = parse_tool_calls(
        "visible before<tool_call>xml</tool_call>visible after",
        tools=None,
    )

    assert content == "visible before visible after"
    assert calls[0]["function"] == {
        "name": "run_command",
        "arguments": '{"command": "pwd"}',
    }


def test_parse_tool_calls_preserves_text_when_parser_returns_no_calls():
    parser = _FakeToolParser(calls=[])
    parse_tool_calls = _make_parse_tool_calls(parser)
    text = "visible before<tool_call>malformed</tool_call>visible after"

    content, calls = parse_tool_calls(text, tools=None)

    assert content == text
    assert calls == []


def _tool_call_messages():
    return [
        {"role": "user", "content": "List the current directory."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "pwd"}',
                    },
                }
            ],
            "prompt_token_ids": [10],
            "generation_token_ids": [11],
            "generation_log_probs": [-0.1],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "/workspace"},
    ]


def _parse_with_trtllm(messages):
    chat_utils = pytest.importorskip("tensorrt_llm.serve.chat_utils")
    transformers = pytest.importorskip("transformers")
    conversation, mm_coroutine, *_ = chat_utils.parse_chat_messages_coroutines(
        messages, transformers.PretrainedConfig()
    )
    assert asyncio.run(mm_coroutine) == (None, None)
    return conversation


def test_trtllm_parser_normalizes_openai_tool_arguments_without_mutation():
    messages = _tool_call_messages()

    conversation = _parse_with_trtllm(messages)

    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pwd"
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command": "pwd"}'
    )
    assert "prompt_token_ids" not in conversation[1]
    assert "generation_token_ids" not in conversation[1]
    assert "generation_log_probs" not in conversation[1]


def test_prompt_and_prefix_render_use_normalized_tool_arguments():
    messages = _tool_call_messages()
    conversation = _parse_with_trtllm(messages)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {"type": "object"},
            },
        }
    ]
    tokenizer = _RecordingTokenizer()

    assert _build_prompt_token_ids(conversation, tokenizer, tools=tools) == [1, 2, 3]
    required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
        messages, conversation, tokenizer, tools, {}
    )

    assert required_prefix_ids == [10, 11]
    assert template_prefix_ids == [1, 2, 3]
    assert len(tokenizer.calls) == 2
    for rendered_messages, kwargs in tokenizer.calls:
        assert rendered_messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "command": "pwd"
        }
        assert kwargs["tools"] == tools


class _SpliceTokenizer:
    """Renders a two-call conversation the way a harness that echoes no ids does.

    The full prompt re-tokenizes the first answer as 1001; the model had
    actually produced 220, 17. Token 2 ends a turn.
    """

    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        del messages
        if kwargs["add_generation_prompt"]:
            return [11, 12, 13, 40, 41, 1001, 2, 50, 51, 60]
        return [11, 12, 13, 40, 41, 1001, 2]


_HARNESS_MESSAGES = [
    {"role": "user", "content": "2 + 2?"},
    {"role": "assistant", "content": " 4"},
    {"role": "user", "content": "and 3 + 3?"},
]


def test_an_explicit_required_prefix_wins_over_the_ids_on_the_messages():
    messages = _tool_call_messages()

    required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
        messages,
        messages,
        _RecordingTokenizer(),
        None,
        {},
        required_prefix_token_ids=[7, 8, 9],
    )

    assert required_prefix_ids == [7, 8, 9]
    assert template_prefix_ids == [1, 2, 3]


@pytest.mark.parametrize("explicit", [None, []])
def test_a_missing_or_empty_required_prefix_falls_back_to_the_message_ids(explicit):
    messages = _tool_call_messages()

    required_prefix_ids, _ = _compute_splice_inputs(
        messages,
        messages,
        _RecordingTokenizer(),
        None,
        {},
        required_prefix_token_ids=explicit,
    )

    assert required_prefix_ids == [10, 11]


def test_the_required_prefix_reaches_the_prompt_when_messages_carry_no_ids():
    # A codex or OpenCode harness sends plain messages, so the previous call's
    # exact tokens arrive only in the top-level field. NeMo-Gym then checks
    # that the returned prompt starts with them.
    tokenizer = _SpliceTokenizer()
    previous_call = [11, 12, 13, 40, 41, 220, 17, 2]

    without, _ = _compute_splice_inputs(
        _HARNESS_MESSAGES, _HARNESS_MESSAGES, tokenizer, None, {}
    )
    assert without == []

    required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
        _HARNESS_MESSAGES,
        _HARNESS_MESSAGES,
        tokenizer,
        None,
        {},
        required_prefix_token_ids=previous_call,
    )
    prompt = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=required_prefix_ids,
        template_prefix_token_ids=template_prefix_ids,
        template_token_ids=_build_prompt_token_ids(_HARNESS_MESSAGES, tokenizer),
    )

    assert prompt[: len(previous_call)] == previous_call
    assert prompt == previous_call + [50, 51, 60]


class _MarkerClosedTokenizer:
    """A template with no end-of-turn token of its own, as GLM's.

    1 opens the prompt, 7 opens a user message, 9 a tool block, 6 a system
    message and 8 an assistant turn. 7 and 9 are also what the model emits to
    stop, so they end a turn, and so does 2; 6 and 8 do not. The re-rendered
    first answer is 1001; the model had actually produced 220, 17 and then
    stopped on 9.
    """

    eos_token_id = 2
    eos_token_ids = (2, 7, 9)
    text = {1: "[bos]", 6: "<system>", 7: "<user>", 8: "<assistant>", 9: "<tool>"}

    def __init__(self, drop_reasoning_once_more_follows=False):
        self.drop_reasoning_once_more_follows = drop_reasoning_once_more_follows

    def apply_chat_template(self, messages, **kwargs):
        ids = [1]
        for index, message in enumerate(messages):
            last = index == len(messages) - 1
            if message["role"] == "user":
                ids += [7, 40, 41] if index == 0 else [7, 70]
            elif message["role"] == "system":
                ids += [6, 80]
            elif message["role"] == "tool":
                ids += [9, 50, 51]
            elif self.drop_reasoning_once_more_follows and not last:
                ids += [8]
            else:
                ids += [8, 1001]
        return ids + ([8] if kwargs["add_generation_prompt"] else [])

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.text.get(t, f"w{t}") for t in token_ids)


class _ChatMLTokenizer:
    """A template that closes every message itself: 3 opens one, 2 and a newline end it.

    ``trailing_prompt`` renders the assistant header after the last message
    even without a generation prompt, as some templates do. ``assistant_end``
    ends assistant messages with another token than the other roles, one the
    server does not know as an end-of-turn token.
    """

    eos_token_id = 2
    eos_token_ids = (2,)
    roles = {"system": 30, "user": 31, "assistant": 32, "tool": 33}

    def __init__(self, trailing_prompt=False, assistant_end=2):
        self.trailing_prompt = trailing_prompt
        self.assistant_end = assistant_end

    def apply_chat_template(self, messages, **kwargs):
        ids = []
        for index, message in enumerate(messages):
            answer = message["role"] == "assistant"
            body = [1001] if answer else [100 + index]
            end = self.assistant_end if answer else 2
            ids += [3, self.roles[message["role"]]] + body + [end, 99]
        if kwargs["add_generation_prompt"] or self.trailing_prompt:
            ids += [3, 32]
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join("\n" if t == 99 else f"w{t}" for t in token_ids)


_FIRST_CALL = [1, 7, 40, 41, 8, 220, 17, 9]
_TOOL_SESSION = [
    {"role": "user", "content": "list the files"},
    {"role": "assistant", "content": "calling ls"},
    {"role": "tool", "content": "a.py"},
]


def _prompt(tokenizer, messages, previous_call):
    return _spliced_prompt_ids(
        messages,
        messages,
        tokenizer,
        None,
        {},
        tokenizer.eos_token_ids,
        required_prefix_token_ids=previous_call,
    )


def test_a_turn_closed_by_the_next_role_marker_is_not_replayed():
    # GLM has no end-of-turn token: <|observation|> or <|user|> closes the
    # assistant turn. Counting only the markers before the turn cut the full
    # render one block early and appended the previous exchange a second time:
    # _FIRST_CALL + [40, 41, 8, 1001, 9, 50, 51, 8].
    prompt = _prompt(_MarkerClosedTokenizer(), _TOOL_SESSION, _FIRST_CALL)

    assert prompt == _FIRST_CALL + [50, 51, 8]


def test_a_user_message_closes_the_turn_like_a_tool_block():
    messages = _TOOL_SESSION[:2] + [{"role": "user", "content": "thanks"}]
    stopped_on_user = [1, 7, 40, 41, 8, 220, 17, 7]

    prompt = _prompt(_MarkerClosedTokenizer(), messages, stopped_on_user)

    assert prompt == stopped_on_user + [70, 8]


def test_the_closing_marker_is_found_when_history_renders_differently():
    # GLM can drop the reasoning of earlier turns once more messages follow,
    # so the rendered prefix is not a prefix of the full render.
    tokenizer = _MarkerClosedTokenizer(drop_reasoning_once_more_follows=True)

    prompt = _prompt(tokenizer, _TOOL_SESSION, _FIRST_CALL)

    assert prompt == _FIRST_CALL + [50, 51, 8]


def test_a_generation_cut_off_before_its_stop_takes_the_templates_marker():
    hit_the_token_limit = _FIRST_CALL[:-1]

    prompt = _prompt(_MarkerClosedTokenizer(), _TOOL_SESSION, hit_the_token_limit)

    assert prompt == hit_the_token_limit + [9, 50, 51, 8]


@pytest.mark.parametrize("stop", [7, 2])
def test_a_message_keeps_its_marker_when_the_model_stopped_on_another(stop):
    # The model ended on <|user|> or <|endoftext|> and a tool block follows:
    # its token stays (the prefix is verbatim) and the block keeps its marker.
    stopped_elsewhere = _FIRST_CALL[:-1] + [stop]

    prompt = _prompt(_MarkerClosedTokenizer(), _TOOL_SESSION, stopped_elsewhere)

    assert prompt == stopped_elsewhere + [9, 50, 51, 8]


@pytest.mark.parametrize("first_stop", [9, 7])
def test_several_calls_stay_one_growing_sequence(first_stop):
    # What Gym's prefix check and the trainer's token walk rely on: every
    # prompt is the previous call plus the rendering of the new messages.
    # With a first turn that stopped on <|user|>, its stop token and the tool
    # block's marker both stay in the sequence, and the rule keeps applying.
    tokenizer = _MarkerClosedTokenizer()
    messages = list(_TOOL_SESSION)
    previous_call = _FIRST_CALL[:-1] + [first_stop]
    for call in range(3):
        marker = [9] if call == 0 and first_stop != 9 else []
        prompt = _prompt(tokenizer, messages, previous_call)
        assert prompt == previous_call + marker + [50, 51, 8]
        previous_call = prompt + [221, 9]
        messages += [
            {"role": "assistant", "content": "again"},
            {"role": "tool", "content": "a.py"},
        ]


@pytest.mark.parametrize("trailing_prompt", [False, True])
def test_a_template_that_closes_the_assistant_message_is_left_alone(trailing_prompt):
    # ChatML ends the assistant message with <|im_end|> and a newline. The
    # second variant goes on to render the next header: the rendered prefix
    # then does not end on the end-of-turn token, yet the turn is closed.
    tokenizer = _ChatMLTokenizer(trailing_prompt=trailing_prompt)
    system_and_user = [3, 31, 100, 2, 99, 3, 32]
    previous_call = system_and_user + [220, 17, 2]

    prompt = _prompt(tokenizer, _TOOL_SESSION, previous_call)

    assert prompt == previous_call + [99, 3, 33, 102, 2, 99, 3, 32]


def test_a_marker_that_ends_a_later_message_is_not_counted():
    # The template closes the assistant message with a token the server does
    # not have in its end-of-turn set (Llama 3.1's <|eom_id|> when only
    # <|eot_id|> is known). The next end-of-turn token then ends the tool
    # message; counting it would cut that message out of the prompt.
    tokenizer = _ChatMLTokenizer(assistant_end=5)
    previous_call = [3, 31, 100, 2, 99, 3, 32, 220, 17, 5]
    full = _build_prompt_token_ids(_TOOL_SESSION, tokenizer)
    _, rendered_prefix = _compute_splice_inputs(
        _TOOL_SESSION, _TOOL_SESSION, tokenizer, None, {}
    )

    assert (
        _turn_closing_marker(
            previous_call,
            rendered_prefix,
            full,
            frozenset(tokenizer.eos_token_ids),
            tokenizer,
        )
        is None
    )
    prompt = _prompt(tokenizer, _TOOL_SESSION, previous_call)
    assert prompt[: len(previous_call)] == previous_call
    assert prompt[-7:] == [3, 33, 102, 2, 99, 3, 32]


@pytest.mark.parametrize(
    "messages",
    [
        # Nothing follows the assistant turn.
        _TOOL_SESSION[:2],
        # A system message opens with a marker that does not end a turn.
        _TOOL_SESSION[:2] + [{"role": "system", "content": "note"}] + _TOOL_SESSION[2:],
    ],
)
def test_a_turn_with_no_closing_marker_after_it_keeps_the_old_splice(messages):
    tokenizer = _MarkerClosedTokenizer()
    full = _build_prompt_token_ids(messages, tokenizer)
    _, rendered_prefix = _compute_splice_inputs(messages, messages, tokenizer, None, {})

    assert (
        _turn_closing_marker(
            _FIRST_CALL,
            rendered_prefix,
            full,
            frozenset(tokenizer.eos_token_ids),
            tokenizer,
        )
        is None
    )
    # The previous exchange comes back, as before; no new message is lost.
    assert _prompt(tokenizer, messages, _FIRST_CALL)[: len(_FIRST_CALL)] == _FIRST_CALL
    assert _prompt(tokenizer, messages, _FIRST_CALL)[-len(full) + 2 :] == full[2:]


def test_whatever_marker_follows_the_open_turn_closes_it():
    # Not a matter of roles: here the system marker (6) ends a turn too, and
    # the system message keeps it after the model's own stop token.
    tokenizer = _MarkerClosedTokenizer()
    messages = (
        _TOOL_SESSION[:2] + [{"role": "system", "content": "note"}] + _TOOL_SESSION[2:]
    )

    prompt = _spliced_prompt_ids(
        messages,
        messages,
        tokenizer,
        None,
        {},
        (2, 6, 7, 9),
        required_prefix_token_ids=_FIRST_CALL,
    )

    assert prompt == _FIRST_CALL + [6, 80, 9, 50, 51, 8]


def test_no_further_end_of_turn_token_means_no_marker():
    marker = _turn_closing_marker(
        _FIRST_CALL,
        [1, 7, 40, 41, 8, 1001],
        [1, 7, 40, 41, 8, 1001, 50, 51, 8],
        frozenset((2, 7, 9)),
        _MarkerClosedTokenizer(),
    )

    assert marker is None


def test_the_open_turn_check_reads_the_tokens_in_order():
    # The same tokens in another order are another message, not the open turn.
    marker = _turn_closing_marker(
        _FIRST_CALL,
        [1, 7, 40, 41, 8, 1001],
        [1, 7, 41, 40, 8, 1001, 9, 50, 8],
        frozenset((2, 7, 9)),
        _MarkerClosedTokenizer(),
    )

    assert marker is None


def test_whitespace_after_the_end_of_turn_token_does_not_reopen_the_turn():
    # ChatML with the message opener (3) also in the end-of-turn set, say
    # <|im_start|> listed as a stop token: the newline after <|im_end|> must
    # not make the turn look open, or the next opener would be counted and
    # the newline cut.
    previous_call = [3, 31, 100, 2, 99, 3, 32, 220, 17, 2]

    prompt = _spliced_prompt_ids(
        _TOOL_SESSION,
        _TOOL_SESSION,
        _ChatMLTokenizer(),
        None,
        {},
        (2, 3),
        required_prefix_token_ids=previous_call,
    )

    assert prompt == previous_call + [99, 3, 33, 102, 2, 99, 3, 32]


def test_a_long_open_turn_is_not_decoded():
    # GLM's real case: far more than a few tokens follow the last end-of-turn
    # token, so the turn is open without asking the tokenizer.
    class _LongTurn(_MarkerClosedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            return [
                extra
                for t in ids
                for extra in [t] + (list(range(300, 312)) if t == 1001 else [])
            ]

        def decode(self, token_ids, skip_special_tokens=False):
            raise AssertionError("decode called for a long tail")

    prompt = _prompt(_LongTurn(), _TOOL_SESSION, _FIRST_CALL)

    assert prompt == _FIRST_CALL + [50, 51, 8]


@pytest.mark.parametrize("eos_token_ids", [None, set()])
def test_without_an_end_of_turn_set_the_tokenizers_eos_closes_the_turn(eos_token_ids):
    class _ToolMarkerIsEos(_MarkerClosedTokenizer):
        eos_token_id = 9

    prompt = _spliced_prompt_ids(
        _TOOL_SESSION,
        _TOOL_SESSION,
        _ToolMarkerIsEos(),
        None,
        {},
        eos_token_ids,
        required_prefix_token_ids=_FIRST_CALL,
    )

    assert prompt == _FIRST_CALL + [50, 51, 8]


def _echoed_session():
    """The session as a harness that echoes token ids on its messages sends it."""
    return [
        _TOOL_SESSION[0],
        {
            **_TOOL_SESSION[1],
            "prompt_token_ids": [1, 7, 40, 41, 8],
            "generation_token_ids": [220, 17, 9],
        },
        _TOOL_SESSION[2],
    ]


def test_ids_echoed_on_the_last_assistant_message_close_the_turn():
    # NeMo-Gym's own harnesses echo the ids on the messages. The parser's
    # conversation has them stripped, so the two lists differ.
    tokenizer = _MarkerClosedTokenizer()
    raw_messages = _echoed_session()
    conversation = [
        {k: v for k, v in m.items() if not k.endswith("_token_ids")}
        for m in raw_messages
    ]

    prompt = _spliced_prompt_ids(
        raw_messages, conversation, tokenizer, None, {}, tokenizer.eos_token_ids
    )

    assert prompt == _FIRST_CALL + [50, 51, 8]


def test_messages_before_a_later_assistant_message_are_not_cut_out():
    # The top-level prefix ends at the model's turn, but the harness appended
    # an assistant message of its own after the tool result. The rendered
    # prefix then shows a turn the model's tokens do not have; closing the
    # last turn would drop the tool result and the harness's message.
    tokenizer = _MarkerClosedTokenizer()
    messages = _TOOL_SESSION + [
        {"role": "assistant", "content": "written by the harness"},
        {"role": "tool", "content": "a.py"},
    ]

    prompt = _prompt(tokenizer, messages, _FIRST_CALL)

    assert prompt == _FIRST_CALL + [50, 51, 8, 1001, 9, 50, 51, 8]


@pytest.mark.parametrize(
    "extra_messages, reason",
    [
        # A system message between the open turn and the next marker.
        ([{"role": "system", "content": "note"}, _TOOL_SESSION[2]], "directly follow"),
        # A tool result and an assistant message the harness wrote.
        (
            [
                _TOOL_SESSION[2],
                {"role": "assistant", "content": "written by the harness"},
                _TOOL_SESSION[2],
            ],
            "messages lie between",
        ),
    ],
)
def test_a_marker_that_is_turned_down_is_reported_once(
    monkeypatch, caplog, extra_messages, reason
):
    # The replay that comes back is silent otherwise: the prompt still starts
    # with the previous call.
    from nemo_rl.models.generation.trtllm import trtllm_http_server

    monkeypatch.setattr(trtllm_http_server, "_reported_splice_fallbacks", set())
    messages = _TOOL_SESSION[:2] + extra_messages

    with caplog.at_level("WARNING", logger=trtllm_http_server.__name__):
        for _ in range(2):
            _prompt(_MarkerClosedTokenizer(), messages, _FIRST_CALL)

    reports = [r.getMessage() for r in caplog.records if "Prefix splice" in r.message]
    assert len(reports) == 1
    assert reason in reports[0]


def test_ids_on_an_earlier_assistant_message_keep_the_old_splice():
    # The prefix found on the messages ends at the first answer, the rendered
    # prefix at the second: closing the second turn would drop what lies
    # between them.
    tokenizer = _MarkerClosedTokenizer()
    messages = [
        _TOOL_SESSION[0],
        {
            **_TOOL_SESSION[1],
            "prompt_token_ids": [1, 7, 40, 41, 8],
            "generation_token_ids": [220, 17, 9],
        },
        _TOOL_SESSION[2],
        {"role": "assistant", "content": "no ids on this one"},
        {"role": "tool", "content": "a.py"},
    ]

    prompt = _spliced_prompt_ids(
        messages, messages, tokenizer, None, {}, tokenizer.eos_token_ids
    )

    # The cut is the second marker of the full render, the one before the
    # first tool block, as it was before the turn-closing rule.
    assert prompt == _FIRST_CALL + [50, 51, 8, 1001, 9, 50, 51, 8]


def _build_through_the_parser(tokenizer, messages, **kwargs):
    """build_spliced_prompt_ids, the function both routes call, with the real parser."""
    pytest.importorskip("tensorrt_llm.serve.chat_utils")
    transformers = pytest.importorskip("transformers")
    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        build_spliced_prompt_ids,
    )

    kwargs.setdefault("tools", None)
    kwargs.setdefault("template_kwargs", {})
    return asyncio.run(
        build_spliced_prompt_ids(
            messages,
            kwargs.pop("tools"),
            tokenizer,
            transformers.PretrainedConfig(),
            kwargs.pop("template_kwargs"),
            tokenizer.eos_token_ids,
            **kwargs,
        )
    )


def test_the_spliced_prompt_is_built_through_the_parser():
    prompt = _build_through_the_parser(
        _MarkerClosedTokenizer(), _TOOL_SESSION, required_prefix_token_ids=_FIRST_CALL
    )

    assert prompt == _FIRST_CALL + [50, 51, 8]


def test_the_parser_route_reads_ids_from_the_raw_messages():
    # Raw messages for the echoed ids, the parsed conversation for the render.
    class _Strict(_MarkerClosedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert all("prompt_token_ids" not in m for m in messages)
            return super().apply_chat_template(messages, **kwargs)

    prompt = _build_through_the_parser(_Strict(), _echoed_session())

    assert prompt == _FIRST_CALL + [50, 51, 8]


def test_render_arguments_reach_both_renders():
    seen = []

    class _Recording(_MarkerClosedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs)
            return super().apply_chat_template(messages, **kwargs)

    tools = [{"type": "function", "function": {"name": "ls"}}]

    _build_through_the_parser(
        _Recording(),
        _TOOL_SESSION,
        tools=tools,
        template_kwargs={"clear_thinking": True},
        chat_template="TEMPLATE",
        required_prefix_token_ids=_FIRST_CALL,
    )

    assert sorted(kwargs["add_generation_prompt"] for kwargs in seen) == [False, True]
    for kwargs in seen:
        assert kwargs["tools"] == tools
        assert kwargs["clear_thinking"] is True
        assert kwargs["chat_template"] == "TEMPLATE"


def test_trtllm_parser_rejects_invalid_tool_arguments():
    messages = _tool_call_messages()
    messages[1]["tool_calls"][0]["function"]["arguments"] = "not-json"

    with pytest.raises(ValueError, match="function.arguments"):
        _parse_with_trtllm(messages)


class _FakeRequest:
    """Minimal Request stand-in: headers plus the two body accessors."""

    def __init__(self, headers: dict, raw: bytes):
        self.headers = headers
        self._raw = raw

    async def body(self) -> bytes:
        return self._raw

    async def json(self):
        return json.loads(self._raw)


def test_a_plain_json_body_is_decoded_as_json():
    payload = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    request = _FakeRequest({}, json.dumps(payload).encode())

    assert asyncio.run(_decode_request_body(request)) == payload


def test_a_msgpack_body_is_decoded_when_the_orchestrator_flags_it():
    """TRT-LLM's disagg orchestrator msgpacks the forwarded worker request.

    It leaves Content-Type as application/json so FastAPI still routes the body
    and signals the encoding with X-TRTLLM-Msgpack instead
    (tensorrt_llm/serve/openai_client.py: MSGPACK_HEADERS). Decoding on
    Content-Type alone therefore hands json.loads a msgpack frame, and every
    orchestrator-forwarded request dies on its first byte.
    """
    msgspec = pytest.importorskip("msgspec")
    payload = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    raw = msgspec.msgpack.Encoder().encode(payload)
    # The header, not the content type, is what selects the decoder.
    request = _FakeRequest(
        {"content-type": "application/json", "x-trtllm-msgpack": "1"}, raw
    )

    assert asyncio.run(_decode_request_body(request)) == payload
    with pytest.raises(UnicodeDecodeError):
        json.loads(raw)


def test_no_disaggregated_params_means_an_aggregated_request():
    assert _parse_disaggregated_params(None) == (None, False)


@pytest.mark.parametrize("requested", [True, False])
def test_the_b64_request_survives_the_wire_to_engine_conversion(requested):
    # The orchestrator sets return_prompt_token_ids_b64 on the *wire* params
    # (gen_tokids_ctxbytes). The LLM-level params the engine takes have no
    # such field, so it must be read before converting, not after.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")

    disagg_params, return_b64 = _parse_disaggregated_params(
        {"request_type": "context_only", "return_prompt_token_ids_b64": requested}
    )

    assert disagg_params.request_type == "context_only"
    assert return_b64 is requested


@pytest.fixture
def stub_to_disaggregated_params(monkeypatch):
    protocol = types.ModuleType("tensorrt_llm.serve.openai_protocol")
    protocol.to_disaggregated_params = lambda params: SimpleNamespace(
        model_dump=lambda: {"request_type": "context_only"}
    )
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)


def _context_leg_body(return_prompt_token_ids_b64: bool) -> dict:
    gen = SimpleNamespace(disaggregated_params=object(), finish_reason="length")
    response = _context_leg_response("m", [1, 2, 3], gen, return_prompt_token_ids_b64)
    return json.loads(response.body)


def test_the_context_leg_returns_b64_prompt_ids_when_asked(
    stub_to_disaggregated_params,
):
    body = _context_leg_body(return_prompt_token_ids_b64=True)

    assert "prompt_token_ids" not in body
    decoded = np.frombuffer(base64.b64decode(body["prompt_token_ids_b64"]), np.int32)
    assert decoded.tolist() == [1, 2, 3]


def test_the_context_leg_returns_the_int_list_by_default(
    stub_to_disaggregated_params,
):
    body = _context_leg_body(return_prompt_token_ids_b64=False)

    assert body["prompt_token_ids"] == [1, 2, 3]
    assert "prompt_token_ids_b64" not in body


# What Gym's converter sends: each tool's keys in the OpenAI SDK's TypedDict
# order, "function" before "type".
_GYM_ORDER_TOOL = {
    "function": {
        "name": "execute_bash",
        "description": "Run a command.",
        "parameters": {"type": "object", "properties": {}},
        "strict": True,
    },
    "type": "function",
}


def test_normalized_tools_put_type_first_like_the_disagg_route_and_vllm():
    # Templates serialize each tool verbatim (`tool | tojson`), so key order is
    # part of the prompt.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        normalize_chat_inputs,
    )

    _, [tool] = normalize_chat_inputs(
        [{"role": "user", "content": "hi"}], [_GYM_ORDER_TOOL]
    )

    assert list(tool) == ["type", "function"]
    assert tool["function"] == _GYM_ORDER_TOOL["function"]


def test_normalized_messages_keep_the_ids_the_splice_reads():
    # _compute_splice_inputs takes the on-policy prefix from these; losing them
    # silently re-renders every later turn from the template.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        normalize_chat_inputs,
    )

    assistant = {
        "role": "assistant",
        "content": "done",
        "reasoning": "think",
        "reasoning_content": "think",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3, 4],
    }

    messages, tools = normalize_chat_inputs(
        [{"role": "user", "content": "hi"}, assistant], None
    )

    assert messages[1]["prompt_token_ids"] == [1, 2]
    assert messages[1]["generation_token_ids"] == [3, 4]
    assert tools is None


def test_the_chat_route_renders_gym_tools_in_normalized_order(monkeypatch):
    # The engine answering Gym directly must render what the disagg frontend
    # renders; it used to pass Gym's raw tool dicts straight to the template.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from fastapi.testclient import TestClient

    from nemo_rl.models.generation.trtllm import trtllm_http_server

    rendered = {}

    async def fake_build(
        messages,
        tools,
        tokenizer,
        model_config,
        template_kwargs,
        eos_token_ids,
        chat_template=None,
        required_prefix_token_ids=None,
    ):
        rendered["tools"] = tools
        # Stop before generation; the route answers a ValueError with a 400.
        raise ValueError("rendered")

    monkeypatch.setattr(trtllm_http_server, "build_spliced_prompt_ids", fake_build)
    app = trtllm_http_server.create_app(
        llm=SimpleNamespace(_hf_model_config=object(), _generation_config=None),
        tokenizer=SimpleNamespace(eos_token_id=0),
        model_name="m",
        max_seq_len=16,
        sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        val_sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        tool_parser="qwen3",
    )

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_GYM_ORDER_TOOL],
        },
    )

    assert response.status_code == 400
    assert [list(tool) for tool in rendered["tools"]] == [["type", "function"]]


_TRAIN = {"temperature": 1.0, "top_p": 1.0, "top_k": None}
_VAL = {"temperature": 0.6, "top_p": 0.95, "top_k": None}


@pytest.mark.parametrize(
    ("request_sampling", "val_sampling_config", "served"),
    [
        ({"temperature": 1.0, "top_p": 1.0}, _VAL, _TRAIN),
        ({"temperature": 0.6, "top_p": 0.95}, _VAL, _VAL),
        # Unpinned params take the train profile.
        ({}, _VAL, _TRAIN),
        ({"temperature": 0.6, "top_p": 0.95}, _TRAIN, None),
        ({"temperature": 0.6, "top_p": 1.0}, _VAL, None),
    ],
)
def test_the_chat_route_serves_the_profile_the_request_pins(
    monkeypatch, request_sampling, val_sampling_config, served
):
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from fastapi.testclient import TestClient

    from nemo_rl.models.generation.trtllm import trtllm_http_server

    class _Stop(Exception):
        pass

    served_with = {}

    def fake_build_sampling_params(_cls, *, sampling_config, **_kwargs):
        served_with["config"] = sampling_config
        # Stop before generation.
        raise _Stop

    monkeypatch.setattr(
        trtllm_http_server, "_build_sampling_params", fake_build_sampling_params
    )
    app = trtllm_http_server.create_app(
        llm=SimpleNamespace(_hf_model_config=object(), _generation_config=None),
        tokenizer=SimpleNamespace(eos_token_id=0),
        model_name="m",
        max_seq_len=16,
        sampling_config=_TRAIN,
        val_sampling_config=val_sampling_config,
        tool_parser="qwen3",
    )
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        # Supplied ids skip prompt rendering.
        "prompt_token_ids": [1, 2],
        **request_sampling,
    }

    expected = _Stop if served is not None else AssertionError
    with pytest.raises(expected):
        TestClient(app).post("/v1/chat/completions", json=body)
    assert served_with.get("config") is served


def _splice_recording_app(monkeypatch, seen):
    from nemo_rl.models.generation.trtllm import trtllm_http_server

    async def fake_build(*args, required_prefix_token_ids=None, **kwargs):
        del args, kwargs
        seen["required_prefix_token_ids"] = required_prefix_token_ids
        # Stop before generation; the route answers a ValueError with a 400.
        raise ValueError("rendered")

    monkeypatch.setattr(trtllm_http_server, "build_spliced_prompt_ids", fake_build)
    return trtllm_http_server.create_app(
        llm=SimpleNamespace(_hf_model_config=object(), _generation_config=None),
        tokenizer=SimpleNamespace(eos_token_id=0),
        model_name="m",
        max_seq_len=16,
        sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        val_sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        tool_parser="qwen3",
    )


def test_the_chat_route_hands_the_top_level_required_prefix_to_the_splice(
    monkeypatch,
):
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from fastapi.testclient import TestClient

    seen = {}
    app = _splice_recording_app(monkeypatch, seen)

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "required_prefix_token_ids": [5, 6, 7],
        },
    )

    assert response.status_code == 400
    assert response.json() == {"error": "rendered"}
    assert seen == {"required_prefix_token_ids": [5, 6, 7]}


@pytest.mark.parametrize("bad", ["5,6", [5, "6"], [1.5], [True], {"ids": [5]}])
def test_the_chat_route_rejects_a_malformed_required_prefix(monkeypatch, bad):
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from fastapi.testclient import TestClient

    seen = {}
    app = _splice_recording_app(monkeypatch, seen)

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "required_prefix_token_ids": bad,
        },
    )

    assert response.status_code == 400
    assert "required_prefix_token_ids" in response.json()["error"]
    assert seen == {}
