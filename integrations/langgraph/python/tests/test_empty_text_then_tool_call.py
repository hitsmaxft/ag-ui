"""
Pin the contract for the empty-text-before-tool-call edge case.

The bug
=======

Some providers (Anthropic, certain OpenAI Responses-API stream shapes)
emit a leading empty-text chunk on the same AIMessage *before* the
tool_call chunks arrive. Sequence on the wire:

    chunk_1: content="", id="ai-msg-X", tool_call_chunks=[]
    chunk_2: content="", id="ai-msg-X", tool_call_chunks=[start(tc-A)]
    chunk_3: content="", id="ai-msg-X", tool_call_chunks=[args(tc-A)]
    chunk_4: content="", id="ai-msg-X", tool_call_chunks=[]   # end

Previously, agent.py silently swallowed chunk_1's empty delta
(TextMessageContentEvent requires delta min_length=1) — so NO
TEXT_MESSAGE_START was emitted for "ai-msg-X". Then chunk_2 dispatched
TOOL_CALL_START with ``parent_message_id="ai-msg-X"`` — a dangling
reference. The frontend's default apply (resolveOrCreateAssistantMessage)
synthesised an empty assistant message keyed on that id, which never
renders.

The fix
=======

When an empty-text chunk would otherwise be swallowed and no message is
currently in progress for this run, emit a TEXT_MESSAGE_START +
TEXT_MESSAGE_END pair (no CONTENT — empty deltas are still illegal on
the wire). The pair declares the assistant message id on the wire so a
follow-up TOOL_CALL_START.parent_message_id=<id> resolves to an
existing message on the frontend.

This file pins:

1. A leading empty-text chunk now emits TEXT_MESSAGE_START + END (no
   CONTENT) so the AIMessage id is declared on the wire.
2. A TOOL_CALL_START that arrives after the empty-text chunk cites
   ``parent_message_id=<id>`` (unchanged from before the fix) AND the
   frontend now has a matching message — no dangling reference.
"""

import asyncio
import unittest

from langchain_core.messages import AIMessageChunk

from ag_ui.core import EventType

from tests.test_nested_tool_end_dedup import (
    _ai_chunk,
    _event,
    _run_stream,
    _stream_end,
    _tool_end,
)


def _text_chunk(text, *, chunk_id="ai-msg-1", node="model"):
    """OnChatModelStream event carrying ONLY a text delta (no tool_call_chunks)."""
    chunk = AIMessageChunk(content=text, id=chunk_id)
    chunk.response_metadata = {}
    chunk.tool_call_chunks = []
    return _event("on_chat_model_stream", node=node, data={"chunk": chunk})


class TestEmptyTextSwallowedBeforeToolCall(unittest.TestCase):
    def test_leading_empty_text_chunk_emits_start_end_pair_no_content(self):
        """The empty delta itself still can't ride on TEXT_MESSAGE_CONTENT
        (min_length=1). But we now emit a TEXT_MESSAGE_START + END pair
        so the AIMessage id is declared on the wire."""
        MSG_ID = "ai-msg-X"
        events = [
            _text_chunk("", chunk_id=MSG_ID),
            _stream_end(),
        ]
        dispatched = asyncio.run(_run_stream(events))

        text_starts = [
            e for e in dispatched
            if e.type == EventType.TEXT_MESSAGE_START and getattr(e, "message_id", None) == MSG_ID
        ]
        text_contents = [
            e for e in dispatched
            if e.type == EventType.TEXT_MESSAGE_CONTENT and getattr(e, "message_id", None) == MSG_ID
        ]
        text_ends = [
            e for e in dispatched
            if e.type == EventType.TEXT_MESSAGE_END and getattr(e, "message_id", None) == MSG_ID
        ]

        self.assertEqual(len(text_starts), 1, "expected one TEXT_MESSAGE_START for MSG_ID")
        self.assertEqual(text_contents, [], "no TEXT_MESSAGE_CONTENT for empty delta")
        self.assertEqual(len(text_ends), 1, "expected one TEXT_MESSAGE_END for MSG_ID")

    def test_tool_call_after_empty_text_has_parent_message_id_that_resolves(self):
        """Headline regression: TOOL_CALL_START.parent_message_id still
        cites the AIMessage id (unchanged), AND the frontend has a
        matching message because the empty-text chunk emitted the
        START+END pair."""
        MSG_ID = "ai-msg-X"
        events = [
            _text_chunk("", chunk_id=MSG_ID),
            _event(
                "on_chat_model_stream",
                node="model",
                data={"chunk": _ai_chunk(name="search", args="", tool_call_id="tc-A", chunk_id=MSG_ID)},
            ),
            _event(
                "on_chat_model_stream",
                node="model",
                data={"chunk": _ai_chunk(args='{"q":"x"}', tool_call_id="tc-A", chunk_id=MSG_ID)},
            ),
            _stream_end(),
            _tool_end("search", "tc-A", content="ok", input_args={"q": "x"}),
        ]
        dispatched = asyncio.run(_run_stream(events))

        # The empty-text chunk emitted a TEXT_MESSAGE_START so the id exists.
        text_starts_for_id = [
            e for e in dispatched
            if e.type == EventType.TEXT_MESSAGE_START and getattr(e, "message_id", None) == MSG_ID
        ]
        self.assertGreaterEqual(
            len(text_starts_for_id), 1,
            "expected TEXT_MESSAGE_START for MSG_ID to declare the assistant message id",
        )

        # And TOOL_CALL_START.parent_message_id still cites MSG_ID (no
        # behaviour change in the tool-call branch).
        tool_starts = [e for e in dispatched if e.type == EventType.TOOL_CALL_START]
        self.assertEqual(len(tool_starts), 1)
        parent = getattr(tool_starts[0], "parent_message_id", None)
        self.assertEqual(
            parent, MSG_ID,
            "TOOL_CALL_START.parent_message_id should resolve to the assistant "
            "message id declared by the empty-text START+END pair",
        )

    def test_empty_text_when_message_already_in_progress_emits_nothing(self):
        """If a text stream is already in progress on the same id (its
        START already fired), a subsequent empty-text chunk must NOT
        emit another START+END pair — that would duplicate the
        declaration. Just swallow."""
        MSG_ID = "ai-msg-X"
        events = [
            _text_chunk("hello", chunk_id=MSG_ID),  # opens stream, fires START + CONTENT
            _text_chunk("", chunk_id=MSG_ID),       # should be swallowed silently
            _stream_end(),
        ]
        dispatched = asyncio.run(_run_stream(events))

        text_starts = [
            e for e in dispatched
            if e.type == EventType.TEXT_MESSAGE_START and getattr(e, "message_id", None) == MSG_ID
        ]
        # Exactly one START — the original one from the "hello" chunk.
        # Empty-delta chunk in the middle did NOT emit a duplicate.
        self.assertEqual(
            len(text_starts), 1,
            "empty-delta chunk arriving mid-stream must not emit an extra TEXT_MESSAGE_START",
        )


if __name__ == "__main__":
    unittest.main()
