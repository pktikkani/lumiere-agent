"""Patch a Pipecat 1.5.0 bug that crashes on messages without a 'role' key.

WHY THIS EXISTS
---------------
`AnthropicLLMAdapter.get_llm_invocation_params_from_context` merges consecutive
same-role messages with `message["role"]` (anthropic_adapter.py ~line 184). But
`_from_universal_context_message` can return a message dict that has no "role"
key (e.g. certain tool-result / spurious-turn shapes). That raises:

    Unknown error occurred: 'role'

which is non-fatal but pushes an ErrorFrame that corrupts the conversation
context, so subsequent voice commands ("pause", "resume") stop working.

This wraps the adapter method to normalize every message so it always has a
"role" (defaulting to "user"), which is what Anthropic expects for stray inputs.
Import this once, before the pipeline is built.
"""

from loguru import logger

try:
    from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter

    # The crash is at anthropic_adapter.py:176 — `_from_universal_context_messages`
    # builds each message via `_from_universal_context_message` (line 169), then reads
    # `message["role"]` (line 176). Some converted messages have no "role" key, so that
    # read raises KeyError('role'). We wrap the PER-MESSAGE converter (called first) to
    # guarantee every message it returns has a "role", so line 176 never crashes.
    _orig = AnthropicLLMAdapter._from_universal_context_message

    def _patched(self, message):
        result = _orig(self, message)
        # The merge loop (anthropic_adapter.py:184/186/190) assumes every message has
        # both "role" and "content". Some converted messages lack one, raising
        # KeyError('role') or KeyError('content'). Guarantee both so it never crashes.
        if isinstance(result, dict):
            if "role" not in result:
                result["role"] = "user"
            if "content" not in result:
                result["content"] = []
        return result

    if getattr(AnthropicLLMAdapter._from_universal_context_message, "_lumiere_patched", False):
        pass
    else:
        _patched._lumiere_patched = True
        AnthropicLLMAdapter._from_universal_context_message = _patched
        logger.info("Applied anthropic 'role' KeyError patch.")
except Exception as e:  # pragma: no cover
    logger.warning(f"Could not apply anthropic role patch: {e}")
