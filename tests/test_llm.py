import sys
import types
import unittest
from unittest.mock import patch

from core import llm


class LlmFactoryTests(unittest.TestCase):
    def test_groq_reasoning_model_hides_reasoning_from_tool_output(self) -> None:
        captured = {}
        groq_module = types.ModuleType("langchain_groq")

        def chat_groq(**kwargs):
            captured.update(kwargs)
            return captured

        groq_module.ChatGroq = chat_groq
        with (
            patch.object(llm.config, "LLM_PROVIDER", "groq"),
            patch.object(llm.config, "GROQ_API_KEY", "test-key"),
            patch.object(llm.config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            patch.dict(sys.modules, {"langchain_groq": groq_module}),
        ):
            llm.create_chat_model()

        self.assertEqual(captured["reasoning_format"], "hidden")
        self.assertEqual(captured["temperature"], 0)

    def test_non_reasoning_groq_model_uses_ordinary_chat_settings(self) -> None:
        captured = {}
        groq_module = types.ModuleType("langchain_groq")
        groq_module.ChatGroq = lambda **kwargs: captured.update(kwargs) or captured
        with (
            patch.object(llm.config, "LLM_PROVIDER", "groq"),
            patch.object(llm.config, "GROQ_API_KEY", "test-key"),
            patch.object(llm.config, "GROQ_MODEL", "llama-3.3-70b-versatile"),
            patch.dict(sys.modules, {"langchain_groq": groq_module}),
        ):
            llm.create_chat_model()

        self.assertNotIn("reasoning_format", captured)

    def test_missing_provider_key_fails_clearly(self) -> None:
        with (
            patch.object(llm.config, "LLM_PROVIDER", "anthropic"),
            patch.object(llm.config, "ANTHROPIC_API_KEY", None),
        ):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                llm.create_chat_model()


if __name__ == "__main__":
    unittest.main()
