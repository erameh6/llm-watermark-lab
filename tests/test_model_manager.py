import os
import unittest
from unittest.mock import patch
from backend.app.model_manager import add_model, inspect_huggingface_model, provider_catalog

class ModelManagerTests(unittest.TestCase):
    def test_huggingface_registration_does_not_download(self):
        result = add_model("huggingface", "distilgpt2")
        self.assertEqual(result["status"], "registered_not_downloaded")
        self.assertTrue(result["download_on_load"])

    def test_local_model_directory_is_detected(self):
        result = inspect_huggingface_model(".")
        self.assertEqual(result["source"], "local_directory")
        self.assertFalse(result["download_on_load"])

    def test_api_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                add_model("openai", "gpt-model")

    def test_api_never_claims_watermark_embedding(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}):
            result = add_model("openai", "gpt-model")
        self.assertEqual(result["watermark_embedding"], "not_exposed_by_standard_api")

    def test_catalog_has_expected_providers(self):
        ids = {item["id"] for item in provider_catalog()}
        self.assertTrue({"huggingface", "ollama", "openai", "anthropic", "deepseek"} <= ids)

if __name__ == "__main__":
    unittest.main()
