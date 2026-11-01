import os
import unittest
from unittest.mock import patch
from backend.app.watermark_engine import _config

class WatermarkConfigTests(unittest.TestCase):
    def test_private_key_is_read_from_environment(self):
        with patch.dict(os.environ, {"WATERMARK_RESEARCH_KEY": "1234567"}):
            params, key = _config(0.5, 2.0, 4.0)
        self.assertEqual(params, {"gamma": 0.5, "delta": 2.0, "z_threshold": 4.0})
        self.assertEqual(key, 1234567)

    def test_missing_key_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "WATERMARK_RESEARCH_KEY"):
                _config(0.5, 2.0, 4.0)

    def test_invalid_gamma_is_rejected(self):
        with patch.dict(os.environ, {"WATERMARK_RESEARCH_KEY": "1234567"}):
            with self.assertRaisesRegex(ValueError, "gamma"):
                _config(1.0, 2.0, 4.0)

if __name__ == "__main__":
    unittest.main()
