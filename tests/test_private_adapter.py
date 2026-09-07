from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from scripts.private_adapter import PrivateAdapterError, load_private_adapter


class PrivateAdapterTests(unittest.TestCase):
    def test_adapter_is_unavailable_without_a_local_operator_setting(self) -> None:
        with patch.dict(os.environ, {"NEURAL_FORGE_PRIVATE_ADAPTER_MODULE": ""}):
            self.assertIsNone(load_private_adapter())

    def test_adapter_module_name_fails_closed(self) -> None:
        with patch.dict(os.environ, {"NEURAL_FORGE_PRIVATE_ADAPTER_MODULE": "../operator_adapter"}):
            with self.assertRaises(PrivateAdapterError):
                load_private_adapter()

    def test_incomplete_operator_module_is_rejected(self) -> None:
        module = types.ModuleType("operator_adapter")
        with patch.dict(sys.modules, {"operator_adapter": module}), \
                patch.dict(os.environ, {"NEURAL_FORGE_PRIVATE_ADAPTER_MODULE": "operator_adapter"}):
            with self.assertRaises(PrivateAdapterError):
                load_private_adapter()


if __name__ == "__main__":
    unittest.main()
