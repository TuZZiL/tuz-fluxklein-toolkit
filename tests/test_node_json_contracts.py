import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from node_json_contracts import parse_layer_strengths_json, parse_slot_data_json  # noqa: E402


class NodeJsonContractTests(unittest.TestCase):
    def test_parse_layer_strengths_accepts_valid_payload(self):
        raw = '{"db":{"0":{"img":1.0,"txt":0.8}},"sb":{"0":0.9}}'
        parsed = parse_layer_strengths_json(raw, "Test")
        self.assertIn("db", parsed)
        self.assertIn("sb", parsed)

    def test_parse_layer_strengths_rejects_invalid_json(self):
        parsed = parse_layer_strengths_json("{bad json", "Test")
        self.assertEqual(parsed, {})

    def test_parse_layer_strengths_rejects_wrong_shape(self):
        parsed = parse_layer_strengths_json('["not","object"]', "Test")
        self.assertEqual(parsed, {})

    def test_parse_slot_data_accepts_valid_list(self):
        parsed = parse_slot_data_json('[{"enabled":true,"lora":"x.safetensors"}]', "Test")
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)

    def test_parse_slot_data_keeps_extra_anatomy_fields(self):
        raw = '[{"enabled":true,"lora":"x.safetensors","anatomy_profile":"Undress Safe","anatomy_strength":0.7,"anatomy_strict_zero":true}]'
        parsed = parse_slot_data_json(raw, "Test")
        self.assertEqual(parsed[0]["anatomy_profile"], "Undress Safe")
        self.assertEqual(parsed[0]["anatomy_strength"], 0.7)
        self.assertTrue(parsed[0]["anatomy_strict_zero"])

    def test_parse_slot_data_rejects_invalid_json(self):
        parsed = parse_slot_data_json("{bad json", "Test")
        self.assertIsNone(parsed)

    def test_parse_slot_data_rejects_non_list(self):
        parsed = parse_slot_data_json('{"enabled":true}', "Test")
        self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()
