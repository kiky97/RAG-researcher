"""Exercise real framework metrics using a schema-conforming model fixture.

No network calls and no claims about real judge quality.
"""
import importlib.util
import json
import os
import unittest
from unittest.mock import patch

from raglab import models


@unittest.skipUnless(importlib.util.find_spec("deepeval"), "Optional DeepEval dependency is not installed")
class DeepEvalAdapterTests(unittest.TestCase):
    def test_real_metric_chains_accept_compatible_judge(self):
        def fixture(messages, **kwargs):
            schema = json.loads(messages[0]["content"].split(" Match this JSON schema: ")[1])

            def populate(node):
                if "$ref" in node:
                    return populate(schema["$defs"][node["$ref"].split("/")[-1]])
                if "enum" in node:
                    return "yes" if "yes" in node["enum"] else node["enum"][0]
                if "anyOf" in node:
                    return populate(next(n for n in node["anyOf"] if n.get("type") != "null"))
                if node.get("type") == "object":
                    return {key: populate(value) for key, value in node.get("properties", {}).items()}
                if node.get("type") == "array":
                    return [populate(node["items"])]
                if node.get("type") == "string":
                    return "The thesis is required."
                return 1

            return json.dumps(populate(schema))

        with patch.dict(os.environ, {"DEEPEVAL_TELEMETRY_OPT_OUT": "YES"}), patch("raglab.models.complete", side_effect=fixture) as complete:
            result = models.judge("What is required?", "A thesis is required.", "A thesis is required.", [{"text": "A thesis is required."}], .5)
        self.assertEqual(set(result), {"answer_relevancy", "faithfulness", "contextual_recall"})
        self.assertTrue(all(value["score"] == 1 and value["passed"] for value in result.values()))
        self.assertGreater(complete.call_count, 3)
