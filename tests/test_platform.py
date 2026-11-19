import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from raglab import core, models, store
from raglab.main import app


class RetrievalTests(unittest.TestCase):
    def test_chunk_overlap_and_coverage(self):
        text = "abcdefghij" * 50
        pieces = core.chunks(text, 100, 20)
        self.assertEqual(pieces[0][-20:], pieces[1][:20])
        self.assertEqual(pieces[0] + "".join(p[20:] for p in pieces[1:]), text)
        self.assertEqual(core.chunks(""), [])

    def test_chinese_retrieval_and_zero_match(self):
        docs = [{"source": "a", "text": "毕业答辩需提交毕业论文。"}, {"source": "b", "text": "实习每周需提交周报。"}]
        for strategy in ["tfidf", "mmr"]:
            self.assertEqual(core.retrieve("毕业答辩", docs, 1, strategy)[0]["source"], "a")
            self.assertEqual(core.retrieve("zzzz", docs, 3, strategy), [])

    def test_metrics_missing_labels_and_duplicate_sources(self):
        result = core.offline_metrics("毕业论文", "毕业论文", [{"source": "a", "text": "毕业论文"}], ["a", "a", "b"])
        self.assertEqual(result["answer_token_f1"], 1)
        self.assertEqual(result["source_recall"], .5)
        self.assertIsNone(core.offline_metrics("", "答案", [], [])["source_recall"])


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {"RAGLAB_DATA": self.temp.name, "LLM_API_KEY": ""})
        self.environment.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.environment.stop()
        self.temp.cleanup()

    def seed(self):
        self.assertEqual(self.client.post("/api/demo").status_code, 200)

    def run_experiment(self, **extra):
        config = {"name": "baseline", "dataset_id": "demo", **extra}
        response = self.client.post("/api/experiments", json=config)
        self.assertEqual(response.status_code, 202, response.text)
        return self.client.get("/api/experiments/" + response.json()["id"]).json()

    def test_full_demo_chat_experiments_export_and_snapshot(self):
        self.seed()
        self.seed()
        self.assertEqual(len(self.client.get("/api/documents").json()), 3)
        chat = self.client.post("/api/chat", json={"question": "毕业答辩需要哪些材料？"}).json()
        self.assertIn("毕业论文", chat["answer"])
        self.assertEqual(chat["contexts"][0]["source"], "毕业设计指南.md")
        baseline = self.run_experiment()
        mmr = self.run_experiment(strategy="mmr", top_k=2, chunk_size=100)
        self.assertEqual(baseline["status"], "completed")
        self.assertEqual(len(baseline["report"]["cases"]), 8)
        self.assertEqual(mmr["config"]["strategy"], "mmr")
        self.assertNotIn("faithfulness", baseline["report"]["summary"])
        first = self.client.get("/api/documents").json()[0]
        self.client.delete("/api/documents/" + first["id"])
        exported = self.client.get(f"/api/experiments/{baseline['id']}/export")
        self.assertIn("attachment", exported.headers["content-disposition"])
        self.assertEqual(len(exported.json()["config"]["documents_snapshot"]), 3)
        self.assertEqual(exported.json()["config"]["cases_snapshot"][0]["question"], baseline["report"]["cases"][0]["question"])

    def test_validation_and_missing_configuration(self):
        self.assertEqual(self.client.post("/api/chat", json={"question": "问题"}).status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"question": " "}).status_code, 422)
        self.seed()
        self.assertEqual(self.client.post("/api/chat", json={"question": "问题", "mode": "llm"}).status_code, 400)
        self.assertEqual(self.client.post("/api/experiments", json={"name": "test", "dataset_id": "demo", "evaluator": "deepeval"}).status_code, 400)
        self.assertEqual(self.client.post("/api/experiments", json={"name": "test", "dataset_id": "demo", "overlap": 100}).status_code, 422)
        self.assertEqual(self.client.post("/api/datasets", json={"name": "empty", "cases": []}).status_code, 422)
        self.assertEqual(self.client.get("/api/experiments/missing").status_code, 404)

    def test_upload_formats_duplicates_and_safe_filename(self):
        result = self.client.post("/api/documents/upload", files={"file": ("../guide.md", "文档内容".encode(), "text/markdown")})
        self.assertEqual(result.status_code, 201)
        self.assertEqual(result.json()["name"], "guide.md")
        self.assertEqual(self.client.post("/api/documents", json={"name": "guide.md", "text": "重复"}).status_code, 409)
        for name, content in [("x.exe", b"abc"), ("x.txt", b"\xff"), ("x.pdf", b"invalid"), ("x.txt", b"")]:
            self.assertEqual(self.client.post("/api/documents/upload", files={"file": (name, content)}).status_code, 400)
        self.assertEqual(self.client.post("/api/documents/upload", files={"file": ("x.txt", b"x" * (5*1024*1024+1))}).status_code, 413)

    def test_failed_judge_is_not_successful_report(self):
        self.seed()
        with patch("raglab.main.validate_backend"), patch("raglab.models.judge", side_effect=RuntimeError("secret-key-should-not-leak")):
            result = self.run_experiment(evaluator="deepeval")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("secret-key", json.dumps(result))
        self.assertIsNone(result["report"])

    def test_model_backend_contract(self):
        self.seed()
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}), patch("raglab.models.complete", return_value="需要毕业论文。[1]") as complete:
            result = self.client.post("/api/chat", json={"question": "毕业答辩需要什么？", "mode": "llm"}).json()
            self.assertEqual(result["answer"], "需要毕业论文。[1]")
            self.assertIn("毕业设计指南", complete.call_args.args[0][1]["content"])
        with patch.dict(os.environ, {"LLM_API_KEY": "test"}), patch("raglab.models.generate", side_effect=ValueError("secret")):
            response = self.client.post("/api/chat", json={"question": "毕业答辩", "mode": "llm"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("secret", response.text)

    def test_restart_marks_interrupted_job(self):
        with store.connection() as db:
            db.execute("INSERT INTO experiments(id,name,status,config_json) VALUES('interrupted','test','running','{}')")
        store.initialize()
        result = self.client.get("/api/experiments/interrupted").json()
        self.assertEqual(result["status"], "failed")

    def test_api_key_not_in_health(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "a-private-secret"}):
            response = self.client.get("/api/health")
        self.assertNotIn("a-private-secret", response.text)
        self.assertTrue(response.json()["models"]["configured"])


if __name__ == "__main__":
    unittest.main()
