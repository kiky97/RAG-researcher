import hashlib
import io
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import core, models, store

ROOT = Path(__file__).resolve().parent.parent
JOB_SLOTS = threading.Semaphore(2)


def uid():
    return uuid.uuid4().hex


@asynccontextmanager
async def lifespan(app):
    store.initialize()
    yield


app = FastAPI(title="RAGLab · 知识库问答与质量评估", version="0.1.0", lifespan=lifespan)


class DocumentInput(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=200000)

    @field_validator("name", "text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("内容不能为空")
        return value.strip()


class Settings(BaseModel):
    top_k: int = Field(default=3, ge=1, le=10)
    chunk_size: int = Field(default=360, ge=100, le=2000)
    overlap: int = Field(default=60, ge=0, le=99)
    strategy: Literal["tfidf", "mmr"] = "tfidf"
    mode: Literal["demo", "llm"] = "demo"
    prompt: str = Field(default="回答简洁，保留关键条件。", max_length=2000)


class Question(Settings):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class Case(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    expected_answer: str = Field(min_length=1, max_length=10000)
    relevant_sources: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("question", "expected_answer")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("内容不能为空")
        return value.strip()


class DatasetInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    cases: list[Case] = Field(min_length=1, max_length=100)


class ExperimentInput(Settings):
    name: str = Field(min_length=1, max_length=100)
    dataset_id: str
    evaluator: Literal["offline", "deepeval"] = "offline"
    threshold: float = Field(default=.5, ge=0, le=1)


def passages(documents, config):
    return [
        {"document_id": doc["id"], "source": doc["name"], "chunk": index + 1, "text": text}
        for doc in documents
        for index, text in enumerate(core.chunks(doc["text"], config.chunk_size, config.overlap))
    ]


def answer(question, config, corpus):
    start = time.perf_counter()
    contexts = core.retrieve(question, corpus, config.top_k, config.strategy)
    if config.mode == "llm" and contexts:
        text = models.generate(question, contexts, config.prompt)
    else:
        text = core.extract_answer(question, contexts)
    return {"question": question, "answer": text, "contexts": contexts,
            "mode": config.mode, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


def validate_backend(config):
    if config.mode == "llm" and not models.configuration()["configured"]:
        raise HTTPException(400, "真实模型模式需要设置 LLM_API_KEY。")
    if isinstance(config, ExperimentInput) and config.evaluator == "deepeval":
        if not models.configuration()["configured"]:
            raise HTTPException(400, "DeepEval 评审需要配置模型 API Key。")
        if find_spec("deepeval") is None:
            raise HTTPException(400, "请安装 requirements-eval.txt 后使用 DeepEval 评审。")


@app.get("/api/health")
def health():
    return {"status": "ok", "models": models.configuration(), "deepeval_installed": find_spec("deepeval") is not None}


@app.get("/api/documents")
def get_documents():
    return [{"id": doc["id"], "name": doc["name"], "characters": len(doc["text"]), "created_at": doc["created_at"]} for doc in store.documents()]


def save_document(doc):
    identifier = uid()
    with store.connection() as db:
        if db.execute("SELECT 1 FROM documents WHERE name=?", (doc.name,)).fetchone():
            raise HTTPException(409, "同名文档已存在，请改名或先删除旧文档。")
        db.execute("INSERT INTO documents(id,name,text) VALUES(?,?,?)", (identifier, doc.name, doc.text))
    return {"id": identifier, "name": doc.name, "characters": len(doc.text)}


@app.post("/api/documents", status_code=201)
def add_document(doc: DocumentInput):
    return save_document(doc)


@app.post("/api/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    name = Path((file.filename or "document").replace("\\", "/")).name
    if not 1 <= len(name) <= 160:
        raise HTTPException(400, "文件名长度必须为 1–160 字符。")
    suffix = Path(name).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, "支持 UTF-8 TXT、Markdown 和文字型 PDF。")
    content = await file.read(5 * 1024 * 1024 + 1)
    await file.close()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "单个文件不能超过 5MB。")
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
        else:
            text = content.decode("utf-8-sig")
    except Exception as exc:
        raise HTTPException(400, "文件解析失败，请检查编码或 PDF 是否加密。") from exc
    if not text.strip() or len(text) > 200000:
        raise HTTPException(400, "文档需包含 1–200000 字符；扫描 PDF 请先做 OCR。")
    return save_document(DocumentInput(name=name, text=text))


@app.delete("/api/documents/{identifier}", status_code=204)
def delete_document(identifier: str):
    with store.connection() as db:
        if db.execute("DELETE FROM documents WHERE id=?", (identifier,)).rowcount == 0:
            raise HTTPException(404, "文档不存在。")


@app.post("/api/chat")
def chat(config: Question):
    validate_backend(config)
    corpus = passages(store.documents(), config)
    if not corpus:
        raise HTTPException(400, "请先添加知识库文档。")
    try:
        return answer(config.question, config, corpus)
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(502, "模型调用失败，请检查服务端模型配置和网络。") from exc


@app.get("/api/datasets")
def get_datasets():
    with store.connection() as db:
        return [{"id": row["id"], "name": row["name"], "cases": json.loads(row["cases_json"])} for row in db.execute("SELECT * FROM datasets ORDER BY rowid DESC")]


@app.post("/api/datasets", status_code=201)
def add_dataset(dataset: DatasetInput):
    identifier = uid()
    with store.connection() as db:
        db.execute("INSERT INTO datasets VALUES(?,?,?)", (identifier, dataset.name, json.dumps([c.model_dump() for c in dataset.cases], ensure_ascii=False)))
    return {"id": identifier, "name": dataset.name}


@app.post("/api/demo")
def seed_demo():
    data = json.loads((ROOT / "sample_data" / "demo.json").read_text())
    with store.connection() as db:
        for doc in data["documents"]:
            if not db.execute("SELECT 1 FROM documents WHERE name=?", (doc["name"],)).fetchone():
                db.execute("INSERT INTO documents(id,name,text) VALUES(?,?,?)", (uid(), doc["name"], doc["text"]))
        if not db.execute("SELECT 1 FROM datasets WHERE id='demo'").fetchone():
            db.execute("INSERT INTO datasets VALUES('demo',?,?)", ("校园知识问答 · 演示集", json.dumps(data["cases"], ensure_ascii=False)))
    return {"message": "演示知识库和测试集已准备好。"}


@app.get("/api/experiments")
def get_experiments():
    with store.connection() as db:
        values = [store.experiment(row) for row in db.execute("SELECT * FROM experiments ORDER BY rowid DESC")]
    # Keep list responses small; detailed case reports live on the detail endpoint.
    for value in values:
        if value["report"]:
            value["report"].pop("cases", None)
    return values


@app.get("/api/experiments/{identifier}")
def get_experiment(identifier: str):
    with store.connection() as db:
        row = db.execute("SELECT * FROM experiments WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise HTTPException(404, "实验不存在。")
    return store.experiment(row)


@app.get("/api/experiments/{identifier}/export")
def export_experiment(identifier: str):
    value = get_experiment(identifier)
    return Response(json.dumps(value, ensure_ascii=False, indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="raglab-{identifier}.json"'})


def run_experiment(identifier, config, docs, cases):
    with JOB_SLOTS:
        try:
            with store.connection() as db:
                db.execute("UPDATE experiments SET status='running' WHERE id=?", (identifier,))
            corpus = passages(docs, config)
            rows = []
            for case in cases:
                result = answer(case["question"], config, corpus)
                result["expected_answer"] = case["expected_answer"]
                result["relevant_sources"] = case["relevant_sources"]
                result["offline"] = core.offline_metrics(result["answer"], case["expected_answer"], result["contexts"], case["relevant_sources"])
                if config.evaluator == "deepeval":
                    result["deepeval"] = models.judge(case["question"], result["answer"], case["expected_answer"], result["contexts"], config.threshold)
                    result["passed"] = all(m["passed"] for m in result["deepeval"].values())
                else:
                    result["passed"] = result["offline"]["answer_token_f1"] >= config.threshold
                rows.append(result)
                with store.connection() as db:
                    db.execute("UPDATE experiments SET report_json=? WHERE id=?", (json.dumps({"completed": len(rows), "total": len(cases)}, ensure_ascii=False), identifier))
            summary = {"pass_rate": sum(r["passed"] for r in rows) / len(rows), "avg_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows)}
            for key in rows[0]["offline"]:
                values = [r["offline"][key] for r in rows if r["offline"][key] is not None]
                summary[key] = sum(values) / len(values) if values else None
            if config.evaluator == "deepeval":
                for key in rows[0]["deepeval"]:
                    summary[key] = sum(r["deepeval"][key]["score"] for r in rows) / len(rows)
            report = {"completed": len(rows), "total": len(rows), "summary": summary, "cases": rows,
                      "pass_rule": "all_deepeval_metrics_gte_threshold" if config.evaluator == "deepeval" else "answer_token_f1_gte_threshold",
                      "measurement_note": "本地词元指标是统计代理指标，不能视作语义正确率或 DeepEval 评分。"}
            with store.connection() as db:
                db.execute("UPDATE experiments SET status='completed', report_json=? WHERE id=?", (json.dumps(report, ensure_ascii=False), identifier))
        except Exception as exc:
            # Exception text from third-party clients can contain sensitive request data.
            with store.connection() as db:
                db.execute("UPDATE experiments SET status='failed', error=? WHERE id=?", (f"实验失败（{type(exc).__name__}）。请检查依赖、模型兼容性及网络后重试；未生成有效评分。", identifier))


@app.post("/api/experiments", status_code=202)
def create_experiment(config: ExperimentInput, background: BackgroundTasks):
    validate_backend(config)
    docs = store.documents()
    if not docs:
        raise HTTPException(400, "请先添加知识库文档。")
    with store.connection() as db:
        dataset = db.execute("SELECT * FROM datasets WHERE id=?", (config.dataset_id,)).fetchone()
        if dataset is None:
            raise HTTPException(404, "测试集不存在。")
        if db.execute("SELECT COUNT(*) FROM experiments WHERE status IN ('queued','running')").fetchone()[0] >= 6:
            raise HTTPException(429, "实验队列已满，请等待当前实验完成。")
        cases = json.loads(dataset["cases_json"])
        snapshot = config.model_dump()
        snapshot.update({"model": models.configuration()["model"] if config.mode == "llm" else "extractive-demo",
                         "judge_model": models.configuration()["judge_model"] if config.evaluator == "deepeval" else None,
                         "dataset_name": dataset["name"], "cases_snapshot": cases,
                         "documents_snapshot": [{"name": d["name"], "sha256": hashlib.sha256(d["text"].encode()).hexdigest(), "text": d["text"]} for d in docs]})
        identifier = uid()
        db.execute("INSERT INTO experiments(id,name,status,config_json) VALUES(?,?,'queued',?)", (identifier, config.name, json.dumps(snapshot, ensure_ascii=False)))
    background.add_task(run_experiment, identifier, config, docs, cases)
    return {"id": identifier, "status": "queued"}


app.mount("/", StaticFiles(directory=ROOT / "raglab" / "static", html=True), name="ui")
