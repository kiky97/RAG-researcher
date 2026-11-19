# RAGLab · 基于 DeepEval 的 RAG 知识库问答与质量评估平台

面向 AI 应用开发求职和本科毕业设计的可运行项目。将文档管理、检索问答、固定测试集、批量实验、失败案例分析与报告导出串成完整流程。

这是建立在 [DeepEval](https://github.com/confident-ai/deepeval) 之上的应用层扩展；上游框架、许可证和作者信息均保留。本仓库包含 RAGLab 应用代码，通过可选依赖调用 DeepEval，没有改写上游评估核心。

## 快速运行（Python 3.10+，推荐 3.12）

在本目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn raglab.main:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 ，点击「载入演示数据」。上传或使用示例文档，在「问答测试」提问，然后在「评估实验」创建 baseline 与 MMR 两组实验。API 文档：http://127.0.0.1:8000/docs 。

默认无需 API Key；**离线模式是原文抽取，不是大模型生成**。首次依赖安装需要联网；安装后离线模式可本地运行，字体联网失败会自动回退为系统字体。

## 实现范围

- 中文工作台与响应式界面，文档、测试集、实验分区。
- UTF-8 TXT / Markdown / 文字型 PDF 上传，支持粘贴文本、删除文档与解析错误提示。
- 字符 1–3 gram TF-IDF 稀疏向量检索、余弦相似度、MMR 多样性重排。
- 可调分块长度、重叠字符数、Top K；问答展示原始引用片段。
- OpenAI Chat Completions 兼容模型接入，服务端读取密钥。
- SQLite 持久化；批量评估记录配置、模型名、测试集、文档原文及 SHA256 快照。
- 实验记录并排显示主要指标，详情查看失败案例、DeepEval 评审理由，导出 JSON。
- 后台实验运行、进度轮询、失败状态、重启后标记中断任务。

## 接入真实模型与 DeepEval

在虚拟环境中安装可选依赖：

```bash
pip install -r requirements-eval.txt
export LLM_API_KEY='你的密钥'
export LLM_BASE_URL='https://api.openai.com/v1'
export LLM_MODEL='gpt-4o-mini'
export JUDGE_MODEL='gpt-4o-mini'
export DEEPEVAL_TELEMETRY_OPT_OUT=YES
uvicorn raglab.main:app --host 127.0.0.1 --port 8000
```

Windows 请通过 `$env:LLM_API_KEY='...'` 等方式设置。`.env.example` 只提供字段示例，本地 Python 启动不会自动载入 `.env`。Docker Compose 会读取同目录 `.env` 进行变量替换。

在界面选择「真实大模型」启用生成；选择「DeepEval · 模型评审」启用 `AnswerRelevancyMetric`、`FaithfulnessMetric`、`ContextualRecallMetric`。两项可独立选择，例如对离线抽取结果进行 DeepEval 评审。评审模型必须支持 JSON object 输出并遵循给定 JSON schema；接口不兼容或调用失败时实验标记失败，不伪造评分。当前版本采用经典指标评审模式。

每条样本的评审会调用多次模型，费用取决于服务商、文本长度和测试集大小。平均问答耗时只计检索和生成，不包含评审；目前未统计 token 费用，也未实现流式输出。

## 指标如何解释

| 指标 | 计算方法 | 局限 |
|---|---|---|
| 回答词元 F1 | 回答与参考答案词元集合的 precision / recall 调和平均 | 中文二元字符词元、英文单词；集合忽略词频和顺序，不能判断语义 |
| 来源召回率 | 找到的相关来源数 / 标注相关来源数 | 文档名级别，非 DeepEval 上下文召回率；未标注时为空 |
| 上下文词元覆盖率 | 回答词元与上下文交集 / 回答词元数 | 原文抽取天然偏高，不能视作忠实度 |
| DeepEval 答案相关性 / 忠实度 / 上下文召回率 | 真正调用 DeepEval 与评审模型 | 有模型判断偏差和成本，需人工抽查 |

本地通过条件：F1 ≥ 阈值。DeepEval 通过条件：三个评审指标全部 ≥ 阈值。**不同评估模式、数据集、资料快照的通过率不能直接比较。** MMR 不保证提升；保留实验中出现的退化结果。

## Docker

```bash
docker compose up --build
# 可选，安装评审依赖（先在同目录 .env 配置 API Key）：
WITH_DEEPEVAL=true docker compose up --build
```

默认只映射到本机 8000 端口，数据保存在命名卷。容器使用非 root 用户。当前是单用户本地应用，没有账号与访问控制，不应直接对公网开放。模型模式会把检索资料和测试内容发送到配置的服务商。报告包含文档原文，分享前需检查敏感信息。

## 测试

```bash
python -m unittest discover -s tests -v
node --check raglab/static/app.js  # 可选的 JS 语法检查
```

覆盖中文检索、分块重叠、本地指标、演示数据幂等、问答、实验、快照、导出、上传异常、配置缺失、失败评审和中断恢复。模型 API 契约使用测试替身；实际模型和 DeepEval 云端评审需要用户自己的密钥单独验证。

## 工程边界

这是能运行的本地毕业设计原型。检索索引每次请求重建，适合小型知识库；它没有实现 dense embedding、向量数据库、OCR、用户隔离、分布式队列或流式生成。后台任务存于单个进程内，最多同时运行两个实验，队列最多六个；请只启一个 uvicorn worker，避免启动恢复逻辑影响其他进程。实验资料快照和导出用于追溯，不是独立的一键重放工具；模型版本漂移也可能影响重复实验。

后续可按实际岗位增加 embedding + BM25 混合检索、reranker、异步任务队列与权限管理；增加前需先建立独立测试集和基线。

毕业设计、实验设计和简历材料见 [docs/PROJECT.md](docs/PROJECT.md)。开源来源见 [UPSTREAM.md](UPSTREAM.md)。
