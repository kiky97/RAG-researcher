# 开源来源与贡献范围

- 上游： https://github.com/confident-ai/deepeval
- 基线提交：`34db63dd005ebc9a905c8ef4741ae0687c93c14c`
- 上游包版本：4.3.4
- 许可证：Apache License 2.0，完整文本见本目录 `LICENSE.md`（独立打包时）或仓库根目录 `LICENSE.md`。
- 上游作者与版权声明按原许可证保留。
- 本应用新增：`raglab/` 的 API、检索与生成流程、持久化、评审适配器、界面、演示数据、部署和测试。
- DeepEval 提供的语义指标及框架能力不属于本应用原创贡献。
- 示例校园规则全部虚构，仅用于演示；本地统计指标也不是实际模型效果的证明。

本目录为新增应用，不包含对 DeepEval 核心模块的修改。独立压缩包通过可选依赖使用 DeepEval；完整仓库中可使用 editable 安装接入同一基线源码。
