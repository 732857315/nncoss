# CI/CD 验收

当前仓库包含设计文档和 `nncoss-llm` 早期原型，验收由 [CI 工作流](../.github/workflows/ci.yml)中的 `docs-acceptance` 与 `llm-acceptance` 检查完成。工作流在推送和 PR 更新时运行，不按修改路径跳过检查。

## 文档检查

[Python 校验脚本](../tools/check_docs.py)使用标准库，在 Python 3.9 及以上版本运行。从仓库根目录执行 `python3 tools/check_docs.py` 可完成相同的本地检查：

- 必需文档存在、可按 UTF-8 读取且非空。
- Markdown 围栏代码块闭合，无未解决的合并冲突标记。
- 行内 Markdown 链接所指向的仓库文件或目录存在。
- Markdown 链接中的标题锚点存在，支持中文标题及重复标题的数字后缀。

脚本处理当前文档使用的 `#` 标题、行内链接及围栏代码块；外部链接可用性、引用式链接、HTML 锚点和代码草案的可执行性不在自动检查范围内。新增其他 Markdown 语法时需同步扩展校验。框架定位、术语、示例与 [开发约定](../AGENTS.md)的一致性还需在 PR 中人工或 AI 审阅。

文档验收不代表框架能力已经实现。其他框架模块实现就绪后，需继续增加适用的 Rust 构建、测试、WASM 与原生目标验证、数值对拍及发布或部署检查。

## LLM 原型检查

`llm-acceptance` 执行以下检查：

- `cargo fmt --all --check`。
- `cargo clippy -p nncoss-llm --all-targets -- -D warnings`。
- 在 debug 与 release 下执行 `cargo test -p nncoss-llm`，覆盖模型解析与非法头字段、手算 RoPE 与 GQA 头映射、采样、KV Cache 扩容失败与回退、prefill／decode 失败后的状态保持，以及多层 GQA 模型各特性组合上的增量解码与无缓存参考前向对拍，并验证 release 的原生 ABI panic 映射、分配错误回归和测试模型与已提交夹具逐字节一致。
- 使用 `python3 -m unittest discover -s tools -p "test_*.py"` 验证导出器拒绝不支持的 bias，并检查 QKV bias 与 QK norm 的跨层一致性。
- 分别构建原生与 `wasm32-unknown-unknown` 发布产物。
- 使用 Python `wasmtime==50.0.1` 执行 `tools/check_wasm_llm.py`，读取已提交的 tiny decoder 夹具，比较原生与 WASM 的 prefill、采样、decode 和 reset 行为及 logits，以及各错误路径的状态码与错误消息。
- Rust 集成测试读取 6 组 Hugging Face golden 夹具，验证 Llama/Qwen2/Qwen3 合成模型的逐位置 logits、prefill 切分与贪心生成；CI 另执行 `tools/check_golden_llm.py`，对原生和 WASM 运行同一组独立参考数据。该检查只依赖 `wasmtime`，不安装 PyTorch。
- 使用 `tools/make_golden_llm.py` 重新生成夹具时，需在仓库外环境安装 `tools/golden_requirements.txt` 的固定依赖；生成器会复现 Hugging Face f64 参考并运行反向变异与不支持检查点拒绝测试。

本地执行完整 LLM 验收需先安装 Rust 的 `wasm32-unknown-unknown` 目标，构建两个发布产物，并在 Python 环境安装对应版本的 `wasmtime`。真实模型导出工具另需 `torch` 与 `safetensors`，默认 CI 不下载公开模型；tiny decoder 对拍通过不代表任何具体公开模型已经完成数值或性能验收。`bench` 示例与 `tools/bench_wasm_llm.py` 仅用于本地记录基线，不属于 CI 检查。

## GitHub 合并门禁

工作流文件只负责执行检查，仓库管理员还需在 GitHub 的规则集或分支保护中为 `main` 启用以下条件：

1. 更改必须通过 PR 合并。
2. 合并前必须通过状态检查，将 GitHub Actions 的 `docs-acceptance` 与 `llm-acceptance` 设为必需检查。
3. 要求分支与目标分支保持同步，并将规则应用于管理员，不配置绕过权限。

首次检查运行后，可以在 GitHub 的必需检查列表中分别选择 `docs-acceptance` 与 `llm-acceptance`。配置入口为[仓库规则设置](https://github.com/732857315/nncoss/settings/rules)。配置完成后需确认规则处于启用状态且覆盖 `main`；仅有本文件或绿色检查不能证明门禁已经生效。

PR 最新提交的全部必需检查完成并通过后才能合并。更新 PR 提交后，必须重新确认验收结果；本地检查不能替代 GitHub 验收。
