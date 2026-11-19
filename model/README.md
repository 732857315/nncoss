# 模型

| 目录 | 来源与许可 | 入库内容 | 获取或复现方式 |
|---|---|---|---|
| `Qwen3-0.6B/` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，Apache License 2.0 | 原始权重、分词器、f32 与 Q4 `.nncossllm` | Q4 导出与校准命令见 [README](../README.md) |
| `Qwen3-0.6B-school-r1/` | 本项目在 Qwen3-0.6B 上做一轮拒绝采样自我学习得到，继承 Apache License 2.0 | f32 合并权重与 Q4 `.nncossllm` | 过程见[实验记录](../docs/brain-experiments.md) |
| `Qwen3-1.7B/` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)，Apache License 2.0 | 只提交本项目导出的 `.nncossllm`；原始权重可公开获取，不入库 | 见下文 |

## Qwen3-1.7B

原始权重从 Hugging Face 下载：

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', local_dir='model/Qwen3-1.7B')"
```

下载后用 `tools/export_llm.py --outtype q4` 导出，校准参数与 Qwen3-0.6B 相同。
