# 模型

模型文件不随仓库分发，只记录来源和生成方法。按下面的步骤可以在本地复现 `model/` 下的全部文件，每步给出了测试机上得到的文件大小和 SHA-256，便于核对。`model/` 目录中除本说明外的内容都不纳入版本库。

测试机为 Intel Core i7-9750H（6 核 12 线程、16 GiB 内存、macOS）。Python 3.11 依赖见 [tools/knowledge_requirements.txt](../tools/knowledge_requirements.txt)。所有命令都从仓库根目录执行。

## 基础模型

| 目录 | 来源 | 许可 |
|---|---|---|
| `Qwen3-0.6B/` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache License 2.0 |
| `Qwen3-1.7B/` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache License 2.0 |

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='model/Qwen3-0.6B')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', local_dir='model/Qwen3-1.7B')"
```

测试机使用的 Qwen3-0.6B 权重 `model.safetensors` 的 SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

## 导出为 nncoss 格式

f32 模型直接复制权重，结果是确定的：

```bash
python3 tools/export_llm.py --model model/Qwen3-0.6B --output model/Qwen3-0.6B/qwen3-0.6b-f32.nncossllm
```

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `Qwen3-0.6B/qwen3-0.6b-f32.nncossllm` | 2,384,199,744 | `1738cc3266f38cd84bee2ba9d9638fa0bca1902bbb46693e0871e8b76c97b632` |

Q4 模型先做 GPTQ 校准量化，再蒸馏行 scale、Norm 与 bias。命令与效果见 [README 的校准量化与蒸馏一节](../README.md)，在测试机上约需 1 小时：

```bash
CAL="tools/calibration/qwen3_chat.txt README.md crates/nncoss-llm/src/session.rs crates/nncoss-llm/src/model.rs tools/export_llm.py crates/nncoss-web/src/lib.rs"
python3 tools/export_llm.py --model model/Qwen3-0.6B --output /tmp/qwen3-0.6b-q4-gptq.nncossllm --outtype q4 --calibration $CAL
python3 tools/distill_q4_llm.py --model model/Qwen3-0.6B --input /tmp/qwen3-0.6b-q4-gptq.nncossllm \
  --output model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --steps 300 --calibration $CAL
```

在 zsh 中 `$CAL` 不会自动按空格拆分，需要直接写出文件列表或改用 bash。

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `Qwen3-0.6B/qwen3-0.6b-q4.nncossllm` | 318,862,912 | `81c66fdfa82eb696172806cb265d5dc98012e435df20cf515f1fd8f9d1bbefba` |

蒸馏使用 PyTorch 在 CPU 上训练，不同机器、线程数或库版本得到的 scale 会有微小差异，因此 Q4 文件不保证逐字节一致；可用 `tools/check_q4_llm.py` 检查质量是否达到 README 给出的水平。

## 自我学习模型（对照组）

`Qwen3-0.6B-school-r1/` 是在 Qwen3-0.6B 上做一轮拒绝采样自我学习得到的模型，继承 Apache License 2.0。实验过程与测评结果见[实验记录](../docs/brain-experiments.md)。

生成它的是 Python 训练流程。按项目约定，功能代码须用 Rust 实现，所以这套流程已经删除，不再提供复现命令。当时的脚本和经验数据可以在提交 `aa625fd` 中查看。测试机上生成的文件如下：

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `Qwen3-0.6B-school-r1/model.safetensors`（f32 合并权重） | 3,006,565,056 | `9a2414a22822a1b830dd3325517906aad404ae1662b5b91c6c5920d5a037dc66` |
| `Qwen3-0.6B-school-r1/qwen3-0.6b-school-r1-q4.nncossllm` | 318,862,912 | `f473d21bab9abb58b0337eef72b8152e4b448eb08c51c6de1308976e64411d85` |
