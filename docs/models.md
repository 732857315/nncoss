# 模型

模型文件不随 Git 仓库分发。已发布的 nncoss 模型与 school-r1 训练数据可从 Hugging Face 直接下载；其余模型在下文记录来源和生成方法，并给出测试机上得到的文件大小和 SHA-256，便于核对。整个 `model/` 目录都不纳入版本库，中间文件写到同样不入库的 `tmp/`。

测试机为 Intel Core i7-9750H（6 核 12 线程、16 GiB 内存、macOS），Python 3.11。完整开发依赖见 [tools/knowledge_requirements.txt](../tools/knowledge_requirements.txt)，可用 `python3 -m pip install -r tools/knowledge_requirements.txt` 安装；只运行下面的已发布 Q4 时使用更小的依赖集合。所有命令都从仓库根目录执行。

## 已发布模型与数据集

| 仓库 | 内容 |
|---|---|
| [w732857315/nncoss-qwen3-0.6b](https://huggingface.co/w732857315/nncoss-qwen3-0.6b) | Qwen3-0.6B 校准蒸馏 Q4、4 个不推荐的权重改写实验模型，以及 `school-r1/` 下与独立 school 仓库重复的存档副本 |
| [w732857315/nncoss-qwen3-1.7b](https://huggingface.co/w732857315/nncoss-qwen3-1.7b) | Qwen3-1.7B 未校准最小二乘 Q4 |
| [w732857315/nncoss-mnn-pocket-0.3b](https://huggingface.co/w732857315/nncoss-mnn-pocket-0.3b) | MNN-Pocket-0.3B 的 nncoss Q4 与由官方 QAT Q4_1 精确还原的 f32 |
| [w732857315/nncoss-Qwen3-0.6B-school](https://huggingface.co/w732857315/nncoss-Qwen3-0.6B-school) | school-r1 的 nncoss Q4、f32 safetensors、配置、分词器与模型卡 |
| [w732857315/nncoss-qwen3-0.6b-school-r1-data](https://huggingface.co/datasets/w732857315/nncoss-qwen3-0.6b-school-r1-data) | 891 条实际训练样本、575 条原始保留轨迹，以及 1,161／427 条训练／留出题库 |

以下公开文件来自本文记录的本地实验，未纳入 CI 模型验收；文件大小和 SHA-256 用于核对，不代表相应模型已经通过所有平台与任务测试。

安装推理与下载依赖，并构建原生和 WASM 目标。以下以 `python3` 指向 Python 3.11 为例；在启用 PEP 668 或不允许向环境安装包的系统上，先执行 `python3.11 -m venv tmp/infer-env`，并用 `tmp/infer-env/bin/python` 替代后续命令中的 `python3`：

```bash
python3 -m pip install huggingface_hub==0.36.2 tokenizers==0.22.2 wasmtime==50.0.1
cargo build --release
rustup target add wasm32-unknown-unknown
cargo build -p nncoss-llm --release --target wasm32-unknown-unknown
```

下载常用模型。四条命令都使用 `allow_patterns`，避免下载实验改写、f32 school 权重、仓库模型卡等不需要或可能覆盖基座模型同名文件的内容：

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('w732857315/nncoss-qwen3-0.6b', local_dir='model/Qwen3-0.6B', allow_patterns=['LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'qwen3-0.6b-q4.nncossllm'])"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('w732857315/nncoss-Qwen3-0.6B-school', local_dir='model/Qwen3-0.6B-school-r1', allow_patterns=['LICENSE', '*.json', '*q4.nncossllm'])"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('w732857315/nncoss-qwen3-1.7b', local_dir='model/Qwen3-1.7B', allow_patterns=['LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'qwen3-1.7b-q4-ls.nncossllm'])"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('w732857315/nncoss-mnn-pocket-0.3b', local_dir='model/MNN-Pocket-0.3B-qat', allow_patterns=['LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', '*.nncossllm'])"
```

运行 school-r1 Q4；把 `--target native` 改成 `--target wasm` 即使用已构建的 WASM 产物：

```bash
python3 tools/run_qwen3.py --target native \
  --model model/Qwen3-0.6B-school-r1/qwen3-0.6b-school-r1-q4.nncossllm \
  --prompt "用一句话介绍 WebAssembly"
```

读取训练数据集：

```bash
python3 -m pip install datasets==3.2.0 pyarrow==17.0.0
python3 -c "from datasets import load_dataset; data=load_dataset('w732857315/nncoss-qwen3-0.6b-school-r1-data', 'train', split='train'); print(len(data)); print(data[0]['text'])"
```

数据集另有 `raw` 与 `questions` 配置、字段说明、生成和筛选流程，详见数据集卡。下载后可按下文表格用 `shasum -a 256 <文件>` 等工具核对哈希；完整开发、导出与校准依赖仍以 `tools/knowledge_requirements.txt` 为准。

## 基础模型

| 目录 | 来源 | 许可 |
|---|---|---|
| `Qwen3-0.6B/` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache License 2.0 |
| `Qwen3-1.7B/` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache License 2.0 |
| `MNN-Pocket-0.3B/`（配置、分词器与 BF16 权重） | [taobao-mnn/MNN-Pocket-0.3B](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B) | Apache License 2.0 |
| `MNN-Pocket-0.3B-GGUF/`（官方 QAT 后的 Q4_1） | [taobao-mnn/MNN-Pocket-0.3B-GGUF](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF) | Apache License 2.0 |

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

Q4 模型先做 GPTQ 校准量化，再蒸馏行 scale、Norm 与 bias。命令与效果见 [README 的校准量化与蒸馏一节](../README.md#校准量化与蒸馏)，在测试机上约需 1 小时：

```bash
CAL="tools/calibration/qwen3_chat.txt README.md crates/nncoss-llm/src/session.rs crates/nncoss-llm/src/model.rs tools/export_llm.py crates/nncoss-web/src/lib.rs"
python3 tools/export_llm.py --model model/Qwen3-0.6B --output tmp/qwen3-0.6b-q4-gptq.nncossllm --outtype q4 --calibration $CAL
python3 tools/distill_q4_llm.py --model model/Qwen3-0.6B --input tmp/qwen3-0.6b-q4-gptq.nncossllm \
  --output model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --steps 300 --calibration $CAL
```

在 zsh 中 `$CAL` 不会自动按空格拆分，需要直接写出文件列表或改用 bash。

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `Qwen3-0.6B/qwen3-0.6b-q4.nncossllm` | 318,862,912 | `81c66fdfa82eb696172806cb265d5dc98012e435df20cf515f1fd8f9d1bbefba` |

蒸馏使用 PyTorch 在 CPU 上训练，不同机器、线程数或库版本得到的 scale 会有微小差异，因此 Q4 文件不保证逐字节一致；可用 `tools/check_q4_llm.py` 检查质量是否达到 README 给出的水平。

### Qwen3-1.7B

1.7B 只做了不经校准的最小二乘 Q4 导出，用时几分钟，结果是确定的：

```bash
python3 tools/export_llm.py --model model/Qwen3-1.7B --output model/Qwen3-1.7B/qwen3-1.7b-q4-ls.nncossllm --outtype q4
```

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `Qwen3-1.7B/qwen3-1.7b-q4-ls.nncossllm` | 917,386,816 | `7307b7eb14782bd1e422a84cb77ead6dc502063d263e0254ca84bcc92caa0ab8` |

GPTQ 校准导出在 16 GiB 内存的测试机上没有完成：第一次进程无报错退出，第二次运行 3 小时 40 分钟后占用约 3.6 GB 交换空间，已手动停止。因此 1.7B 目前没有校准版。在 0.6B 上，不经校准的 Q4 与 f32 的平均 KL 约为校准加蒸馏版本的两倍（0.215 对 0.111）；1.7B 的对应数值没有测过。

### MNN-Pocket-0.3B（与 ncnn-MoE-Runtime 对比用）

MNN-Pocket-0.3B 是 Qwen3 结构的稠密模型（28 层、hidden 896、14/2 个注意力头、head_dim 64、FFN 2560、词表 65,536、嵌入与输出层共用）。为了与 [ncnn-MoE-Runtime](https://github.com/crafcat7/ncnn-MoE-Runtime) 用同一份权重比较，本项目的模型不从 BF16 权重导出，而是把官方 Q4_1 GGUF 精确还原成 f32（`tools/gguf_to_safetensors.py`，需要 `pip install gguf`），再导出成 f32 与 Q4 两种格式。f32 版与 ncnn-MoE-Runtime 运行 Q4_1 时的权重完全相同；Q4 版在此基础上又按本项目的对称 4-bit 格式量化一次。下载时 Hugging Face 只能经本机代理访问，`huggingface_hub` 的下载停在 0 字节，改用带断点续传的 curl：

```bash
curl -L -C - -o model/MNN-Pocket-0.3B-GGUF/MNN-Pocket-0.3B-Instruct-Q4_1.gguf \
  https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF/resolve/main/MNN-Pocket-0.3B-Instruct-Q4_1.gguf
# config.json、分词器文件与 model.safetensors 同样从 taobao-mnn/MNN-Pocket-0.3B 下载到 model/MNN-Pocket-0.3B/
python3 tools/gguf_to_safetensors.py --gguf model/MNN-Pocket-0.3B-GGUF/MNN-Pocket-0.3B-Instruct-Q4_1.gguf \
  --config model/MNN-Pocket-0.3B --output model/MNN-Pocket-0.3B-qat
python3 tools/export_llm.py --model model/MNN-Pocket-0.3B-qat --output model/MNN-Pocket-0.3B-qat/mnn-pocket-0.3b-qat-q4.nncossllm --outtype q4
python3 tools/export_llm.py --model model/MNN-Pocket-0.3B-qat --output model/MNN-Pocket-0.3B-qat/mnn-pocket-0.3b-qat-f32.nncossllm --outtype f32
```

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `MNN-Pocket-0.3B-GGUF/MNN-Pocket-0.3B-Instruct-Q4_1.gguf` | 191,896,672 | `a281c127e388b16537fd3f5d31ab32d3f87739e6dc514ac1093c80d9eac8d097` |
| `MNN-Pocket-0.3B/model.safetensors`（BF16，未在对比中使用） | 723,137,248 | — |
| `MNN-Pocket-0.3B-qat/mnn-pocket-0.3b-qat-q4.nncossllm` | 162,233,920 | `c10cbd1ad2de3ea5acf7b2730d6fa3f0a9b70f61bf4bf4e3d133a7e7a7661b0e` |
| `MNN-Pocket-0.3B-qat/mnn-pocket-0.3b-qat-f32.nncossllm` | 1,211,323,968 | `55d1c285954adf016c48da6d4baec1daf53af65ce6a841e77eb7e3732190137e` |

对比方法与结果见[决策记录](decisions.md)。

### Qwen3-0.6B INT8（与 ncnn_llm 同一模型对比用）

为了与 [ncnn_llm](https://github.com/futz12/ncnn_llm) 在同一个 INT8 模型上比较，本项目不自己量化，而是把 ncnn 模型库中的 `qwen3_0.6b_int8`（ncnn weight block quant：8 bit、每 64 个元素一块、minmax）原样转换成 INT8 格式（见 README 的 [INT8 分块权重](../README.md#int8-分块权重)）。7 个投影矩阵的 int8 码逐字节保留，ncnn 每块的 scale（还原值 = 码 / scale）换成还原系数 1/scale；RMSNorm 权重原样保留；嵌入在 ncnn 中为 bf16，展开成 f32（无损），与 LM Head 共用。

ncnn_llm 从这个模型的 `model.json` 读取 RoPE theta，值为 100000；Qwen3-0.6B 原版 `config.json` 为 1000000。为了让两边计算同一个函数，转换默认使用 ncnn_llm 实际用的 100000，可用 `--rope-theta` 改回原版的值。

```bash
for f in model.json qwen3_decoder.ncnn.param qwen3_decoder.ncnn.bin qwen3_embed_token.ncnn.bin qwen3_proj_out.ncnn.param; do
  curl -L -C - -o model/qwen3_0.6b_int8/$f https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_0.6b_int8/$f
done
python3 tools/ncnn_int8_to_nncoss.py model/qwen3_0.6b_int8 model/Qwen3-0.6B-ncnn-int8/qwen3-0.6b-ncnn-int8.nncossllm
```

转换需要 numpy。脚本解析 ncnn 的 param，按层读取 bin，检查每个矩阵的形状，并要求 bin 恰好读完。

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `qwen3_0.6b_int8/model.json` | 1,727 | `33e433565ba3fb4982c8bf854e64f62952a6193ce94c26154f82bc1ae1ed6ee5` |
| `qwen3_0.6b_int8/qwen3_decoder.ncnn.param` | 69,902 | `c71fac81c6afc3250cd88611dd2ab6c1aabe60a015a086a634b0f9ff0cd6b22d` |
| `qwen3_0.6b_int8/qwen3_decoder.ncnn.bin` | 468,189,184 | `d625e72d8d4bc2d0557e3f96ccc85720e31b44a3cd78b74d01041e5492fd4c54` |
| `qwen3_0.6b_int8/qwen3_embed_token.ncnn.bin` | 311,164,932 | `297c38d88b9a0196016090f18ac0b84b8975566233bdafc2f610048d8c64cb5b` |
| `qwen3_0.6b_int8/qwen3_proj_out.ncnn.param` | 179 | `f541d86b9c837bc2718edfacca1ae5e21607f8bf5fdf0f20612386654cb86ca8` |
| `Qwen3-0.6B-ncnn-int8/qwen3-0.6b-ncnn-int8.nncossllm` | 1,090,519,104 | `70e18ed6e8f9097d4392144ce06e37edf424bb36d4803b6c83d24cf02c00de31` |

对比方法与结果见[决策记录](decisions.md)。

## 直接改写权重得到的模型

`nncoss-edit` 按改动计划改写 Q4 文件中的行 scale 或删除整层，几秒内完成，结果是确定的。用法见 [README](../README.md#知识库与权重改写实验)。目前保留的改写模型：

```bash
cargo build -p nncoss-edit --release
target/release/nncoss-edit apply --input model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm \
  --output model/edits/qwen3-0.6b-q4-l1n2997-off.nncossllm \
  --edit '{"op": "scale", "target": "mlp", "layer": 1, "index": [2997], "factor": 0.0}'
```

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `edits/qwen3-0.6b-q4-l1n2997-off.nncossllm`（第 1 层 2997 号神经元置零） | 318,862,912 | `51c5a98653a6acb5a5d89aa310aff1d4d7e2fca128b52590021a2ff827f8cd8c` |
| `edits/qwen3-0.6b-q4-retrieval-gate.nncossllm`（检索门改写，16 个 MLP 神经元） | 318,862,912 | `8533724547a891030df0da9a9563b90d82f50603dd002766146293e13588a45e` |
| `edits/qwen3-0.6b-q4-gate-repetition.nncossllm`（检索门 + 重复循环改写，另加 8 个 KV 头） | 318,862,912 | `3c45b262024e7d68e00d7abe4767e3ff591df0cd15144c1aaa4b584f0ab33f2e` |
| `edits/qwen3-0.6b-q4-repetition.nncossllm`（只有重复循环改写的 8 个 KV 头） | 318,862,912 | `677c230fad9ae3eacf9f8e2b99176b99ca35be02636da42add6996ee3586cd1d` |

后三个模型的改动计划由 `tools/retrieval_gate.py` 与 `tools/repetition_edit.py` 定位得到（见[实验记录 5.5](brain-experiments.md#55-定位只依据资料作答的部件)），计划原文如下。把它们分别存为 `tmp/retrieval-gate.json` 与 `tmp/repetition.json` 后执行：

```bash
target/release/nncoss-edit apply --input model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm \
  --output model/edits/qwen3-0.6b-q4-retrieval-gate.nncossllm --plan tmp/retrieval-gate.json
target/release/nncoss-edit apply --input model/edits/qwen3-0.6b-q4-retrieval-gate.nncossllm \
  --output model/edits/qwen3-0.6b-q4-gate-repetition.nncossllm --plan tmp/repetition.json
target/release/nncoss-edit apply --input model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm \
  --output model/edits/qwen3-0.6b-q4-repetition.nncossllm --plan tmp/repetition.json
```

`tmp/retrieval-gate.json`：

```json
[
  {"op": "scale", "target": "mlp", "layer": 0, "index": [2], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 8, "index": [753], "factor": 0.5},
  {"op": "scale", "target": "mlp", "layer": 11, "index": [1372], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 13, "index": [1498], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 14, "index": [3007], "factor": 0.5},
  {"op": "scale", "target": "mlp", "layer": 14, "index": [1460], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 15, "index": [1357], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 17, "index": [3033], "factor": 0.5},
  {"op": "scale", "target": "mlp", "layer": 22, "index": [734], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 23, "index": [1706], "factor": 0.5},
  {"op": "scale", "target": "mlp", "layer": 24, "index": [613], "factor": 1.5},
  {"op": "scale", "target": "mlp", "layer": 27, "index": [47, 84, 1608], "factor": 0.5},
  {"op": "scale", "target": "mlp", "layer": 27, "index": [386, 1274], "factor": 1.5}
]
```

`tmp/repetition.json`：

```json
[
  {"op": "scale", "target": "kv_head", "layer": 6, "index": [5], "factor": 0.5},
  {"op": "scale", "target": "kv_head", "layer": 6, "index": [3], "factor": 1.5},
  {"op": "scale", "target": "kv_head", "layer": 10, "index": [3], "factor": 0.5},
  {"op": "scale", "target": "kv_head", "layer": 13, "index": [2], "factor": 0.5},
  {"op": "scale", "target": "kv_head", "layer": 13, "index": [0], "factor": 1.5},
  {"op": "scale", "target": "kv_head", "layer": 18, "index": [2], "factor": 0.5},
  {"op": "scale", "target": "kv_head", "layer": 18, "index": [1], "factor": 1.5},
  {"op": "scale", "target": "kv_head", "layer": 21, "index": [4], "factor": 0.5}
]
```

这三个改写模型在 108 道高考测验题上不用工具的正确率都低于原模型（35.2%、36.1%、30.6% 对 41.7%），只作为实验记录保留，不推荐使用。

以上哈希以本文给出的 Q4 基础模型为输入；基础模型的 scale 若因重新蒸馏而不同，改写结果也会不同。

## 自我学习模型（对照组）

`Qwen3-0.6B-school-r1/` 是在 Qwen3-0.6B 上做一轮拒绝采样自我学习得到的模型，继承 Apache License 2.0。实验过程与测评结果见[实验记录](brain-experiments.md)。可直接按[已发布模型与数据集](#已发布模型与数据集)下载，不必重新训练或量化。

生成它的是 Python 训练流程。按项目约定，功能代码须用 Rust 实现，所以这套流程已经从当前开发树删除，不再提供一键复现命令；当时的脚本和原始经验数据可以在提交 `aa625fd` 中查看。整理后的[训练数据集](https://huggingface.co/datasets/w732857315/nncoss-qwen3-0.6b-school-r1-data)包含 575 条原始保留轨迹、实际训练的 891 条增强与筛选后样本，以及训练／留出题库；文本、token ID、损失掩码和统计均已公开。

测试机上生成的模型文件如下；仓库还包含模型卡、`LICENSE`、`config.json`、`generation_config.json`、`tokenizer.json` 与 `tokenizer_config.json`：

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `Qwen3-0.6B-school-r1/model.safetensors`（f32 合并权重） | 3,006,565,056 | `9a2414a22822a1b830dd3325517906aad404ae1662b5b91c6c5920d5a037dc66` |
| `Qwen3-0.6B-school-r1/qwen3-0.6b-school-r1-q4.nncossllm` | 318,862,912 | `f473d21bab9abb58b0337eef72b8152e4b448eb08c51c6de1308976e64411d85` |
