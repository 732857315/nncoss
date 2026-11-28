# 与其他推理框架的速度对比

在同一台机器上、用同一份权重，比较 nncoss 与 llama.cpp、ncnn_llm、ncnn-MoE-Runtime 的 prefill 与 decode 速度。结果只代表这台机器和这几个模型，不构成性能承诺。方法与历次结果的来龙去脉见[决策记录](decisions.md)，模型来源与校验值见[模型说明](models.md)。每次运行的结果与各框架基准程序的原始输出发布在 Hugging Face 数据集 [w732857315/nncoss](https://huggingface.co/datasets/w732857315/nncoss)。

## 环境

- **CPU**：Intel Core i7-9750H，6 核 12 线程，支持 AVX2、FMA、F16C，不支持 AVX-512 与 VNNI。
- **内存**：16 GB。
- **系统与编译器**：macOS 15（Darwin 24.6），Apple Clang 17。
- **OpenMP**：本地 libomp，供需要它的框架使用。
- **机器负载**：机器同时运行着其他程序，测量期间负载在 1.8–7.0 之间。
- **测量时间**：2026-11-28。

## 方法

- **同一份权重**：每组对比里各框架运行的权重逐位相同。nncoss 用的三个 `.nncossllm` 文件与 Qwen3-0.6B 的 Q8_0 GGUF 已发布到 Hugging Face，下载命令见[模型说明](models.md#已发布模型与数据集)。
  - **Q8_0 组**：llama.cpp 与 ncnn-MoE-Runtime 直接读 GGUF，nncoss 读由 `tools/gguf_to_nncoss.py` 原样转换的文件。
  - **INT8 组**：ncnn_llm 读 ncnn 模型库的 `qwen3_0.6b_int8`，nncoss 读由 `tools/ncnn_int8_to_nncoss.py` 原样转换的文件。
- **各用自带的基准**：
  - **llama.cpp**（8e23278，CPU 构建）：`llama-bench -p 256 -n 64 -d 0,256 -r 3`。prefill 取空上下文的 pp256，decode 取深度 256 的 tg64。
  - **nncoss**（0c03a94）：`bench --tokens <256 个 token> --decode 64 --greedy --iterations 3`。
  - **ncnn_llm**（0538ae8）：自带的 `benchllm 4 <线程数>`（pp256、tg64）。
  - **ncnn-MoE-Runtime**（bc51ff2）：CLI 的 `run --metrics` 报告的 Prompt 与 Generation 速度。它的 CLI 会给提示加上聊天模板，多出十来个 token。
- **输入**：提示都是 256 个 token，由 ncnn_llm 的 benchllm 所用的同一段英文构造，贪心生成 64 个 token。
- **线程与轮次**：6 线程与 1 线程。每个组合跑两轮，第二轮倒序，以抵消机器状态随时间的漂移。表中两个数字分别是两轮的结果，两轮相同时只写一个。
- **原版与修正版**：ncnn_llm 与 ncnn-MoE-Runtime 同时测原版和加上本项目所提 PR 的版本。

  | 框架 | PR | 修正内容 |
  |---|---|---|
  | ncnn-MoE-Runtime | [crafcat7/ncnn-MoE-Runtime#28](https://github.com/crafcat7/ncnn-MoE-Runtime/pull/28) | GGUF 量化内核原先只在 MSVC 下编译 SIMD 版本 |
  | ncnn_llm | [futz12/ncnn_llm#61](https://github.com/futz12/ncnn_llm/pull/61) | LM Head 改读精确的 bf16 副本 |

  原版的 ncnn-MoE-Runtime 在 macOS 上要加上 [#27](https://github.com/crafcat7/ncnn-MoE-Runtime/pull/27) 才能编译。
- **ncnn_llm 的 ncnn 子模块**：子模块固定的 ncnn 在只有 AVX2 的 CPU 上 INT8 结果错误，生成乱码，所以 ncnn_llm 的两个版本都带着 [futz12/ncnn#4](https://github.com/futz12/ncnn/pull/4) 的修正。

## 结果

单位为 tokens/s。

### MNN-Pocket-0.3B Q8_0（官方 GGUF）

| 框架 | prefill 6 线程 | decode 6 线程 | prefill 1 线程 | decode 1 线程 |
|---|---|---|---|---|
| **nncoss** | **465.7 / 444.5** | **78.9 / 80.1** | **156.0 / 157.3** | **57.8 / 58.2** |
| llama.cpp | 290.4 / 309.3 | 72.2 / 68.7 | 122.4 / 117.3 | 29.7 |
| ncnn-MoE-Runtime + #28 | 73.8 / 71.0 | 36.9 / 37.3 | 23.1 / 19.2 | 12.9 / 13.3 |
| ncnn-MoE-Runtime 原版 | 74.0 / 73.8 | 9.6 / 9.2 | 23.1 / 23.2 | 2.4 |

### Qwen3-0.6B Q8_0（由 Hugging Face 权重经 llama.cpp 转换量化）

| 框架 | prefill 6 线程 | decode 6 线程 | prefill 1 线程 | decode 1 线程 |
|---|---|---|---|---|
| **nncoss** | **276.3 / 275.9** | **37.9 / 41.2** | **87.5 / 86.0** | **27.3 / 28.0** |
| llama.cpp | 206.6 / 179.8 | 32.3 / 36.1 | 64.2 / 64.7 | 14.3 / 14.5 |
| ncnn-MoE-Runtime + #28 | 46.1 / 40.3 | 26.8 / 26.1 | 12.7 / 10.2 | 6.4 |
| ncnn-MoE-Runtime 原版 | 48.3 / 45.5 | 5.2 | 12.5 / 12.6 | 1.2 |

### Qwen3-0.6B INT8（ncnn 模型库，64 元素一块）

| 框架 | prefill 6 线程 | decode 6 线程 | prefill 1 线程 | decode 1 线程 |
|---|---|---|---|---|
| nncoss | 306.8 / 306.6 | 27.1 / 27.5 | 92.1 / 92.5 | 15.2 / 15.7 |
| ncnn_llm 原版 | **400.7 / 397.2** | 16.7 / 17.0 | **103.7 / 103.0** | 15.1 / 15.0 |
| ncnn_llm + #61 | 397.1 / 396.3 | **29.9 / 30.1** | 102.5 / 102.7 | **17.6 / 17.5** |

## 结论

- **与 llama.cpp 比（同一份 Q8_0 权重）**：nncoss 的 prefill 在 6 线程下是 llama.cpp 的 1.3–1.6 倍，1 线程约为 1.3 倍；decode 在 6 线程下快 9–17%，1 线程约为 llama.cpp 的 1.9–2.0 倍。
- **与 ncnn_llm 比（同一个 INT8 模型）**：
  - 两边的 LM Head 都读 622 MB 的 f32 时，nncoss 的 decode 在 6 线程下约为 ncnn_llm 的 1.6 倍。
  - ncnn_llm 加上 #61 后 LM Head 改读 bf16，每个 token 少读 311 MB，decode 反超 nncoss 约 10%。nncoss 的 INT8 模型目前 LM Head 只能是 f32。
  - prefill 仍是 ncnn_llm 快，6 线程约为 nncoss 的 1.3 倍，1 线程约为 1.1 倍。
- **ncnn-MoE-Runtime**：在这台机器上明显慢。#28 让 decode 提高到原来的 3.8–5.4 倍，但 prefill 仍把量化权重还原成 f32 再做 f32 GEMM，只有其他框架的 1/4 到 1/6 左右。
- **误差范围**：机器负载变化时，同一格两轮相差可达约 15%，小于这个幅度的差别不应解读。
