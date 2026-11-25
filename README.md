<div align="center">

# 吻灵枢 · nncoss

**基于 WebAssembly，面向任意平台，支持任意语言调用与扩展的神经网络框架**

A WebAssembly-based neural network framework for any platform and language, with LLVM-powered native compilation

*吻灵枢，一点通。*

![status](https://img.shields.io/badge/status-pre--alpha-orange)
![compiler](https://img.shields.io/badge/compiler-LLVM-262d3a)
![deployment](https://img.shields.io/badge/deployment-WASM%20%2B%20native-654ff0)
![interface](https://img.shields.io/badge/interface-language%20neutral-dea584)

[项目定位](#项目定位) · [编译流水线](#编译流水线) · [多语言调用](#多语言调用) · [扩展机制](#扩展机制) · [使用流程草案](#使用流程草案) · [引擎](#引擎) · [训练](#训练) · [推理](#推理) · [WASM LLM 原型](#wasm-llm-推理原型) · [知识库与权重改写](#知识库与权重改写实验) · [模型迁移](#模型迁移) · [路线图](#路线图)

</div>

---

吻灵枢（nncoss）是一个**基于 WebAssembly、以 LLVM 为编译基础、面向任意平台的神经网络框架**。它支持 **WASM 与原生二进制**两种部署形式，通过语言无关的接口，让任何语言都能调用框架并扩展其能力，覆盖引擎、训练、推理、示例模型与模型迁移。

WASM 承载可移植的计算与扩展模块；LLVM 负责优化与目标代码生成。同一模块既可以交给兼容的 WASM 运行环境执行，也可以提前编译为目标平台的可执行文件、动态库或静态库。应用与扩展通过统一协议接入，具体平台由运行时和宿主适配层连接。

「枢」是门轴，也是中枢。吻灵枢想做模型的枢纽：外部模型迁移进来，自己的模型训练出来，最终都能部署出去；同时保持克制，只接管你不想写的部分，点到为止。

> **项目状态**：当前仓库包含设计文档、验收工具、`nncoss-llm` 早期原型，以及基于它的本地对话界面 `nncoss-web`、外部知识库 `nncoss-knowledge` 与权重改写工具 `nncoss-edit`。`nncoss-llm` 已能构建并执行 WASM 与原生 decoder-only 推理，但尚无可安装的完整框架软件包、浏览器宿主示例或已完成验收的公开模型。除 [WASM LLM 原型](#wasm-llm-推理原型)与[知识库与权重改写实验](#知识库与权重改写实验)明确列出的能力外，下文 API、平台与命令仍为设计目标或使用草案。
>
> **当前优先级**：框架主线先完成可复现的最小纵向原型，以 f32 张量、标量 CPU 内核和固定结构的 MLP 为边界，验证 WASM 与原生 CPU 的数值一致性，再打通 MNIST 训练闭环。`nncoss-llm` 作为独立实验原型提前验证 LLM 会话、KV Cache 与 WASM ABI，不代表框架 v0.1 已完成；GPU、通用迁移和更多平台仍在核心链路验证后分批推进。

## 项目定位

| 核心目标 | 设计方式 |
|----------|----------|
| 任意平台运行 | 以 WASM 为可移植核心，平台可通过兼容的 WASM 运行环境或 LLVM 原生编译与宿主适配接入 |
| WASM 与原生二进制 | 分发 `.wasm`，或针对目标操作系统与架构产出可执行文件、动态库、静态库 |
| 编译基于 LLVM | 框架编译器负责 WASM 与模型语义的转换，LLVM 负责优化和目标代码生成，LLD 负责链接 |
| 任意语言调用与扩展 | 提供 WASM 接口、原生 C ABI 与宿主桥接，支持外部算子、模型组件及运行时扩展 |

“任意平台”是可移植性的设计目标：每个平台需满足模块声明的 WASM 特性与资源要求，或具备相应的 LLVM 目标、链接环境和平台适配。CPU 路径提供基础计算能力，SIMD、多线程与 GPU 作为可选加速能力。

**AI 使用 Rust 编写框架与功能代码，辅助脚本只使用 Python；人类可以使用 Rust、C/C++ 等任意语言。** 这一开发规则与框架面向所有语言开放的接口共同成立。人类编写的代码可通过 WASM 模块、原生 C ABI 或宿主桥接接入；具体约定见[参与贡献](#参与贡献)与 [AGENTS.md](AGENTS.md)。

## 核心模块

| 模块 | 规划职责 | 规划能力 |
|------|----------|----------|
| [引擎](#引擎) | 一切计算的地基 | 张量、可移植算子内核、反向自动微分、CPU / GPU 调度 |
| [训练](#训练) | 把模型训出来 | 层与损失、优化器与学习率调度、Trainer、回调与断点、浏览器内微调 |
| [推理](#推理) | 把模型跑得又快又省 | 图优化与算子融合、整模型编译为 WASM、INT8 量化、KV Cache、多端接口 |
| [示例模型](#示例模型) | 开箱即用的参考实现 | MLP、LeNet-5、ResNet、Transformer、Tiny GPT、BERT，训练与推理俱全 |
| [模型迁移](#模型迁移) | 把外部模型接进来 | PyTorch / ONNX / Hugging Face 导入，键名映射、布局转换、数值校验 |

五个模块共用基于 LLVM 的[编译流水线](#编译流水线)，并通过[多语言接口](#多语言调用)与[扩展机制](#扩展机制)开放能力。引擎是地基，训练与推理构建其上；模型迁移负责接入外部模型，支持范围内的模型可继续训练或直接推理。

## 编译流水线

WASM 是统一的可移植分发格式；LLVM IR 是编译器内部用于优化与生成目标代码的表示。AI 编写的框架实现与扩展使用 Rust，人类贡献可使用 C/C++ 等语言，符合接口约定的外部 WASM 模块也可接入：

```text
框架源码与内置算子（AI：Rust；人类：Rust / C / C++ 等）
   │  rustc / Clang 等语言前端 + LLVM + LLD
   ▼
WASM 模块 ◀── 符合接口约定的外部 WASM 扩展
   │
   ├──▶ WASM 执行路径：兼容的浏览器或运行时 + 宿主适配
   │
   └──▶ nncoss-compiler：校验与转换 → LLVM IR → 目标代码
           ├── CPU：链接运行支持与宿主适配 → 可执行文件 / 动态库 / 静态库
           └── GPU：符合内核约束的计算部分 → NVPTX / AMDGPU / SPIR-V
```

- **生成 WASM**：Rust 实现通过 rustc 的 LLVM 后端编译，人类编写的 C/C++ 可通过 Clang 等工具链接入。语言前端负责语言语义，LLVM 与 LLD 负责优化、代码生成与链接。
- **WASM 执行**：宿主通过兼容的 WASM 运行环境实例化模块，并提供所需的导入接口。文件、网络、日志和线程等能力由宿主适配层提供；浏览器适配与 WASI 适配分别实现这些约定。Rust 标准库在目标上的可用范围参见 [Rust WASM 目标说明](https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html)。
- **原生二进制**：`nncoss-compiler` 将受支持的 WASM 转换为 LLVM IR，针对目标操作系统、架构和 CPU 特性生成机器码，再链接必要的运行支持。AOT 部署目标是不要求用户另行安装 LLVM 或通用 WASM 虚拟机；按需编译与动态加载未经 AOT 的 WASM 模块则需要额外的编译或执行组件。
- **GPU 加速**：编译器识别满足内核协议的计算部分，再交给 LLVM 的 GPU 后端。协议需定义线程索引、地址空间、共享内存、同步与可用操作；模型控制逻辑和设备调度保留在宿主侧。未满足约束的操作应使用已声明的 CPU 实现或报告不支持。后端约束可参考 [LLVM NVPTX 文档](https://llvm.org/docs/NVPTXUsage.html)。
- **依赖边界**：内置计算路径以 Rust 实现为基础，编译器依赖 LLVM 开发库，GPU 后端依赖对应驱动。人类贡献的其他语言实现需声明自身工具链与运行依赖，并遵守相同接口协议。

下面是 Rust 实现启用 WASM SIMD 的配置草案。发布时需分别提供基础与加速配置，并声明所需 WASM 特性；原生编译器与各宿主适配另行构建：

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

> Apple GPU（Metal）与浏览器 WebGPU 的适配列为探索方向；初期在这些环境使用 CPU 路径。具体平台能力以[运行目标](#运行目标)与验收结果为准。

## 多语言调用

框架的张量计算、训练、推理、模型加载与扩展注册都通过版本化接口开放。Rust API 与其他语言绑定共享同一套底层能力；语言绑定负责参数转换、资源管理和错误映射。

| 接入方式 | 调用方举例 | 部署形式 |
|----------|------------|----------|
| WASM 导入与导出接口 | JavaScript / TypeScript，以及可嵌入 WASM 运行时的语言 | `.wasm` + 宿主适配 |
| 原生 C ABI | C/C++、Rust、Python、Go、Java / Kotlin、C#、Swift 等具备相应 FFI 或桥接的语言 | 动态库或静态库 + 语言绑定 |
| 宿主桥接 | 需要自身解释器、虚拟机或专用调用协议的语言 | 由宿主适配层连接框架与该语言的运行环境 |

统一接口需要明确以下约定：

- **版本与能力**：调用前协商 ABI 版本，查询可用设备、算子、数据类型及可选功能。
- **张量与资源**：用不透明句柄表示会话、张量与模型，定义形状、数据类型、布局、设备、所有权及释放方式；跨设备或跨模块传递时明确是否复制数据。
- **内存边界**：WASM 接口使用模块线性内存中的偏移与长度；原生 C ABI 使用本机指针或句柄。两者共享操作语义，各自遵守对应的内存约定。
- **错误与调用行为**：以状态码和可读取的消息报告错误，并定义同步、异步、线程安全与回调生命周期。Rust 绑定映射为 `Result`，其他绑定按语言习惯处理。

C ABI 入口可以由 Rust 实现。其他语言的 SDK、绑定与示例由人类维护；AI 负责 Rust 功能实现、Python 辅助脚本、接口文档和非程序配置。各语言的现成绑定将分批提供，接入新语言可以围绕统一接口实现。

## 扩展机制

扩展点覆盖自定义算子及梯度、模型组件、损失与优化器、数据源、前后处理和训练回调；设备后端通过运行时扩展接口接入。

| 扩展形式 | 编写与接入方式 | 可移植范围 |
|----------|----------------|------------|
| WASM 扩展 | 将扩展编译为符合 nncoss 协议的 WASM 模块，声明入口与所需宿主能力 | 在满足模块特性与导入要求的宿主运行，也可纳入原生 AOT 编译 |
| 原生扩展 | 通过 C ABI 注册算子、回调或设备后端，可作为原生库链接或加载 | 按操作系统、架构及 ABI 构建对应产物 |
| 宿主语言扩展 | 人类使用 Python、JavaScript 等语言编写回调，由宿主桥接注册 | 在相应语言运行环境中执行；分发时需携带该运行依赖 |

每个扩展需声明名称与版本、接口版本、输入输出约束、支持设备和所需能力。训练用算子需提供反向计算或由可微分的基础算子组合；只有前向实现的算子标记为仅用于推理。注册时校验协议和能力，在执行前校验张量约束。

WASM 扩展的 CPU 可移植性与 GPU 内核兼容性分别声明。GPU 编译需满足[编译流水线](#编译流水线)中的内核约束；依赖宿主解释器或回调的扩展在对应宿主执行。

AI 编写的扩展同样只使用 Rust。人类可以选择 C/C++ 或其他语言，按上述任一路径接入。

## 使用流程草案

> 以下是实现完成后的预期使用流程，用于说明设计目标；当前命令与示例尚不可运行，仓库地址为占位符。

开发 Rust 实现计划使用 Rust 1.85+（edition 2024）；构建 `nncoss-compiler` 另需兼容版本的 LLVM 开发库（初步以 LLVM 20+ 为基础，具体版本随原型验证固定）。使用预编译 WASM 或原生库的应用只需对应运行环境和语言绑定。以下 Rust 与浏览器示例展示预期开发流程，浏览器示例另需 Node.js：

```bash
rustup target add wasm32-unknown-unknown
```

跑一个示例模型：

```bash
git clone <仓库地址> nncoss && cd nncoss

# 训练：MNIST 手写数字识别（内核由 LLVM 从 WASM 编译为本机代码）
cargo run --release --example mnist -- train

# 推理：识别一张图片
cargo run --release --example mnist -- infer digit.png

# 浏览器：同一个模型跑在网页里
cd examples/web-mnist && npm install && npm run dev
```

Rust 依赖配置草案（仓库实现就绪后可使用，当前尚未发布到 crates.io / npm）：

```toml
[dependencies]
nncoss = { git = "<仓库地址>" }
```

## 引擎

引擎规划提供张量计算与自动微分，目标是在 WASM、原生 CPU 与支持的 GPU 后端保持算子语义一致，并按数据类型与运算定义数值误差容限。其余模块及各语言接口都规划构建于此。

- **张量**：f32 / f16 / bf16 / i8 等数据类型；步长视图、广播与切片，转置等视图变换零拷贝。
- **算子内核**：AI 使用 Rust 编写，人类可用其他语言通过扩展协议提供实现。基础路径提供标量实现，WASM SIMD 与原生向量化是可选优化；原生 AOT 根据目标优化，具体向量宽度与性能以生成代码和基准验证。符合 GPU 内核约束的实现可映射为并行线程。各后端均与标量参考实现对拍测试。
- **自动微分**：反向模式、动态建图（define-by-run），内置数值梯度检查。
- **调度**：CPU 上多线程并行（原生线程 / Web Worker），GPU 上按网格与工作组并行；张量内存由引擎统一分配与复用。

内核长这样：

```rust
use nncoss::kernel::prelude::*;

/// 逐元素 ReLU：每个线程处理一个元素
#[kernel]
pub fn relu(x: &[f32], y: &mut [f32]) {
    let i = global_id(0);
    if i < x.len() {
        y[i] = x[i].max(0.0);
    }
}
```

上述 `#[kernel]` 的设计需由启动包装层校验输入输出长度等约束；内核协议负责描述线程索引及设备能力。

内核之上，是张量接口。下面使用 Rust 展示 API，其他语言通过绑定访问相同操作：

```rust
use nncoss::prelude::*;

let device = Device::auto(); // 优先选择已支持的加速设备，否则用 CPU
let x = Tensor::randn([64, 784], &device)?;
let w = Tensor::randn([784, 10], &device)?.requires_grad();

let loss = x.matmul(&w)?.relu()?.mean()?;
let grads = loss.backward()?;
assert_eq!(grads[&w].dims(), [784, 10]);
```

## 训练

训练模块规划包括以下能力：

- **层与模型**：Linear、Conv2d、BatchNorm / LayerNorm、Embedding、多头注意力等；`#[derive(Module)]` 自动收集参数，一行保存与加载。
- **损失与优化器**：交叉熵、MSE 等损失；SGD、Adam、AdamW；Warmup、余弦退火等学习率调度。
- **Trainer**：训练 / 验证循环、指标、早停、断点续训、梯度累积与裁剪，进度与日志开箱即用。
- **端侧训练**：同一套训练代码可以编译到 WASM，直接在浏览器或设备上微调，数据不出端。

定义模型：

```rust
use nncoss::prelude::*;

#[derive(Module)]
pub struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Mlp {
    pub fn new(device: &Device) -> nncoss::Result<Self> {
        let fc1 = nn::Linear::new(28 * 28, 256, device)?;
        let fc2 = nn::Linear::new(256, 10, device)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Forward for Mlp {
    fn forward(&self, x: &Tensor) -> nncoss::Result<Tensor> {
        let h = self.fc1.forward(&x.flatten_from(1)?)?.relu()?;
        self.fc2.forward(&h)
    }
}
```

训练它：

```rust
fn main() -> nncoss::Result<()> {
    let device = Device::auto();
    let (train_set, val_set) = data::mnist("data")?.shuffled(42).split_at(55_000);

    let mut trainer = Trainer::new(Mlp::new(&device)?)
        .loss(loss::cross_entropy)
        .optimizer(optim::AdamW::new(1e-3))
        .metric(metrics::Accuracy)
        .callback(callbacks::EarlyStopping::new("val/loss").patience(3))
        .batch_size(128)
        .epochs(10);

    trainer.fit(&train_set, &val_set)?;  // 进度条、断点、日志全自动
    trainer.save("mnist.safetensors")?;  // 权重：可继续训练，或由 Rust / WASM 加载
    trainer.export_onnx("mnist.onnx")?;  // 计算图 + 权重：交给推理与整模型编译
    Ok(())
}
```

输出示意：

```text
nncoss 0.1.0 · device=cuda:0 (sm_89) · params=203.5K
内核已由 LLVM 从 WASM 编译为 PTX（sm_89，命中缓存）
epoch 1/10 ━━━━━━━━━━━━━━━━━━━━ 430/430 · loss 0.3121 · val/loss 0.1547 · val/accuracy 0.9541
epoch 2/10 ━━━━━━━━━━━━━━━━━━━━ 430/430 · loss 0.1283 · val/loss 0.1036 · val/accuracy 0.9689
...
epoch 9/10 ━━━━━━━━━━━━━━━━━━━━ 430/430 · loss 0.0214 · val/loss 0.0763 · val/accuracy 0.9784
✓ 已收敛：早停于 epoch 9，最佳 val/loss 0.0698（epoch 6）
  梯度下降千万次，终于收敛于你。
```

多任务、GAN 等非标准流程，覆写 `Trainer` 的 `training_step` 即可；也可以完全手写训练循环，只复用优化器与自动微分。

## 推理

推理模块规划包括以下能力：

- **图捕获与优化**：把模型前向捕获为静态计算图，做常量折叠、算子融合（如 Conv + BN + ReLU）与死代码消除。
- **整模型编译**：将优化后的计算图、权重与所需内核打包为 WASM；再按部署需要通过 LLVM 编译为目标平台的原生二进制。GPU 路径编译符合约束的内核，并保留 CPU 侧的模型控制与设备调度。
- **内存规划**：按张量生命周期复用缓冲区，压低峰值内存——这对内存受限的 WASM 尤其重要。
- **量化与半精度**：INT8 权重量化；GPU 上可用 f16。
- **生成式推理**：KV Cache 与贪心、Top-k、Top-p 等采样策略，服务 GPT 类模型。
- **多端接口**：Rust 推理会话、WASM 接口与原生 C ABI 共享模型和张量约定。JavaScript / TypeScript、Python、C/C++ 及其他语言通过对应绑定或宿主桥接使用，详见[多语言调用](#多语言调用)。

部署产物按用途选择：

| 产物 | 使用方式 |
|------|----------|
| `.wasm` 模块 | 由兼容运行环境加载，宿主提供模块声明的导入接口 |
| 原生可执行文件 | 将模型与运行支持编译、链接为目标平台程序，按约定的输入输出方式运行 |
| 动态库（`.so` / `.dylib` / `.dll`） | 由各语言通过原生 C ABI 加载与调用 |
| 静态库（`.a` / `.lib`） | 链接进宿主应用，通过原生 C ABI 调用 |

原生产物分别针对操作系统、架构和 ABI 构建。模型包需记录接口版本、输入输出描述、权重及扩展依赖，供 WASM 与原生路径共同校验。

整模型编译命令草案（CLI 尚未发布；原生产物类型的选择参数待实现时固定，GPU 目标表示设备内核的编译）：

```bash
cargo install nncoss-cli

# 1. 计算图与用到的内核编译为一个 WASM（LLVM 生成并优化）
nncoss build mnist.onnx -o mnist.wasm

# 2. 同一个 WASM，按目标硬件再由 LLVM 编译
nncoss compile mnist.wasm --target x86_64-v3    # AVX2 CPU
nncoss compile mnist.wasm --target cuda:sm_89   # NVIDIA GPU
```

Rust：

```rust
use nncoss::prelude::*;
use nncoss::infer::{Optimize, Quantize, Session};

let session = Session::builder()
    .device(Device::auto())
    .optimize(Optimize::Full)  // 常量折叠、算子融合、内存复用
    .quantize(Quantize::Int8)  // 可选
    .load("mnist.onnx")?;      // 也可加载 nncoss build 产出的 .wasm，或用 Session::from_module 捕获 Rust 模型

let x = Tensor::from_slice(&pixels, [1, 1, 28, 28], session.device())?;
let digit = session.run(&x)?.argmax()?;
```

浏览器（现有 TypeScript 接口示意，相关代码由人类维护）：

```ts
import { init, Model, Tensor } from "nncoss";

await init(); // 加载引擎 WASM（C ABI + 手写 TypeScript 封装）

const model = await Model.load("/models/mnist.wasm"); // nncoss build 的产物，也可直接加载 .onnx
const pixels = new Float32Array(28 * 28); // 例如从 <canvas> 读取并归一化到 [0, 1]
const logits = await model.run(Tensor.from(pixels, [1, 1, 28, 28]));

console.log("识别结果：", logits.argmax());
```

浏览器首个验收路径使用预编译的 `.wasm`；示例中直接加载 `.onnx` 的能力需另行实现图解析与执行，再纳入支持清单。

需要自定义前后处理（分词、采样等）时，可将 Rust 或人类编写的其他语言实现接入为 WASM 扩展、原生扩展或宿主回调。规划中的 `examples/web-poetry` 将展示模型与前后处理的组合部署。

## WASM LLM 推理原型

`crates/nncoss-llm` 参考 [ncnn 的 KV Cache 设计](https://github.com/Tencent/ncnn/wiki/kvcache)与 [ncnn_llm](https://github.com/futz12/ncnn_llm) 的嵌入／decoder／LM Head 拆分方式。当前可运行代码是 Rust 实现：标量路径作为参考与 WASM 后端，x86_64 原生目标在检测到 AVX2 时启用向量化内核与线程池；采用会话持有的 KV Cache、prefill 与单 token decode，并复用绑定嵌入权重作为 LM Head；目标部署架构默认复用 ncnn 的 WASM 图执行与 KV Cache，但 ncnn 后端尚未接入，现有 `.wasm` 产物不包含 ncnn。

### 已实现范围

| 能力 | 当前实现 |
|------|----------|
| 模型结构 | 单 batch decoder-only；RMSNorm、RoPE、GQA、SwiGLU；可选 QKV bias、QK norm；绑定或独立 LM Head |
| 数值与算子 | f32 行主序矩阵乘按 16 通道分组求和，x86_64 AVX2 与标量路径逐位一致，同一行权重对批内多个 token 复用并预取下一行；稳定 softmax 与 SiLU；可选 Q4 权重与按组 int8 激活的整数点积；x86_64 AVX2 内核与原生线程池，与标量路径逐位一致；RoPE 与指数函数使用不依赖平台 libm 的实现 |
| 会话 | 多 token prefill、单 token decode、KV Cache 几何增长与容量复用、reset、EOS 停止 |
| 采样 | 贪心、temperature、top-k、top-p；可复现的 SplitMix64 随机序列 |
| 调用接口 | Rust API 与版本为 1 的 WASM／原生 C ABI；模型可整块从内存加载，或按声明的总长度分块流式加载；会话独立持有权重生命周期 |
| 验证 | 原生单元测试；手算 RoPE 旋转（两种布局、多频率）与 GQA 头映射；增量解码与无缓存参考前向在多层 GQA 模型的绑定／独立 LM Head、QKV bias、QK norm 与 RoPE 布局全部组合上对拍；分块加载在全部特性组合、每个二分切点与每个截断长度上与整块加载对拍；同一 tiny decoder 夹具的原生／WASM 数值与错误路径对拍 |

CI 验收覆盖仓库生成的 tiny decoder 和 6 个 Hugging Face 合成小模型。Qwen3-0.6B 的 f32 导出另在本地完成了短序列的原生／WASM／Hugging Face 对拍，见[分块加载与真实模型本地对拍](#分块加载与真实模型本地对拍)；该结果不属于 CI 门禁，也不代表该模型已通过长上下文、Tokenizer 或性能验收，其他公开尺寸的 Llama 或 Qwen 模型尚未验证。架构兼容与模型支持必须分别验证。

### ncnn 后端与优化边界

- **默认复用通用能力**：嵌入、矩阵乘、归一化、注意力和 KV Cache 优先使用 ncnn 已支持的 WASM 实现，不再为部署后端重复编写通用算子。Rust 侧负责版本化 ABI、模型与会话管理；如需额外 C/C++ 桥接，由人类按仓库约定维护。
- **保留参考后端**：当前 Rust 标量路径作为数值参考和最小验收工具，不作为高性能部署承诺。ncnn 后端接入后，需在同一模型与输入上与参考后端逐层或分阶段对拍。
- **分离构建能力**：ncnn 的基础、WASM SIMD 与多线程构建分别声明和验收，不能把某一构建的结果外推到全部浏览器。
- **按热点优化**：先分别测量 prefill 延迟、decode ms/token、tokens/s、峰值内存和模块体积，再针对表现不达标的 GEMM、注意力、KV Cache 或采样路径增加自定义 WASM kernel；通用路径不因预设判断而重写。
- **优化必须可验收**：自定义 kernel 需保留回退路径，并通过标量参考对拍和同输入基准；至少报告相对 ncnn 基线的延迟、吞吐与内存变化。

当前尚未完成 ncnn 后端接入和性能基准，因此本节为实施边界，不是已有性能结果。

### 构建与验收

```bash
cargo test -p nncoss-llm
cargo build -p nncoss-llm --release
rustup target add wasm32-unknown-unknown
cargo build -p nncoss-llm --release --target wasm32-unknown-unknown
```

WASM 产物位于 `target/wasm32-unknown-unknown/release/nncoss_llm.wasm`。安装 Python `wasmtime==50.0.1` 后，可以实际执行原生与 WASM 跨目标验收：

```bash
python3 tools/check_wasm_llm.py
```

该工具是验收脚本而非 Python 绑定，依次覆盖模型加载、prefill、采样、logits 复制、decode、reset，以及模型句柄释放后的会话生命周期；并在两个目标上比较截断模型、越界 token、未 prefill 读取 logits、输出缓冲区过小和非法采样参数的错误码与错误消息。tiny decoder 来自已提交的夹具 `crates/nncoss-llm/tests/fixtures/tiny_v1.nncossllm`，Rust 单元测试会确认测试模型构造器与该夹具逐字节一致。

### 独立 PyTorch 黄金对拍

`crates/nncoss-llm/tests/golden.rs` 使用已提交的 Hugging Face 黄金数据，覆盖 Llama、Qwen2、Qwen3 的 6 个合成小模型，包括 MHA、GQA、MQA、绑定／独立 LM Head、QKV bias、QK norm 和 `head_dim != hidden_size / head_count`。参考值来自 float64 无缓存完整前向并转换为 f32；测试比较每个 token 位置的 logits、所有 prefill 切分点和贪心生成，容差为 `1e-4 + 1e-4 × |参考值|`。

原生与 WASM 使用同一组夹具验收，无需安装 PyTorch：

```bash
python3 tools/check_golden_llm.py
```

重新生成夹具需要仓库外的隔离 Python 环境和 `tools/golden_requirements.txt` 中的固定测试依赖。`make_golden_llm.py` 还会检查 HF 的 f32/f64 与 KV Cache 一致性，并对 RoPE 和每个导出张量执行反向变异对照：

```bash
python3 -m pip install -r tools/golden_requirements.txt
python3 tools/make_golden_llm.py
```

当前生成器使用 Transformers 4.57.6 的顶层 `rope_theta` 配置；Transformers 5 的 `rope_parameters` 格式暂不支持。黄金对拍验证的是所列合成配置，不替代公开模型的真实权重、Tokenizer 与长上下文验收。

### 分块加载与真实模型本地对拍

`ModelLoader` 与 `nncoss_llm_model_loader_*` 接口按声明的总长度接收任意大小的连续分块，逐个张量解析 `.nncossllm`，模型格式保持不变。加载器只保留 64 字节的头部与跨分块的不完整数值，每个张量在进入时按文件头给出的尺寸一次性可失败地分配；声明长度不足以容纳下一个张量时，在分配前即报告截断。整块加载 `Model::from_bytes` 与 `nncoss_llm_model_load` 复用同一解析器，已有错误码与错误消息不变。

整块加载要求输入缓冲区与解析后的权重同时驻留内存，且 wasm32 的单次分配不能超过 `isize::MAX`（2 GiB − 1 字节）。分块加载不需要整文件输入缓冲区，最大的单次分配是最大的单个权重张量。

模型文件不随仓库分发，按[模型说明](docs/models.md)下载 Qwen3-0.6B 并导出 `model/Qwen3-0.6B/qwen3-0.6b-f32.nncossllm`（2,384,199,744 字节）。下列命令以 1 MiB 分块把它分别加载到原生动态库与 `wasm32-unknown-unknown` 产物，执行 5 个 token 的 prefill 与 4 步贪心 decode 并比较 logits；`--hf-model` 为可选项，需要 `tools/golden_requirements.txt` 的依赖，用 Hugging Face f32 eager 前向作为参考：

```bash
python3 tools/check_real_llm.py
python3 tools/check_real_llm.py --hf-model model/Qwen3-0.6B
```

2026-11-16 在 x86_64 macOS、Python `wasmtime==50.0.1` 上的本地结果：

| 项目 | 结果 |
|------|------|
| 整块路径（wasm32） | `nncoss_llm_alloc(2384199744)` 返回空指针，无法加载 |
| 分块路径最大单次分配 | 622,329,856 字节（词嵌入张量），分块缓冲区 1,048,576 字节 |
| WASM 线性内存 | 初始 1,114,112 字节；加载后 2,390,622,208 字节；上下文容量 11、推进 9 个 token 后仍为 2,390,622,208 字节（wasm32 上限 4,294,967,296 字节） |
| 原生／WASM | 4 个贪心 token 一致，5 组 logits 最大误差 5.722e-05 |
| 原生／Hugging Face | 4 个贪心 token 一致，5 组 logits 最大误差 8.965e-05 |
| WASM／Hugging Face | 4 个贪心 token 一致，5 组 logits 最大误差 1.001e-04 |

该对拍使用固定 token id，不经过 Tokenizer；只覆盖 9 个位置的短序列，不包含长上下文、采样分布或性能验收。KV Cache 随上下文增长，该模型每个 token 占 229,376 字节，更长上下文在 wasm32 的可用余量尚未实测。该脚本依赖本地导出的模型，不纳入 CI。

### Q4 量化与原生加速

`tools/export_llm.py --outtype q4` 提供项目内的 Q4 权重编码，不改变模型 ABI 或分块加载协议。矩阵按 32 列分组，每个矩阵依次保存三段数据：每行一个 f32 行 scale、每组一个 i8 组倍率（可为负）、每组 16 字节的 4-bit 打包值，末尾补零到 4 字节对齐；权重还原值为 `行 scale × 组倍率 × q`，`q ∈ [-8, 7]`，平均约 4.25 bit／权重。导出器对每组尝试 17 个候选步长与两种符号，用最小二乘重新拟合步长并取还原误差最小者，在 Qwen3-0.6B 上比按 absmax/7 取整的权重相对均方误差低约 32%。Norm 与 bias 保持 f32，嵌入查表和 LM Head 共用同一编码。

推理时线性层的输入按同样的 32 列分组量化为 int8（每组一个 f32 scale），组内做整数点积后乘以 `组倍率 × 输入 scale` 累加。x86_64 在运行时检测到 AVX2 时使用向量化内核和常驻线程池（默认取物理核数，最多 8 个，可用环境变量 `NNCOSS_THREADS` 覆盖），每个线程顺序读取一段连续的权重行；prefill 每批 32 个 token，同一行权重对多个 token 复用，注意力把每个 KV 头的历史位置切成固定的 256 个位置一块，按块分别计算局部最大值、指数和与未归一化输出，再按块的顺序合并；单 token decode 时（KV 头 × 块）并行，长上下文下 6 个线程负载均衡，prefill 每块的 K/V 对整批 token 复用。指数小于 e^-41 的注意力权重视为 0，以免产生非规格化浮点数拖慢计算，对结果的影响低于 f32 精度。分块与合并顺序与线程数无关，标量与 AVX2 路径逐位一致。KV Cache 按 `[KV 头][位置][head_dim]` 连续存放；Q4 模型的 KV Cache 以 f16 存储（Qwen3-0.6B 每个位置 114,688 字节），f32 模型保持 f32（每个位置 229,376 字节），f16 与 f32 之间的转换使用软件实现的就近舍入，原生目标在支持 F16C 时批量转换，两者逐位一致。其他目标（包括 WASM）使用标量实现。另有可选的 int8 KV Cache，见下文。向量化与标量路径按相同的通道划分和求和顺序计算，RoPE 的幂与正余弦、softmax 与 SiLU 的指数使用只依赖 IEEE 754 基本运算的实现，因此原生与 WASM 的 logits 逐位一致；批量 prefill 与逐 token decode 的结果也逐位一致。

```bash
python3 tools/export_llm.py --model model/Qwen3-0.6B --output model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --outtype q4
python3 tools/check_real_llm.py --model model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --targets native,wasm
python3 tools/check_q4_llm.py
cargo run -p nncoss-llm --release --example bench -- --model model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --prompt 24 --decode 64
```

本地 Qwen3-0.6B Q4 文件为 318,862,912 字节，约 304.1 MiB；不经校准导出约需 6 分钟，校准与蒸馏见下文。以下数据在一台 Intel Core i7-9750H（6 核 12 线程、16 GiB、macOS、接通电源）上测得，均为本地结果，不纳入 CI，也不构成性能承诺：

| 场景 | 优化前 | 当前 |
|------|--------|------|
| 原生 decode（prompt 24、生成 64、temperature 0.6／top-k 20／top-p 0.95） | 25.1 tokens/s | 78–81 tokens/s |
| 原生 decode（prompt 256、生成 64） | — | 68–69 tokens/s |
| 原生 decode（prompt 1024、生成 64） | — | 50–54 tokens/s |
| 原生 decode（prompt 2048、生成 64） | — | 约 36 tokens/s |
| 原生 prefill（24 token） | 31.2 tokens/s | 约 230 tokens/s |
| 原生 prefill（1024 token） | — | 约 6.1 秒（约 165 tokens/s） |
| WASM decode（wasmtime，标量） | 约 1.2 tokens/s | 约 1.1–1.3 tokens/s |

“优化前”为本次改动之前的 Q4 实现（每组 f32 scale、f32 激活）。KV Cache 改为按头连续存放并对 Q4 模型使用 f16 之前，prompt 1024 的 decode 为 31 tokens/s，prefill 1024 个 token 约需 9.8 秒。单 token decode 每步需要读取全部约 317 MB 权重，另加随上下文增长的 KV Cache，速度受内存带宽限制。该机实测多线程顺序读带宽峰值约 30–33 GB/s，但受其他程序干扰时平均只有约 19–24 GB/s，对应的 decode 上限约为 75–100 tokens/s，当前已接近实际可用带宽。要稳定超过 100 tokens/s，需要更低比特的权重或带宽更高的硬件，这些均未采用，因为前者会降低质量。

`Session::generate` 默认启用推测解码：在会话已有的 token 中查找与当前末尾相同的 2–3 个 token 片段，取其后一个 token 作为草稿，与新采样的 token 组成 2 个 token 的批次一起前向，再按草稿验证规则接受或拒绝。贪心时只在草稿等于 argmax 时接受；采样时以目标分布中草稿的概率接受，拒绝时从去掉草稿后重新归一化的分布中采样，因此生成分布与逐 token 采样完全相同。贪心输出与关闭推测解码时逐 token 一致；采样输出的分布不变，但同一种子下随机数的使用顺序不同，具体序列可能不同。`Session::set_speculation(false)` 可关闭该功能；C ABI 的 `decode`／`sample` 逐 token 接口不受影响。2 个 token 的前向约为单步的 1.2 倍，在 7 组真实多轮对话上实测：代码类约快 10–20%，摘要类约 5–10%，开放式对话约 0–10%，整体约 5–8%（测量噪声约 ±6%）。

**int8 KV Cache（可选）**：`Session::set_kv_precision(KvPrecision::I8)` 把 K、V 存成 int8，每个位置的每个 KV 头另存一个 f32 系数（绝对值最大值 / 127），每个位置约 58 KB，是 f16 的一半。键的系数乘到注意力分数上，值的系数乘到注意力权重上，因此复用同一套分块注意力内核，标量与 AVX2 路径逐位一致。Qwen3 的键里有少数数值很大的维度，按行缩放会让其余维度几乎没有精度；因此 int8 模式下对 Q、K、V 做同一个归一化的 Hadamard 旋转把离群值摊开（Q·K 不变），注意力输出再旋转回来。在 3 段约 1500 token 的真实检索对话上，以 f32 KV Cache 为参照逐位置比较后半段的下一个 token 分布：

| KV Cache | 平均 KL | 最大 KL | top-1 一致率 |
|----------|---------|---------|--------------|
| f16（Q4 模型默认） | 0.0016–0.0018 | 0.028–0.050 | 0.969–0.983 |
| int8 + Hadamard 旋转 | 0.0040–0.0046 | 0.12–0.20 | 0.965–0.975 |
| int8，不旋转 | 0.10–0.14（以 f16 为参照） | 6.5–9.2 | 0.86–0.87 |

int8 的误差约为 f16 的 2.5 倍，但远小于 Q4 权重本身的误差（平均 KL 0.111）。速度收益只在长上下文时出现；测量时机器上同时运行着其他任务，两轮结果互相矛盾，所以还没有可靠的速度数据，默认仍使用 f16。

**decode 时的输入通道稀疏化（可选，实验性）**：`Session::set_input_sparsity(InputSparsity)` 让每个 token 只保留矩阵乘法输入中绝对值最大的一部分通道，被丢弃的通道对应的权重列不再读取。作用于三部分：q/k/v 投影的输入（`attention`）、gate/up 投影的输入（`mlp`）和输出层的输入（`head`），三者约占每步权重读取量的四分之三；o 与 down 只有 1024 行，按列跳读时每个线程每列只读 128 字节，比稠密内核更慢，所以始终完整计算。实现上为 Q4 矩阵另存一份按列存放的副本：每行的 scale 与每行每 32 列的 code 与原矩阵相同，只是半字节按列连续存放，q/k/v 与 gate/up 按行拼接，使每列成为 2–3 KB 的连续数据。因此结果等于把丢弃的通道置零后用原矩阵计算，只有浮点求和顺序不同。第一次启用时建立副本，内存约增加 240 MB（Qwen3-0.6B）。稀疏化只用于逐 token decode；prefill 与推测解码的两 token 校验仍完整计算。默认关闭，只支持 Q4 模型。文本形式为单个比例（三部分相同）或 `attn=0.15,mlp=0.15,head=0.3`，`bench`、`nncoss-know` 与测评脚本的 `--sparsity` 都用这种写法。

在约 1500 token 的真实对话上以完整计算为参照（方法同上表），以及在 24 token prompt、128 token decode 上交替测 4 轮的速度中位数：

| 丢弃比例（attn / mlp / head） | 少读的权重 | 平均 KL | top-1 一致率 | decode 速度 |
|---|---|---|---|---|
| 15% / 15% / 30% | 15% | 0.026 | 0.906 | 1.06 倍 |
| 0 / 0 / 50% | 13% | 0.086 | 0.867 | 1.09 倍 |
| 25% / 25% / 25% | 19% | 0.049 | 0.879 | 1.13 倍 |
| 40% / 40% / 40% | 30% | 0.209 | 0.772 | 1.26 倍 |
| 10% / 10% / 10% | 8% | — | — | 0.94 倍 |

测速时机器处于温度降频（CPU 速度上限在 42%–100% 之间变化），稠密 decode 只有约 45 tokens/s，倍数仅供参考。丢 10% 时选择通道等固定开销超过省下的读取，反而更慢。这些误差都不小：丢 25% 的 KL 约为 Q4 量化本身（0.111）的一半；但在高考单选题上（贪心，不用工具），三部分各丢 25% 不降正确率：108 道测验题上都是 41.7%，另外 373 道题上为 41.3% 对 40.5%，见[实验记录](docs/brain-experiments.md#64-逐-token-的动态结构)。实际生成时提速比上表小：`Session::generate` 默认开启的推测解码会把约 56% 的前向变成两 token 的校验，校验仍完整计算，373 道题的测评用时只少 2.6%。

`tools/check_q4_llm.py` 在内置的 7 段文本（中英文问答、故事、邮件、数学推导、JSON 与 Rust 代码）上按 teacher forcing 比较 Q4 与 f32 模型的下一 token 分布，共 908 个位置。本地 Q4 模型按下文的校准量化与蒸馏生成，top-1 一致率为 0.859，平均 KL(f32‖Q4) 为 0.111，困惑度从 9.489 升至 9.676（比值 1.020）。脚本默认阈值按这组结果留出余量，只用于发现回归；不经校准导出的 Q4 达不到默认阈值。该脚本需要 numpy、tokenizers 与本地导出的模型，不纳入 CI。

#### 校准量化与蒸馏

两步都只改变权重数值，不改变 Q4 文件格式、文件大小与推理速度，运行时代码不需要任何改动：

- **GPTQ 量化**：`tools/export_llm.py --outtype q4 --calibration FILE...` 先用 transformers 以 f32 运行原模型，统计每个线性层输入的二阶矩 E[xxᵀ]，再逐列量化：每列量化后把舍入误差按输入协方差分摊到尚未量化的列，使每层输出的误差最小。组步长仍按最小二乘候选搜索。校准 token 按文件平均分配，默认最多 65,536 个。不传 `--calibration` 时保持原来的最小二乘量化。
- **蒸馏**：`tools/distill_q4_llm.py` 以 f32 原模型为教师，在校准文本上最小化 KL(f32‖Q4)，只训练 Q4 文件中本来就以 f32 保存的参数，即每行 scale、RMSNorm 权重与 QKV bias；4-bit 值与组倍率保持不变。

`tools/calibration/qwen3_chat.txt` 是 Qwen3-0.6B 按 96 条中英文提示（科普、生活、写作、编程、数学、翻译等）生成的对话，与 `check_q4_llm.py` 的评估文本不重叠。只用对话文本校准的效果不如混合文本，因此当前模型同时使用了 README 与几份 Rust、Python 源码。本地 Q4 模型由以下命令生成，在测试机上导出约 25 分钟、蒸馏约 37 分钟，需要 torch 与 transformers：

```bash
python3 tools/export_llm.py --model model/Qwen3-0.6B --output tmp/qwen3-0.6b-q4-gptq.nncossllm --outtype q4 \
  --calibration tools/calibration/qwen3_chat.txt README.md crates/nncoss-llm/src/session.rs \
  crates/nncoss-llm/src/model.rs tools/export_llm.py crates/nncoss-web/src/lib.rs
python3 tools/distill_q4_llm.py --model model/Qwen3-0.6B --input tmp/qwen3-0.6b-q4-gptq.nncossllm \
  --output model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm --steps 300 \
  --calibration tools/calibration/qwen3_chat.txt README.md crates/nncoss-llm/src/session.rs \
  crates/nncoss-llm/src/model.rs tools/export_llm.py crates/nncoss-web/src/lib.rs
```

校准结果取决于具体文本，README 与源码改动后重新生成的模型会有细微差别。同一组 908 个位置上三种导出方式的对比：

| 导出方式 | top-1 一致率 | 平均 KL(f32‖Q4) | 困惑度比值 |
|----------|--------------|-----------------|------------|
| 最小二乘，无校准 | 0.801 | 0.215 | 1.091 |
| GPTQ | 0.849 | 0.137 | 1.066 |
| GPTQ + 蒸馏（当前） | 0.859 | 0.111 | 1.020 |

### 本地 Web 对话界面

`crates/nncoss-web` 提供本地浏览器聊天界面，使用 Rust HTTP 服务、Rust Tokenizer 和 `nncoss-llm` 原生后端（见 [Q4 量化与原生加速](#q4-量化与原生加速)），不包含 JavaScript。模型只加载一次，浏览器通过 HttpOnly Cookie 保持独立会话和 KV Cache；每次生成由全局推理锁串行执行，空消息不会进入历史，页面提供清空对话操作。服务默认只绑定 `127.0.0.1`。

```bash
cargo build -p nncoss-web --release
target/release/nncoss-web
```

随后在浏览器打开 `http://127.0.0.1:8765/`。未指定 `--model` 时，若本地已导出 `model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm` 则优先加载该 Q4 模型（约 304 MiB），否则加载本地导出的约 2.4 GiB f32 模型；启动日志会打印实际加载的路径。f32 模型在上述测试机上 decode 约 11.8 tokens/s、prefill 约 115 tokens/s，受每个 token 读取约 2.4 GB 权重的内存带宽限制。常用参数：

```bash
target/release/nncoss-web --decode 32 --temperature 0.6 --top-k 20 --top-p 0.95
target/release/nncoss-web --model /path/to/model.nncossllm --tokenizer /path/to/tokenizer.json
```

界面不嵌入 WASM 运行时，使用原生后端；WASM 推理仍通过 `nncoss-llm` 的流式 ABI 验证。无模型的 mock 验收会构建服务并检查表单、UTF-8、HTML 转义、Cookie 隔离、空消息和清空操作：

```bash
python3 tools/check_web_ui.py
```

### 基准

`bench` 示例生成确定性的合成 decoder（tiny、small 两档，绑定 LM Head、带 QKV bias），测量模型加载、prefill 与贪心 decode 的中位耗时；WASM 脚本复用同一模型文件与 token 序列，两端输出的 `greedy_checksum` 应一致：

```bash
cargo run -p nncoss-llm --release --example bench -- --preset small
python3 tools/bench_wasm_llm.py --preset small
```

传入 `--model` 时改为测量真实模型：`--prompt`、`--decode` 设置 token 数，默认使用 temperature 0.6／top-k 20／top-p 0.95 采样，`--greedy` 改为贪心；输出 decode 的中位与最佳 tokens/s。`--tokens FILE` 从文件读取逗号或空白分隔的 prompt token；`--generate` 改为比较关闭与开启推测解码时 `Session::generate` 的速度，`--eos` 指定停止 token，贪心时同时检查两者输出是否一致。`--kv f32|f16|i8` 指定 KV Cache 精度；`--kv-compare` 配合 `--tokens` 以 f32 KV Cache 为参照，逐位置比较 f16 与 int8 的下一个 token 分布。`--sparsity SPEC` 启用 decode 时的输入通道稀疏化；`--sparsity-compare SPEC`（可重复）配合 `--tokens` 以完整计算为参照比较各稀疏设置。`examples/sparse_read.rs` 单独测量按块随机跳读内存时能省下多少时间。

基准用于记录基线与优化前后的对比，结果依赖具体机器，不构成性能承诺，也不纳入 CI 门禁。WASM 脚本的 decode 计时包含 Python 宿主逐 token 调用的开销。

### 导出模型

`tools/export_llm.py` 将具有常见 Llama／Qwen 权重命名的本地 safetensors 模型转换为 `.nncossllm`。导出环境需要 `torch` 与 `safetensors`：

```bash
python3 tools/export_llm.py --model /path/to/local-model --output model.nncossllm
```

导出 Q4 时可用 `--calibration` 做 GPTQ 量化，再用 `tools/distill_q4_llm.py` 蒸馏，见[校准量化与蒸馏](#校准量化与蒸馏)。导出器不加载 `.pt/.pth`，仅接受 `model_type` 为 `llama`、`qwen2` 或 `qwen3` 的已知 HalfSplit RoPE 配置，并按名称和形状写入 f32 权重（`--outtype q4` 时二维矩阵改写为 Q4，见 [Q4 量化与原生加速](#q4-量化与原生加速)）；导出日志会记录该 RoPE 约定。当前拒绝 RoPE scaling、partial rotary 和 sliding window attention；Tokenizer、聊天模板与文本解码由调用方处理。`.nncossllm` 是版本为 1 的原型格式，尚未承诺跨版本兼容。

### Rust 调用

```rust
use std::sync::Arc;
use nncoss_llm::{Model, SamplingConfig, Session};

fn generate(model_bytes: &[u8]) -> nncoss_llm::Result<Vec<u32>> {
    let model = Arc::new(Model::from_bytes(model_bytes)?);
    let mut session = Session::new(model, 4096)?;
    session.set_sampler(SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
    }, 42)?;

    let prompt = vec![1, 2, 3];
    session.generate(&prompt, 128, &[4])
}
```

大模型可以从文件按固定大小分块加载，避免整文件缓冲区：

```rust
use std::fs::File;
use std::io::Read;
use nncoss_llm::{Model, ModelLoader};

fn load(path: &str) -> Result<Model, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut loader = ModelLoader::new(usize::try_from(file.metadata()?.len())?);
    let mut chunk = vec![0_u8; 1 << 20];
    loop {
        let length = file.read(&mut chunk)?;
        if length == 0 {
            break;
        }
        loader.write(&chunk[..length])?;
    }
    Ok(loader.finish()?)
}
```

### WASM ABI

宿主先通过 `nncoss_llm_alloc` 分配线性内存并写入模型，再调用 `nncoss_llm_model_load`；加载后可释放输入缓冲区。模型较大时改用分块加载：以模型总字节数调用 `nncoss_llm_model_loader_new`，分配一个固定大小的分块缓冲区，反复写入下一段数据并调用 `nncoss_llm_model_loader_write`，最后调用 `nncoss_llm_model_loader_finish` 取得模型句柄，并用 `nncoss_llm_model_loader_free` 释放加载器。`nncoss_llm_session_new` 创建独立会话，随后循环调用 `nncoss_llm_prefill`、`nncoss_llm_sample` 与 `nncoss_llm_decode`。若采样到 EOS，宿主停止循环且不再 decode 该 token。达到长度上限暂停时，最后一个已采样 token 尚未 decode；恢复同一会话前必须先 decode 它，或 reset 后重建上下文。Rust `Session::generate` 会自动提交所有返回的非 EOS token。

| 接口 | 用途 |
|------|------|
| `nncoss_llm_abi_version` / `nncoss_llm_last_error` | 查询 ABI 版本与当前线程最近的错误消息 |
| `nncoss_llm_alloc` / `nncoss_llm_dealloc` | 分配与释放供宿主写入的线性内存 |
| `nncoss_llm_model_load` / `nncoss_llm_model_free` | 整块加载模型与释放模型句柄 |
| `nncoss_llm_model_loader_new` / `nncoss_llm_model_loader_write` / `nncoss_llm_model_loader_finish` / `nncoss_llm_model_loader_free` | 按声明的总长度分块加载模型；加载器须单独释放 |
| `nncoss_llm_session_new` / `nncoss_llm_session_free` | 创建与释放会话；会话在内部共享模型权重 |
| `nncoss_llm_prefill` / `nncoss_llm_decode` | 推进提示词与生成 token |
| `nncoss_llm_set_sampler` / `nncoss_llm_sample` | 配置采样并从最近 logits 选择 token |
| `nncoss_llm_copy_logits` / `nncoss_llm_session_position` / `nncoss_llm_session_reset` | 读取 logits、查询位置与重置会话 |

返回状态码的操作以 0 表示成功、负数表示失败；`nncoss_llm_alloc` 分配失败返回空指针。模型权重与 KV Cache 扩容的分配失败返回 `-9`，不再中止进程；prefill 失败时会话回退到调用前的位置并保留原有 logits，decode 失败不推进位置。Rust 的 `Error` 标记为 `#[non_exhaustive]`，后续可新增变体，已有状态码保持不变。错误指针在下一次 API 调用前有效。原生 release 的内部 panic 经 unwind 映射为 `-99`；`wasm32-unknown-unknown` 的内部 panic 按目标约定使当前调用 trap，不能依赖 `catch_unwind` 恢复。宿主必须使用匹配的分配长度释放缓冲区，不得并发调用同一会话。释放模型句柄不会销毁仍被会话引用的权重。

分块加载接口是 ABI 版本 1 的增量扩展，已有接口的签名与行为不变；宿主可通过导出符号是否存在判断其可用性。分块可以在任意字节处切分，累计写入不得超过声明的总长度。加载器报告模型格式、长度或分配错误后即释放已解析的权重，后续写入与 `finish` 返回同一错误；空句柄、空指针等参数错误不改变加载器状态。数据未写满就调用 `finish` 按截断处理。`finish` 无论成败只能调用一次，加载器句柄始终需要 `nncoss_llm_model_loader_free` 释放，模型句柄的生命周期与加载器无关。

当前已实现 Q4 权重量化与原生多线程；尚未实现 WASM SIMD、WASM 多线程、GPU、浏览器 JavaScript 宿主和流式文本绑定。f32 权重约占每个参数 4 字节，Q4 约 0.53 字节，模型、KV Cache 与临时张量必须共同满足 wasm32 地址空间及宿主内存限制；超过 `isize::MAX` 字节的模型文件在 wasm32 上只能分块加载，权重总量接近或超过 4 GiB 的模型无法以当前 f32 格式装入 wasm32。

## 知识库与权重改写实验

这部分是“大脑与知识分离”的探索性实验：模型只负责理解和推理，知识放在外部文件里现查现学；同时尝试不经训练、直接改写权重或网络结构来改进模型。方向、判断标准与每项改动的原因见[决策记录](docs/decisions.md)，实验过程与结果见[实验记录](docs/brain-experiments.md)，用到的资料见[外部资料](docs/modelbook.md)，模型见[模型说明](docs/models.md)。以下工具都是实验原型，不属于框架的公开接口。

### 外部知识库：nncoss-knowledge

`crates/nncoss-knowledge` 提供命令行 `nncoss-know`，用 `nncoss-llm` 原生后端运行 Qwen3，并把知识库作为工具交给模型使用：

- **知识库**：段落与模型自己写的笔记分别保存在 `passages.jsonl` 与 `notes.jsonl`，检索用 BM25，中文按相邻两字切分，英文与数字按词切分，笔记的得分加权 1.5 倍。
- **工具**：`search_knowledge` 检索，`write_note` 记笔记，可选的 `browse` 用 curl 打开网页并提取正文；调用格式是 Qwen3 的工具调用模板。模型抄进参数的 LaTeX 反斜杠不构成合法 JSON 转义时，会改成字面反斜杠后再解析；工具轮数用尽时，告诉模型直接作答，再给它一次机会。
- **笔记依据检查**：笔记里的检索词至少 60% 要出现在本题检索到的课本段落里，否则拒绝记录，避免把模型的臆测写进知识库。作答后若检索过资料，会再请模型归纳一条笔记（`--no-learn` 关闭）。
- **对照条件**：`--no-tools` 不提供工具；`--prefetch` 在作答前先用题目自动检索一次，作为“必然查到资料”的对照，配合 `--tool-rounds 0 --no-learn` 时模型只读这次预取的资料、各题互不影响；`--reference` 同样先用题目检索一次，但把查到的段落作为参考资料写在问题前面，不经过工具调用，配合 `--no-tools --no-learn` 时模型只读资料、不能再检索（1.7B 在 `--prefetch` 条件下常常坚持再次调用工具而不作答，见[实验记录 5.9](docs/brain-experiments.md#59-17b只读资料的两种写法)）；`--reply-prefix TEXT` 让第一轮回复由给定文字开头，模型接着写；`--think` 开启思考模式（默认按 Qwen 推荐的 temperature 0.6、top-k 20、top-p 0.95 采样，非思考模式默认贪心）。
- **把握不足才查书（单选题）**：`--consult-below P` 先不用工具作答，再在同一会话里追问“你的最终答案是哪个选项？只回答一个字母”，读出模型给 A–D 的概率；最高概率不低于 P 时采用这次回答，否则按其余选项（如 `--prefetch` 或 `--reference`）查资料重新作答。两次作答的记录都留在 transcript 中，`batch` 输出的 `confidence` 与 `consulted` 记录追问的把握和是否查了资料。阈值 0.8 在两组高考题上合计没有净收益（[实验记录 5.6](docs/brain-experiments.md#56-先自己作答没把握才查书)），这一方式只用于实验。

```bash
target/release/nncoss-know ingest --store modelbook/knowledge/store-base --input modelbook/knowledge/oiwiki.jsonl
target/release/nncoss-know search --store modelbook/knowledge/store-base --query "光合作用 产物"
target/release/nncoss-know ask --store tmp/store-try --model model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm \
  --tokenizer model/Qwen3-0.6B/tokenizer.json --question "光合作用的产物是什么？"
target/release/nncoss-know batch --store <知识库> --model <模型> --tokenizer <分词器> --input <题目.jsonl> --output <结果.jsonl>
target/release/nncoss-know prompt --input <题目.jsonl> --output <提示.jsonl> [--prefetch|--reference --store <知识库>]
target/release/nncoss-know probe --model <模型> --tokenizer <分词器> --input <batch 结果.jsonl> --output <把握.jsonl>
```

`ask` 与 `batch` 会把模型写的笔记存进 `--store` 指向的目录，试用时应先复制一份知识库。`prompt` 只输出模型开口前看到的完整提示，供测量脚本使用。`probe` 对 `batch` 已有的作答记录逐条追问最终选项，输出 A–D 的概率，追问与 `--consult-below` 相同，用来离线比较“把握不足才查书”等作答方式而不必重新作答；记录须以作答回复结束（`--no-learn`）。

### 权重改写：nncoss-edit

`crates/nncoss-edit` 提供命令行 `nncoss-edit`，按 JSON 改动计划直接改写 `.nncossllm` 文件，不做任何训练：

- **缩放**：`{"op": "scale", "target": ..., "layer": ..., "index": [...], "factor": ...}`。`mlp` 的第 i 个神经元对应 up_proj 的第 i 行（SwiGLU 对 up 是线性的，缩放这一行等于缩放该神经元的输出）；`kv_head` 对应 v_proj 中该头的全部行；另可缩放 `q_norm`、`k_norm`、`attn_norm`、`ffn_norm`、`final_norm` 的单个元素和词嵌入 `token` 的行。Q4 文件只改每行的 f32 scale，因此不接受负数因子。
- **删层**：`{"op": "drop_layers", "layers": [...]}` 删除整层，其余张量逐字节保留，层号随后重新连续编号。
- 改动后默认用 `nncoss-llm` 的加载器检查模型能否加载，再原子地写出新文件。

```bash
target/release/nncoss-edit inspect model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm
target/release/nncoss-edit apply --input <模型> --output <新模型> --plan <改动.json> --edit '{"op": "drop_layers", "layers": [20]}'
```

### 测评与测量脚本

以下 Python 脚本只做辅助的抽题、判分与测量，作答都由 `nncoss-know` 完成。脚本不改动参数，改动计划交给 `nncoss-edit` 执行。输出默认写到不入库的 `tmp/`。

| 脚本 | 作用 |
|------|------|
| `tools/quick_eval.py` | 快速考评，满分 100 分：随机抽 6 道高考客观题、2 道主观题、2 道 ACM 题，每题 10 分。ACM 程序在本机用官方测试数据判题；主观题由评审者按参考答案打分后用 `--finalize` 计分。90 分以上优秀、80 分良好、70 分普通、60 分及格，60 分以下为没学会 |
| `tools/eval_gaokao.py` | 从 9 个单选题科目各抽 12 道高考题。`--split test` 用于测验，`--split tune` 用于定位改动，两者不重叠；`--split wide` 是与两者都不重叠的大题组（每科 48 道，物理、语文题少，共 373 道），用于分辨较小的差别。可选择是否思考、是否用工具、是否预取检索或把资料写进提问（`--reference`，`--reference-query other` 换成不对题的资料作对照），以及稀疏化、回复开头与“把握不足才查书”等设置，结果写到 `tmp/<标签>/`；`--no-learn` 时各题互不影响，中断后可用 `--resume` 只答剩下的题 |
| `tools/tool_decision.py` | 在开口位置测量模型选择调用检索的概率，以及各 MLP 神经元与 KV 头对这一选择的梯度归因 |
| `tools/retrieval_gate.py` | 以模型自己的回答为目标（不用标准答案），定位“检索结果为空时拒答、不再自己推理”这一行为相关的神经元和 KV 头，并在留出的一半题上比较候选改动 |
| `tools/layer_importance.py` | 逐层删除并测量模型自己回答的负对数似然变化，给出删层顺序；`--part attn|mlp` 改为只删注意力或 MLP 子层 |
| `tools/repetition_edit.py` | 在模型自己的回答中人为重复一句，测量它继续重复的概率；定位让模型陷入重复循环的神经元与 KV 头，给出改动计划 |
| `tools/dynamic_sparsity.py` | 测量每个 token 只保留一部分 MLP 神经元的代价，以及几种不训练的预判方式（gate 投影、前一个 token、SVD 低秩近似） |
| `tools/moefy.py` | 按神经元的共同激活把稠密 MLP 切成等大小的专家，测量按 token 选专家的代价 |
| `tools/regions.py` | 读题时为每层选出一组神经元作为这道题的“区域”，测量作答时只用这些神经元的代价，并与全局共用区域、随机区域比较 |
| `tools/input_sparsity.py` | 在 f32 模型上测量每个 token 丢弃各矩阵输入中一部分通道的代价，并按参数量估计少读的权重比例 |
| `tools/answer_confidence.py` | 同一道题作答两次（如不用书与先查书）后，追问最终选项并读出 A–D 的概率，检验能否由模型自己的把握决定采用哪一次的答案 |

## 示例模型

每个示例模型计划提供完整的参考实现：模型定义（位于 `nncoss-models`）、训练与推理程序、测试，部分附浏览器演示。实现完成后，它们将同时作为教程与各模块的验收标准；另需验证同一模型通过 WASM 接口与原生 C ABI 被不同语言调用。

| 模型 | 任务 | 数据 / 权重来源 | 重点展示 |
|------|------|-----------------|----------|
| MLP | 手写数字识别 | MNIST | 从零训练、浏览器推理 |
| LeNet-5 | 手写数字识别 | MNIST | 卷积网络、浏览器内训练 |
| ResNet-18 | 图像分类 | CIFAR-10 | 数据增强、GPU 训练、INT8 量化 |
| Transformer | 中文情感分类 | ChnSentiCorp | 嵌入、多头注意力、变长序列 |
| Tiny GPT | 唐诗生成（字符级） | 《全唐诗》 | 自回归生成、KV Cache、浏览器写诗 |
| ResNet-50 | 图像分类 | torchvision 预训练权重 | PyTorch 迁移、数值校验 |
| BERT | 中文文本表示 | Hugging Face `bert-base-chinese` | Hugging Face 迁移、下游微调 |

```bash
cargo run --release --example poetry -- train
cargo run --release --example poetry -- generate --prompt "心有灵犀"
```

## 模型迁移

迁移不只是「把权重读进来」：键名要对得上，布局要转换，数值要一致。吻灵枢规划将这三步整合为工具，并以参考实现和数值对拍验证迁移结果。

- **多种来源**：safetensors（PyTorch / Hugging Face）、PyTorch `.pt` / `.pth`（只解析张量数据，不执行任意代码）、ONNX 计算图、Hugging Face Hub（`config.json` + 权重）。
- **映射与转换**：内置常见模型的键名映射规则，也可用正则自定义；自动处理转置、NCHW / NHWC 布局、QKV 拆分与合并、数据类型转换。
- **数值校验**：与 PyTorch 导出的参考输入输出逐层比对，报告最大误差与余弦相似度，把差异定位到具体层。
- **双向互通**：ONNX 模型可直接推理，也可生成等价的 Rust 模型代码继续训练；nncoss 模型可导出为 ONNX / safetensors，回流到其他生态。

```rust
use nncoss::prelude::*;
use nncoss::migrate::{self, rules};

let mut model = models::resnet50(&Device::auto())?;

// 从 torchvision 权重迁移：按规则改名、转换布局，缺失与多余的键逐一报告
let report = migrate::safetensors("resnet50.safetensors")?
    .rules(rules::torchvision::resnet())
    .load_into(&mut model)?;
println!("{report}");

// 与 PyTorch 导出的参考输入输出逐层比对
migrate::verify(&model, "resnet50.reference.safetensors")?.assert_close(1e-4)?;
```

输出示意：

```text
迁移报告 · resnet50.safetensors → ResNet50
  已映射：267 / 320
  已忽略：53（*.num_batches_tracked，推理与微调都用不到）
  缺失：0 · 形状不符：0
数值校验 · 最大绝对误差 2.4e-5 · 余弦相似度 1.000000 · 通过（容差 1e-4）
```

也可以用命令行一步完成；参考输入输出可用 `tools/export_reference.py` 在 PyTorch 侧导出：

```bash
# 迁移前先看一眼：键名、形状、参数量（ONNX 还会列出算子支持情况）
nncoss inspect resnet50.safetensors

# 映射 + 转换 + 数值校验
nncoss migrate resnet50.safetensors --model resnet50 \
  --verify resnet50.reference.safetensors -o resnet50-nncoss.safetensors
```

## 运行目标

以下均为规划目标，尚未完成平台验收。每个发布产物将列出已验证的平台、WASM 特性、可用算子与设备能力。

| 环境 | 运行方式 | 计算目标 | 典型用途 |
|------|----------|------|----------|
| 浏览器 | WASM 模块 + 浏览器宿主适配 | CPU；SIMD128 与 Web Worker 多线程按能力启用 | 端侧推理与微调 |
| Node.js · Deno · Bun · Cloudflare Workers | WASM 模块 + 对应宿主适配 | CPU；SIMD 按运行环境支持情况启用 | 边缘与 Serverless 推理 |
| WASI 运行时（Wasmtime、WasmEdge 等） | WASM 模块 + 匹配的导入接口或 WASI 适配 | CPU；声明所需 WASM / WASI 版本与特性 | 插件、沙箱化推理服务 |
| 桌面与服务器（Linux · macOS · Windows） | 原生可执行文件、动态库或静态库 | x86_64 / aarch64 等；可选 SIMD 与线程加速 | 训练、迁移、服务端推理与多语言嵌入 |
| 移动设备（Android · iOS 等） | 原生 AOT 库 + 平台语言桥接，或兼容的 WASM 运行环境 | CPU；按平台能力选择加速路径 | App 内推理与微调 |
| 嵌入式及其他平台 | 兼容的 WASM 运行环境，或 LLVM 目标与平台适配 | 按内存、操作系统服务与指令集裁剪能力 | 设备端计算 |
| 支持的原生 GPU | 宿主程序 + 编译后的设备内核 + 驱动适配 | NVIDIA（NVPTX）、AMD（AMDGPU）、OpenCL 设备（SPIR-V） | 加速训练与推理 |

> 浏览器共享内存多线程路径依赖 `SharedArrayBuffer` 与跨源隔离（COOP / COEP），并需相应 WASM 线程构建；另提供单线程产物。Apple GPU 与浏览器 WebGPU 初期使用 CPU 路径。

## 设计原则

- **WASM 核心，双路径部署**：可移植核心与扩展以 WASM 分发，同时支持原生可执行文件、动态库和静态库。
- **编译基于 LLVM**：由语言前端与框架编译器处理输入语义，LLVM 负责优化和目标代码生成，LLD 负责链接。
- **语言无关的调用与扩展**：通过版本化 WASM 接口、原生 C ABI 与宿主桥接接入，各语言共享操作语义和资源约定。
- **AI 功能实现用 Rust，辅助脚本用 Python**：AI 生成或修改的框架与功能代码限于 Rust，脚本工具辅助只使用 Python；人类可用 C/C++ 等任意语言参与，实现须遵守接口与平台约定。
- **可移植性优先**：基础 CPU 路径与可选加速能力分别声明，跨平台的数值与功能一致性由测试验证。
- **错误跨语言可处理**：可恢复错误经接口返回状态码与消息，Rust 映射为 `Result`，其他绑定采用对应语言的错误处理方式；异常不得跨越 ABI 边界。
- **按需裁剪**：只编译用到的内核，借助 LLVM 的 LTO 与死代码消除，产物保持精简。
- **中文优先**：文档与示例中文先行；形状不匹配、设备不一致等常见错误附带中文排查建议。

## 项目结构

当前仓库中已有的内容：

```text
nncoss/
├── Cargo.toml              # workspace
├── crates/
│   ├── nncoss-llm/         # f32／Q4 decoder-only 推理原型，WASM 与原生，C ABI
│   ├── nncoss-web/         # 本地浏览器对话界面，原生后端
│   ├── nncoss-knowledge/   # 外部知识库、检索与笔记，命令行 nncoss-know
│   └── nncoss-edit/        # 直接改写 .nncossllm 的权重与删层，命令行 nncoss-edit
├── modelbook/              # 外部资料：测验题、课本清单、OI Wiki；课本 PDF 与知识库只保存在本机
├── docs/                   # 文档：外部资料、模型、实验记录、CI/CD 验收、命名
├── tools/                  # 辅助脚本（Python）：导出、校验、下载、测评与测量
├── model/                  # 本地模型文件，不入库，来源见 docs/models.md
└── tmp/                    # 临时数据：实验输出、日志与中间文件，不入库
```

以下模块、绑定和示例目录仍是规划结构：

```text
nncoss/
├── src/                  # 门面 crate nncoss：统一 API，按 feature 组合各模块
├── crates/
│   ├── nncoss-engine/    # 引擎：张量、自动微分、内存管理、设备调度
│   ├── nncoss-kernels/   # 内置算子与参考实现：AI 使用 Rust，编译为 WASM
│   ├── nncoss-compiler/  # 编译器：WASM → LLVM IR → CPU 机器码 / GPU 内核
│   ├── nncoss-runtime/   # 运行支持与宿主适配：WASM / 原生执行、CPU 线程与 GPU
│   ├── nncoss-nn/        # 层、激活、嵌入、损失、初始化
│   ├── nncoss-train/     # 训练：优化器、学习率调度、Trainer、回调、断点
│   ├── nncoss-infer/     # 推理：图捕获与优化、整模型编译、量化、推理会话
│   ├── nncoss-models/    # 示例模型：MLP、LeNet、ResNet、Transformer、GPT、BERT
│   ├── nncoss-migrate/   # 模型迁移：多来源导入、映射与转换、数值校验
│   ├── nncoss-abi/       # Rust 实现的版本化接口：WASM 接口与原生 C ABI
│   ├── nncoss-extensions/ # 扩展协议、注册与能力校验
│   └── nncoss-cli/       # 命令行：build / compile / inspect / migrate
├── bindings/             # 各语言绑定；非 Rust 程序代码由人类维护
│   ├── js/               # JavaScript / TypeScript SDK
│   └── python/           # Python 绑定；其他语言按需加入
└── examples/             # 可运行示例：mnist、cifar10、poetry、web-mnist、web-poetry……
```

## 路线图

路线图按可验证的纵向切片推进：每个阶段以可运行产物、可复现测试和明确的支持范围为完成标准。后续能力可以先行设计，但在对应验收完成前不作为已支持功能宣传。

- [ ] **v0.1 最小计算与双路径验证**：限定 f32 张量、逐元素 / 归约 / 矩阵乘标量内核和固定结构 MLP 前向计算，形成最小张量与会话接口草案；实现一种 WASM 运行时路径和一个原生 CPU 的 LLVM AOT 可执行产物。验收：同一组输入与权重在两条路径上通过数值对拍，并记录目标特性、工具链版本和误差容限
- [ ] **v0.2 MNIST 训练闭环**：实现反向自动微分、Linear、交叉熵、SGD、数据加载、基础训练循环与断点保存。验收：MLP 在 MNIST 上完成训练、断点恢复和推理，同一模型在 WASM 与已支持原生 CPU 上获得容差内一致的结果
- [ ] **v0.3 多语言调用与扩展**：提供版本化 WASM 接口和 C ABI，补齐原生动态库与静态库，加入语言绑定、WASM 扩展与原生扩展注册、宿主回调。验收：至少两种宿主语言调用同一 MNIST 模型并使用同一 WASM 算子扩展；非 Rust 语言绑定由人类贡献
- [ ] **v0.4 推理与部署**：图捕获与优化、整模型 WASM 编译、原生模型库与可执行产物、推理会话。验收：同一模型经浏览器、原生程序及外部语言绑定获得容差内一致的结果
- [ ] **v0.5 模型迁移**：明确 PyTorch / Hugging Face / ONNX 支持范围，实现映射规则、数值校验与迁移工具。验收：ResNet-50 / BERT 参考模型迁移及逐层校验
- [ ] **v0.6 GPU 与高级推理**：按后端逐步验证内核子集，接入 NVPTX / AMDGPU / SPIR-V，推进 f16、INT8 与 KV Cache。验收：受支持算子的 CPU / GPU 对拍、训练与推理基准
- [ ] **v1.0 稳定**：冻结公开 API 与 ABI，发布平台及扩展兼容性矩阵、中文文档和多语言接入教程
- **持续扩展**：移动与嵌入式平台适配、更多语言绑定、Apple GPU 与浏览器 WebGPU 路径、riscv64 向量扩展调优

## 名字由来

| 名字 | 拆解 | 出处 |
|------|------|------|
| **nncoss** | **nn**（neural network，神经网络）+ **coss**（吻） | *coss* 为古英语名词"吻"，动词作 *cyssan*，后演变为现代英语 *kiss* |
| **吻灵枢** | **吻** + **灵枢**（神经中枢） | 吻，《说文解字》释"口边也"；灵枢，即《黄帝内经·灵枢经》，主论经络腧穴 |

合起来，便是「神经网络之吻」。取古形 *coss* 而不用 *kiss*，既为简短，也为添一层时间的浪漫——千年以前的吻，至今未变。

中文名读作 wěn líng shū。动词前置，有《楚辞》句式之韵，如一声私语；乍看又像医典篇目、针灸穴名。灵枢古称《针经》——一吻落下，正中穴位，一针见血。

古人早有"网络"之梦：经络之中气血传信，《华严经》里因陀罗网宝珠互映，《列子·汤问》中偃师造出能歌善舞的假人。吻灵枢借这些旧梦，做一件新事。

完整的词源考据与文化背景见 [docs/naming.md](docs/naming.md)。

### 命名彩蛋

深度学习里的几个术语，恰好都能读出另一层意思：

| 术语 | 技术含义 | 另一层意思 |
|------|----------|------------|
| 深吻 · deep kiss | 深度学习（deep learning） | 深度学习之吻 |
| 激活 · activation | 激活函数，决定神经元是否被点亮 | 一吻激活 |
| 嵌入 · embedding | 把离散符号映射为稠密向量 | 你嵌入我心里 |
| 收敛 · converge | 损失趋于稳定，训练完成 | 爱已定 |

所以训练收敛时，吻灵枢会在日志末尾轻轻补上一句：「梯度下降千万次，终于收敛于你。」（可关闭）

## 参与贡献

项目刚刚起步，欢迎围绕平台适配、调用接口、扩展协议与神经网络能力讨论设计。代码贡献请先开 Issue 对齐方案，再提交 PR。

- **AI 贡献**：框架、算子、扩展、语言绑定等功能代码必须使用 Rust；辅助脚本只能使用 Python，用于数据处理、结果校验与开发自动化等工作。AI 可以维护文档与非程序配置；新增功能代码示例使用 Rust，辅助脚本示例只使用 Python。
- **人类贡献**：人类可以使用 Rust、C/C++、Python、JavaScript 等任意语言，维护相应语言的绑定、示例与工具，并声明所需工具链和运行环境。
- **AI 提交流程**：默认在独立工作分支修改、验证和提交，通过 PR 合并进入主分支。只有人类明确确认允许直接提交主分支时才可例外，授权限于已确认的范围。未配置远程仓库或无法创建 PR 时，保留工作分支并说明情况。
- **CI/CD 验收**：PR 最新提交必须通过 GitHub 上全部必需的 CI/CD 验收检查，才能合并。检查未配置、未运行、失败、取消、跳过或状态不明时均不得合并；本地检查通过不能替代 GitHub 验收。需在 GitHub 规则集或分支保护中配置相应的合并门禁。
- **接口与平台**：贡献需遵守版本化调用或扩展协议。可移植核心与 WASM 扩展需通过所声明的 WASM / 原生路径测试；平台专用扩展需明确支持范围并在对应平台验证。
- **验证与示例**：新内核需附带标量参考实现和数值对拍；新语言绑定需验证资源释放与错误传递；新功能需附带可复现示例。

AI 协作的仓库规则见 [AGENTS.md](AGENTS.md)。当前文档阶段的检查范围、本地运行方式及 GitHub 门禁配置见 [CI/CD 验收](docs/ci.md)。

## 致谢

吻灵枢的设计借鉴了 Rust 生态中的 [burn](https://github.com/tracel-ai/burn)、[candle](https://github.com/huggingface/candle) 与 [tract](https://github.com/sonos/tract)；LLM 会话与 KV Cache 原型参考了 [ncnn](https://github.com/Tencent/ncnn) 和 [ncnn_llm](https://github.com/futz12/ncnn_llm)；WASM 到本机代码的编译思路参考了 [WAMR](https://github.com/wasm-micro-runtime/wasm-micro-runtime)、[WasmEdge](https://github.com/WasmEdge/WasmEdge) 等基于 LLVM 的 AOT 编译器；整个项目构建于 [LLVM](https://llvm.org/) 之上，在此致谢。

## 许可证

尚未设置许可证。

---

<div align="center">
<sub>始于 2026 年秋。千年古词，一吻如新。</sub>
</div>
