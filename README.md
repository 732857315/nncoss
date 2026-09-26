<div align="center">

# 吻灵枢 · nncoss

**基于 WebAssembly，面向任意平台，支持任意语言调用与扩展的神经网络框架**

A WebAssembly-based neural network framework for any platform and language, with LLVM-powered native compilation

*吻灵枢，一点通。*

![status](https://img.shields.io/badge/status-pre--alpha-orange)
![compiler](https://img.shields.io/badge/compiler-LLVM-262d3a)
![deployment](https://img.shields.io/badge/deployment-WASM%20%2B%20native-654ff0)
![interface](https://img.shields.io/badge/interface-language%20neutral-dea584)

[项目定位](#项目定位) · [编译流水线](#编译流水线) · [多语言调用](#多语言调用) · [扩展机制](#扩展机制) · [快速开始](#快速开始) · [引擎](#引擎) · [训练](#训练) · [推理](#推理) · [模型迁移](#模型迁移) · [路线图](#路线图)

</div>

---

吻灵枢（nncoss）是一个**基于 WebAssembly、以 LLVM 为编译基础、面向任意平台的神经网络框架**。它支持 **WASM 与原生二进制**两种部署形式，通过语言无关的接口，让任何语言都能调用框架并扩展其能力，覆盖引擎、训练、推理、示例模型与模型迁移。

WASM 承载可移植的计算与扩展模块；LLVM 负责优化与目标代码生成。同一模块既可以交给兼容的 WASM 运行环境执行，也可以提前编译为目标平台的可执行文件、动态库或静态库。应用与扩展通过统一协议接入，具体平台由运行时和宿主适配层连接。

「枢」是门轴，也是中枢。吻灵枢想做模型的枢纽：外部模型迁移进来，自己的模型训练出来，最终都能部署出去；同时保持克制，只接管你不想写的部分，点到为止。

> **项目状态**：当前仓库仅有设计文档，尚无可安装的软件包或可运行的示例。下文的能力、平台、API 与命令均为设计目标或使用草案，实际支持情况将随实现与测试公布。

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

| 模块 | 负责什么 | 关键能力 |
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

## 快速开始

> 以下是实现完成后的预期使用流程，当前命令与示例尚不可运行，仓库地址为占位符。

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

引擎提供张量计算与自动微分，在 WASM、原生 CPU 与支持的 GPU 后端保持算子语义一致，并按数据类型与运算定义数值误差容限。其余模块及各语言接口都构建于此。

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

## 示例模型

每个示例模型计划提供完整的参考实现：模型定义（位于 `nncoss-models`）、训练与推理程序、测试，部分附浏览器演示。它们既是教程，也是各模块的验收标准；另需验证同一模型通过 WASM 接口与原生 C ABI 被不同语言调用。

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

迁移不只是「把权重读进来」：键名要对得上，布局要转换，数值要一致。吻灵枢把这三步都做成了工具。

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

## 项目结构（规划）

```text
nncoss/
├── Cargo.toml            # workspace，同时是门面 crate nncoss
├── src/                  # 门面：统一 API，按 feature 组合各模块
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
├── examples/             # 可运行示例：mnist、cifar10、poetry、web-mnist、web-poetry……
└── tools/                # 辅助工具：AI 功能实现用 Rust，辅助脚本只用 Python
```

## 路线图

- [ ] **v0.1 可移植核心与原生编译**：f32 张量、逐元素 / 归约 / 矩阵乘内核、接口草案；WASM 执行与 LLVM 原生 AOT，覆盖可执行文件和库产物。验收：同一模块在 WASM 与已支持原生 CPU 上通过数值对拍
- [ ] **v0.2 多语言调用与扩展**：版本化 WASM 接口和 C ABI、语言绑定、WASM 扩展与原生扩展注册、宿主回调。验收：至少两种宿主语言调用同一模型并使用同一 WASM 算子扩展；非 Rust 语言绑定由人类贡献
- [ ] **v0.3 训练**：自动微分、常用层与损失、SGD / AdamW、Trainer、断点与 safetensors 读写。验收：MLP / LeNet-5 训练、推理与断点恢复
- [ ] **v0.4 推理与部署**：图捕获与优化、整模型 WASM 编译、原生模型库与可执行产物、推理会话。验收：同一模型经浏览器、原生程序及外部语言绑定获得容差内一致的结果
- [ ] **v0.5 模型迁移**：PyTorch / Hugging Face / ONNX 支持范围、映射规则、数值校验与迁移工具。验收：ResNet-50 / BERT 参考模型迁移及逐层校验
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

AI 协作的仓库规则见 [AGENTS.md](AGENTS.md)。

## 致谢

吻灵枢的设计借鉴了 Rust 生态中的 [burn](https://github.com/tracel-ai/burn)、[candle](https://github.com/huggingface/candle) 与 [tract](https://github.com/sonos/tract)；WASM 到本机代码的编译思路参考了 [WAMR](https://github.com/wasm-micro-runtime/wasm-micro-runtime)、[WasmEdge](https://github.com/WasmEdge/WasmEdge) 等基于 LLVM 的 AOT 编译器；整个项目构建于 [LLVM](https://llvm.org/) 之上，在此致谢。

## 许可证

尚未设置许可证。

---

<div align="center">
<sub>始于 2026 年秋。千年古词，一吻如新。</sub>
</div>
