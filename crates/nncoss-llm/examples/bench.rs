use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use nncoss_llm::{
    InputSparsity, KvPrecision, MODEL_FORMAT_VERSION, Model, SamplingConfig, Session,
};

#[derive(Clone, Copy)]
struct Preset {
    name: &'static str,
    vocab: u32,
    hidden: u32,
    intermediate: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    prompt: usize,
    decode: usize,
}

const PRESETS: [Preset; 2] = [
    Preset {
        name: "tiny",
        vocab: 512,
        hidden: 64,
        intermediate: 176,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        prompt: 32,
        decode: 32,
    },
    Preset {
        name: "small",
        vocab: 4096,
        hidden: 256,
        intermediate: 704,
        layers: 4,
        heads: 8,
        kv_heads: 2,
        head_dim: 32,
        prompt: 64,
        decode: 64,
    },
];

impl Preset {
    fn max_context(&self) -> usize {
        self.prompt + self.decode
    }

    fn prompt_token(&self, index: usize) -> u32 {
        ((index * 31 + 7) % self.vocab as usize) as u32
    }

    fn parameters(&self) -> u64 {
        let (v, h, i) = (
            u64::from(self.vocab),
            u64::from(self.hidden),
            u64::from(self.intermediate),
        );
        let q = u64::from(self.heads * self.head_dim);
        let kv = u64::from(self.kv_heads * self.head_dim);
        let layer = h + q * h + 2 * kv * h + q + 2 * kv + h * q + h + 3 * i * h;
        v * h + u64::from(self.layers) * layer + h
    }

    fn model_bytes(&self) -> Vec<u8> {
        let q = (self.heads * self.head_dim) as usize;
        let kv = (self.kv_heads * self.head_dim) as usize;
        let (v, h, i) = (
            self.vocab as usize,
            self.hidden as usize,
            self.intermediate as usize,
        );
        let mut out = Vec::with_capacity(64 + self.parameters() as usize * 4);
        out.extend_from_slice(b"NNCSLLM\0");
        for value in [
            MODEL_FORMAT_VERSION,
            1,
            self.vocab,
            self.hidden,
            self.intermediate,
            self.layers,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.max_context() as u32,
            1 | 2,
            0,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&1e-6_f32.to_le_bytes());
        out.extend_from_slice(&10_000.0_f32.to_le_bytes());
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
        let mut random = |out: &mut Vec<u8>, count: usize, scale: f32| {
            for _ in 0..count {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let unit = (rng >> 40) as f32 / (1_u32 << 24) as f32;
                out.extend_from_slice(&((unit * 2.0 - 1.0) * scale).to_le_bytes());
            }
        };
        let ones = |out: &mut Vec<u8>, count: usize| {
            for _ in 0..count {
                out.extend_from_slice(&1.0_f32.to_le_bytes());
            }
        };
        let scale = |fan_in: usize| (fan_in as f32).sqrt().recip();
        random(&mut out, v * h, 1.0);
        for _ in 0..self.layers {
            ones(&mut out, h);
            random(&mut out, q * h, scale(h));
            random(&mut out, kv * h, scale(h));
            random(&mut out, kv * h, scale(h));
            random(&mut out, q + 2 * kv, 0.02);
            random(&mut out, h * q, scale(q));
            ones(&mut out, h);
            random(&mut out, i * h, scale(h));
            random(&mut out, i * h, scale(h));
            random(&mut out, h * i, scale(i));
        }
        ones(&mut out, h);
        out
    }
}

struct Options {
    preset: Preset,
    iterations: usize,
    write_model: Option<String>,
    model: Option<String>,
    prompt: usize,
    decode: usize,
    greedy: bool,
    tokens: Option<String>,
    generate: bool,
    print_tokens: bool,
    eos: Vec<u32>,
    kv: Option<KvPrecision>,
    kv_compare: bool,
    sparsity: InputSparsity,
    sparsity_compare: Vec<InputSparsity>,
}

fn parse_options() -> Result<Options, String> {
    let mut preset = PRESETS[0];
    let mut iterations = 5;
    let mut write_model = None;
    let mut model = None;
    let mut prompt = 24;
    let mut decode = 64;
    let mut greedy = false;
    let mut tokens = None;
    let mut generate = false;
    let mut print_tokens = false;
    let mut eos = Vec::new();
    let mut kv = None;
    let mut kv_compare = false;
    let mut sparsity = InputSparsity::NONE;
    let mut sparsity_compare = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} 需要参数"));
        match arg.as_str() {
            "--preset" => {
                let name = value()?;
                preset = *PRESETS
                    .iter()
                    .find(|preset| preset.name == name)
                    .ok_or(format!("未知 preset：{name}（可选 tiny、small）"))?;
            }
            "--iterations" => {
                iterations = value()?
                    .parse()
                    .map_err(|_| "--iterations 需要正整数".to_string())?;
                if iterations == 0 {
                    return Err("--iterations 需要正整数".into());
                }
            }
            "--write-model" => write_model = Some(value()?),
            "--model" => model = Some(value()?),
            "--prompt" => {
                prompt = value()?
                    .parse()
                    .ok()
                    .filter(|count| *count > 0)
                    .ok_or("--prompt 需要正整数")?;
            }
            "--decode" => {
                decode = value()?
                    .parse()
                    .ok()
                    .filter(|count| *count > 0)
                    .ok_or("--decode 需要正整数")?;
            }
            "--greedy" => greedy = true,
            "--kv" => {
                kv = Some(match value()?.as_str() {
                    "f32" => KvPrecision::F32,
                    "f16" => KvPrecision::F16,
                    "i8" => KvPrecision::I8,
                    other => return Err(format!("--kv 只能是 f32、f16 或 i8，收到 {other}")),
                });
            }
            "--tokens" => tokens = Some(value()?),
            "--generate" => generate = true,
            "--print-tokens" => print_tokens = true,
            "--kv-compare" => kv_compare = true,
            "--sparsity" => {
                sparsity = value()?
                    .parse()
                    .map_err(|error| format!("--sparsity：{error}"))?;
            }
            "--sparsity-compare" => {
                let level = value()?
                    .parse()
                    .map_err(|error| format!("--sparsity-compare：{error}"))?;
                sparsity_compare.push(level);
            }
            "--eos" => {
                eos = value()?
                    .split(',')
                    .map(|token| token.trim().parse::<u32>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| "--eos 需要逗号分隔的 token id".to_string())?;
            }
            "--help" | "-h" => {
                return Err(
                    "用法：bench [--preset tiny|small] [--iterations N] [--write-model PATH]\n      bench --model PATH [--prompt N | --tokens FILE] [--decode N] [--greedy] [--print-tokens] [--kv f32|f16|i8] [--sparsity DROP|attn=…,mlp=…,head=…] [--generate [--eos IDS]] [--iterations N]\n      bench --model PATH --tokens FILE --kv-compare\n      bench --model PATH --tokens FILE --sparsity-compare SPEC [--sparsity-compare SPEC …]"
                        .into(),
                );
            }
            _ => return Err(format!("未知参数：{arg}")),
        }
    }
    Ok(Options {
        preset,
        iterations,
        write_model,
        model,
        prompt,
        decode,
        greedy,
        tokens,
        generate,
        print_tokens,
        eos,
        kv,
        kv_compare,
        sparsity,
        sparsity_compare,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run_model(path: &str, options: &Options) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("读取 {path} 失败：{error}"))?;
    let started = Instant::now();
    let model = Arc::new(Model::from_bytes(&bytes).map_err(|error| error.to_string())?);
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    drop(bytes);
    let vocab = model.config().vocab_size;
    let prompt: Vec<u32> = match &options.tokens {
        Some(file) => std::fs::read_to_string(file)
            .map_err(|error| format!("读取 {file} 失败：{error}"))?
            .split(|character: char| character == ',' || character.is_whitespace())
            .filter(|token| !token.is_empty())
            .map(|token| {
                token
                    .parse::<u32>()
                    .map_err(|_| format!("{file} 包含无效 token：{token}"))
            })
            .collect::<Result<_, _>>()?,
        None => (0..options.prompt)
            .map(|index| ((index * 7919 + 1000) % vocab.min(100_000)) as u32)
            .collect(),
    };
    if prompt.is_empty() {
        return Err("prompt 不能为空".into());
    }
    let context = prompt.len() + options.decode + 1;
    let sampling = if options.greedy {
        SamplingConfig::default()
    } else {
        SamplingConfig {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
        }
    };
    if options.generate {
        return compare_generation(&model, &prompt, context, sampling, options);
    }
    if options.kv_compare {
        return compare_kv(&model, &prompt);
    }
    if !options.sparsity_compare.is_empty() {
        return compare_sparsity(&model, &prompt, &options.sparsity_compare);
    }
    let mut prefill_ms = Vec::new();
    let mut decode_ms = Vec::new();
    let mut generated = Vec::new();
    for iteration in 0..=options.iterations {
        let mut session = Session::new(model.clone(), context).map_err(|e| e.to_string())?;
        if let Some(precision) = options.kv {
            session.set_kv_precision(precision);
        }
        session
            .set_input_sparsity(options.sparsity)
            .map_err(|e| e.to_string())?;
        session
            .set_sampler(sampling, 42)
            .map_err(|e| e.to_string())?;
        let started = Instant::now();
        session.prefill(&prompt).map_err(|e| e.to_string())?;
        let prefill = started.elapsed().as_secs_f64() * 1e3;
        let mut tokens = Vec::with_capacity(options.decode);
        let started = Instant::now();
        for _ in 0..options.decode {
            let token = session.sample_next().map_err(|e| e.to_string())?;
            session.decode(token).map_err(|e| e.to_string())?;
            tokens.push(token);
        }
        let decode = started.elapsed().as_secs_f64() * 1e3;
        if iteration > 0 {
            prefill_ms.push(prefill);
            decode_ms.push(decode);
        }
        if !generated.is_empty() && generated != tokens {
            return Err("同一输入与种子的解码在迭代之间不一致".into());
        }
        generated = tokens;
    }
    let prefill = median(&mut prefill_ms);
    let decode = median(&mut decode_ms);
    let fastest = decode_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let checksum = generated.iter().fold(0_u64, |acc, token| {
        acc.wrapping_mul(31).wrapping_add(u64::from(*token))
    });
    println!("target: native ({})", std::env::consts::ARCH);
    println!(
        "model: {path} prompt={} decode={} iterations={} sampler={}",
        prompt.len(),
        options.decode,
        options.iterations,
        if options.greedy {
            "greedy"
        } else {
            "t0.6/k20/p0.95"
        }
    );
    println!(
        "load_ms: {load_ms:.2}  input_sparsity: {}",
        options.sparsity
    );
    println!(
        "prefill_ms: {prefill:.2}  prefill_tok_s: {:.1}",
        prompt.len() as f64 / prefill * 1e3
    );
    println!(
        "decode_ms_per_token: {:.3}  decode_tok_s: {:.1}  best_decode_tok_s: {:.1}",
        decode / options.decode as f64,
        options.decode as f64 / decode * 1e3,
        options.decode as f64 / fastest * 1e3
    );
    println!("token_checksum: {checksum}");
    if options.print_tokens {
        let ids: Vec<String> = generated.iter().map(|token| token.to_string()).collect();
        println!("generated_tokens: {}", ids.join(","));
    }
    Ok(())
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = logits
        .iter()
        .map(|value| f64::from(value - maximum).exp())
        .sum();
    let offset = f64::from(maximum) + total.ln();
    logits
        .iter()
        .map(|value| f64::from(*value) - offset)
        .collect()
}

/// 以 f32 缓存为参照：前一半 token 作为 prompt，后一半逐个强制输入，
/// 比较每个位置上 f16 与 int8 缓存给出的下一个 token 分布（KL 与首选一致率）。
fn compare_kv(model: &Arc<Model>, tokens: &[u32]) -> Result<(), String> {
    let mut sessions = Vec::new();
    for (name, precision) in [
        ("f32", KvPrecision::F32),
        ("f16", KvPrecision::F16),
        ("i8", KvPrecision::I8),
    ] {
        let mut session = Session::new(model.clone(), tokens.len()).map_err(|e| e.to_string())?;
        session.set_kv_precision(precision);
        sessions.push((name.to_string(), session));
    }
    compare_sessions("kv_compare", tokens, sessions)
}

/// 以完整计算为参照，比较 decode 时丢弃各比例输入通道后的下一个 token 分布。
fn compare_sparsity(
    model: &Arc<Model>,
    tokens: &[u32],
    levels: &[InputSparsity],
) -> Result<(), String> {
    let mut sessions = Vec::new();
    for level in std::iter::once(InputSparsity::NONE).chain(levels.iter().copied()) {
        let mut session = Session::new(model.clone(), tokens.len()).map_err(|e| e.to_string())?;
        session
            .set_input_sparsity(level)
            .map_err(|e| e.to_string())?;
        sessions.push((level.to_string(), session));
    }
    compare_sessions("sparsity_compare", tokens, sessions)
}

/// 第一个会话为参照：前一半 token 作为 prompt，后一半逐个强制输入，
/// 比较其余会话在每个位置上给出的下一个 token 分布（KL 与首选一致率）。
fn compare_sessions(
    title: &str,
    tokens: &[u32],
    mut sessions: Vec<(String, Session)>,
) -> Result<(), String> {
    if tokens.len() < 4 {
        return Err(format!("{title} 至少需要 4 个 token"));
    }
    let split = tokens.len() / 2;
    for (_, session) in &mut sessions {
        session
            .prefill(&tokens[..split])
            .map_err(|e| e.to_string())?;
    }
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
    };
    let mut totals = vec![(0.0_f64, 0.0_f64, 0_usize); sessions.len() - 1];
    let mut count = 0;
    for (index, token) in tokens[split..].iter().enumerate() {
        let reference = sessions[0].1.last_logits().map_err(|e| e.to_string())?;
        let reference_log = log_softmax(reference);
        for (slot, (_, session)) in sessions[1..].iter().enumerate() {
            let candidate = session.last_logits().map_err(|e| e.to_string())?;
            let divergence: f64 = reference_log
                .iter()
                .zip(log_softmax(candidate))
                .map(|(p, q)| p.exp() * (p - q))
                .sum();
            let (total, worst, agree) = &mut totals[slot];
            *total += divergence;
            *worst = worst.max(divergence);
            *agree += usize::from(argmax(reference) == argmax(candidate));
        }
        count += 1;
        if index + 1 < tokens.len() - split {
            for (_, session) in &mut sessions {
                session.decode(*token).map_err(|e| e.to_string())?;
            }
        }
    }
    let reference = &sessions[0].0;
    for ((name, _), (total, worst, agree)) in sessions[1..].iter().zip(totals) {
        println!(
            "{title} {name} vs {reference}: positions={count} context={split}..{} mean_kl={:.6} max_kl={:.6} top1_agree={:.4}",
            tokens.len(),
            total / count as f64,
            worst,
            agree as f64 / count as f64
        );
    }
    Ok(())
}

fn compare_generation(
    model: &Arc<Model>,
    prompt: &[u32],
    context: usize,
    sampling: SamplingConfig,
    options: &Options,
) -> Result<(), String> {
    let (history, last) = prompt.split_at(prompt.len() - 1);
    let mut outputs = Vec::new();
    for speculation in [false, true] {
        let mut rates = Vec::new();
        let mut counts = (0, 0);
        let mut output = Vec::new();
        for iteration in 0..=options.iterations {
            let mut session = Session::new(model.clone(), context).map_err(|e| e.to_string())?;
            session
                .set_sampler(sampling, 42)
                .map_err(|e| e.to_string())?;
            session.set_speculation(speculation);
            session
                .set_input_sparsity(options.sparsity)
                .map_err(|e| e.to_string())?;
            if !history.is_empty() {
                session.prefill(history).map_err(|e| e.to_string())?;
            }
            let started = Instant::now();
            let generated = session
                .generate(last, options.decode, &options.eos)
                .map_err(|e| e.to_string())?;
            let seconds = started.elapsed().as_secs_f64();
            if iteration > 0 {
                rates.push(generated.len() as f64 / seconds);
            }
            counts = session.speculation_counts();
            output = generated;
        }
        let rate = median(&mut rates);
        println!(
            "speculation={speculation}: generated={} tokens_per_s={rate:.1} drafted={} accepted={}",
            output.len(),
            counts.0,
            counts.1
        );
        outputs.push(output);
    }
    if options.greedy {
        println!("greedy_outputs_identical: {}", outputs[0] == outputs[1]);
    }
    Ok(())
}

fn run(options: Options) -> Result<(), String> {
    if let Some(path) = options.model.clone() {
        return run_model(&path, &options);
    }
    let preset = options.preset;
    let bytes = preset.model_bytes();
    if let Some(path) = options.write_model {
        std::fs::write(&path, &bytes).map_err(|error| format!("写入 {path} 失败：{error}"))?;
        println!("wrote {path} ({} bytes)", bytes.len());
        return Ok(());
    }
    let mut load_ms = Vec::new();
    let mut loaded = None;
    for iteration in 0..=options.iterations {
        drop(loaded.take());
        let started = Instant::now();
        let model = Model::from_bytes(&bytes).map_err(|error| error.to_string())?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        if iteration > 0 {
            load_ms.push(elapsed);
        }
        loaded = Some(model);
    }
    let load_ms = median(&mut load_ms);
    let model = Arc::new(loaded.ok_or("模型加载次数为 0")?);
    let prompt: Vec<u32> = (0..preset.prompt)
        .map(|index| preset.prompt_token(index))
        .collect();

    let mut prefill_ms = Vec::new();
    let mut decode_ms = Vec::new();
    let mut generated = Vec::new();
    for iteration in 0..=options.iterations {
        let mut session =
            Session::new(model.clone(), preset.max_context()).map_err(|e| e.to_string())?;
        let started = Instant::now();
        session.prefill(&prompt).map_err(|e| e.to_string())?;
        let prefill = started.elapsed().as_secs_f64() * 1e3;
        let mut tokens = Vec::with_capacity(preset.decode);
        let started = Instant::now();
        for _ in 0..preset.decode {
            let token = session.sample_next().map_err(|e| e.to_string())?;
            session.decode(token).map_err(|e| e.to_string())?;
            tokens.push(token);
        }
        let decode = started.elapsed().as_secs_f64() * 1e3;
        if iteration > 0 {
            prefill_ms.push(prefill);
            decode_ms.push(decode);
        }
        if !generated.is_empty() && generated != tokens {
            return Err("同一输入的贪心解码在迭代之间不一致".into());
        }
        generated = tokens;
    }
    let prefill = median(&mut prefill_ms);
    let decode = median(&mut decode_ms);
    let checksum = generated.iter().fold(0_u64, |acc, token| {
        acc.wrapping_mul(31).wrapping_add(u64::from(*token))
    });
    println!("target: native ({})", std::env::consts::ARCH);
    println!(
        "preset: {} params={} model_bytes={} prompt={} decode={} iterations={}",
        preset.name,
        preset.parameters(),
        bytes.len(),
        preset.prompt,
        preset.decode,
        options.iterations
    );
    println!("load_ms: {load_ms:.2}");
    println!(
        "prefill_ms: {prefill:.2}  prefill_tok_s: {:.1}",
        preset.prompt as f64 / prefill * 1e3
    );
    println!(
        "decode_ms_per_token: {:.3}  decode_tok_s: {:.1}",
        decode / preset.decode as f64,
        preset.decode as f64 / decode * 1e3
    );
    println!("greedy_checksum: {checksum}");
    Ok(())
}

fn main() -> ExitCode {
    match parse_options().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
