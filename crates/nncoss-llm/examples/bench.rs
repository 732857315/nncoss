use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use nncoss_llm::{MODEL_FORMAT_VERSION, Model, Session};

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
}

fn parse_options() -> Result<Options, String> {
    let mut preset = PRESETS[0];
    let mut iterations = 5;
    let mut write_model = None;
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
            "--help" | "-h" => {
                return Err(
                    "用法：bench [--preset tiny|small] [--iterations N] [--write-model PATH]"
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
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run(options: Options) -> Result<(), String> {
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
