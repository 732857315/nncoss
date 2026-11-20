use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use nncoss_knowledge::{
    Agent, AgentConfig, KnowledgeBase, opening_prompt, prefetch_prompt, reference_prompt,
};
use nncoss_llm::{KvPrecision, Model, ModelLoader};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

const USAGE: &str = "用法：
  nncoss-know ingest --store <目录> --input <段落.jsonl>
  nncoss-know search --store <目录> --query <问题> [--limit N]
  nncoss-know ask    --store <目录> --model <模型> --tokenizer <分词器> --question <问题> [选项]
  nncoss-know batch  --store <目录> --model <模型> --tokenizer <分词器> --input <题目.jsonl> --output <结果.jsonl> [选项]
  nncoss-know prompt (--question <问题> | --input <题目.jsonl> --output <提示.jsonl>) [--prefetch|--reference --store <目录>] [选项]

选项：
  --think                启用思考模式
  --no-tools             不提供任何工具（模型不能自己检索或记笔记）
  --browse               提供 browse 工具，允许模型打开网页
  --no-learn             答题后不自动归纳笔记
  --prefetch             作答前先用题目自动检索一次知识库，写成模型的一次工具调用（对照实验用）
  --reference            作答前先用题目检索一次，把查到的段落作为参考资料写在问题前面；
                         配合 --no-tools --no-learn 时只读资料、各题互不影响（对照实验用）
  --turn-tokens N        每轮最多生成的 token 数，默认 1024
  --tool-rounds N        最多调用几轮工具，默认 6；配合 --prefetch 用 0 时只读预取的资料
  --temperature T        采样温度，默认思考模式 0.6、非思考模式 0
  --top-k K / --top-p P  采样参数
  --seed S               随机种子
  --kv f32|f16|i8        KV Cache 精度，默认 Q4 模型用 f16；i8 在长上下文时读得更少
  --sparsity SPEC        decode 时丢弃的输入通道比例：单个数（各部分相同），
                         或 attn=…,mlp=…,head=…（q/k/v、gate/up 与输出层的输入）；只支持 Q4 模型
  --reply-prefix TEXT    第一轮回复由这段文字开头，模型接着写（对照实验用）
  --consult-below P      先不用工具作答，再追问最终选项（A–D）；最高概率低于 P（0–1）时
                         才按其余选项（如 --prefetch 或 --reference）查资料重新作答；只适用于单选题";

#[derive(Default)]
struct Options {
    command: String,
    store: Option<PathBuf>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    query: Option<String>,
    limit: usize,
    temperature: Option<f32>,
    config: AgentConfig,
}

#[derive(Deserialize)]
struct Question {
    id: String,
    question: String,
}

#[derive(Serialize)]
struct Record<'a> {
    id: &'a str,
    reply: &'a str,
    searches: &'a [String],
    pages: &'a [String],
    notes: &'a [u64],
    rejected: usize,
    generated: usize,
    seconds: f64,
    confidence: Option<f32>,
    consulted: bool,
    transcript: &'a str,
}

fn value<I: Iterator<Item = String>>(args: &mut I, name: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{name} 需要参数"))
}

fn number<T: std::str::FromStr>(text: String, name: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{name} 的参数无效：{text}"))
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        command: args.next().ok_or(USAGE)?,
        limit: 5,
        ..Options::default()
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--store" => options.store = Some(value(&mut args, &arg)?.into()),
            "--input" => options.input = Some(value(&mut args, &arg)?.into()),
            "--output" => options.output = Some(value(&mut args, &arg)?.into()),
            "--model" => options.model = Some(value(&mut args, &arg)?.into()),
            "--tokenizer" => options.tokenizer = Some(value(&mut args, &arg)?.into()),
            "--query" | "--question" => options.query = Some(value(&mut args, &arg)?),
            "--limit" => options.limit = number(value(&mut args, &arg)?, &arg)?,
            "--think" => options.config.thinking = true,
            "--no-tools" => options.config.tools = false,
            "--browse" => options.config.browse = true,
            "--no-learn" => options.config.learn = false,
            "--prefetch" => options.config.prefetch = true,
            "--reference" => options.config.reference = true,
            "--tool-rounds" => options.config.tool_rounds = number(value(&mut args, &arg)?, &arg)?,
            "--turn-tokens" => options.config.turn_tokens = number(value(&mut args, &arg)?, &arg)?,
            "--temperature" => options.temperature = Some(number(value(&mut args, &arg)?, &arg)?),
            "--top-k" => options.config.sampling.top_k = number(value(&mut args, &arg)?, &arg)?,
            "--top-p" => options.config.sampling.top_p = number(value(&mut args, &arg)?, &arg)?,
            "--seed" => options.config.seed = number(value(&mut args, &arg)?, &arg)?,
            "--kv" => {
                options.config.kv = Some(match value(&mut args, &arg)?.as_str() {
                    "f32" => KvPrecision::F32,
                    "f16" => KvPrecision::F16,
                    "i8" => KvPrecision::I8,
                    other => return Err(format!("--kv 只能是 f32、f16 或 i8，收到 {other}")),
                })
            }
            "--sparsity" => {
                options.config.sparsity = value(&mut args, &arg)?
                    .parse()
                    .map_err(|error| format!("--sparsity：{error}"))?;
            }
            "--reply-prefix" => options.config.reply_prefix = value(&mut args, &arg)?,
            "--consult-below" => {
                let threshold: f32 = number(value(&mut args, &arg)?, &arg)?;
                if !(threshold > 0.0 && threshold <= 1.0) {
                    return Err(format!(
                        "--consult-below 要在 0 到 1 之间（不含 0），收到 {threshold}"
                    ));
                }
                options.config.consult_below = Some(threshold);
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("未知参数 {other}\n\n{USAGE}")),
        }
    }
    if options.config.reference && options.config.prefetch {
        return Err("--reference 与 --prefetch 都是先检索一次，只能选一个".to_string());
    }
    if options.config.consult_below.is_some() && !options.config.tools && !options.config.reference
    {
        return Err(
            "--consult-below 的第二次作答要查资料，与 --no-tools 同用时需要 --reference"
                .to_string(),
        );
    }
    options.config.sampling.temperature = options
        .temperature
        .unwrap_or(if options.config.thinking { 0.6 } else { 0.0 });
    if options.config.thinking {
        if options.config.sampling.top_k == 0 {
            options.config.sampling.top_k = 20;
        }
        if options.config.sampling.top_p >= 1.0 {
            options.config.sampling.top_p = 0.95;
        }
    }
    Ok(options)
}

fn required<T: Clone>(value: &Option<T>, name: &str) -> Result<T, String> {
    value
        .clone()
        .ok_or_else(|| format!("缺少 {name}\n\n{USAGE}"))
}

fn load_model(path: &PathBuf) -> Result<Arc<Model>, String> {
    let file =
        File::open(path).map_err(|error| format!("无法打开模型 {}: {error}", path.display()))?;
    let total = file.metadata().map_err(|error| error.to_string())?.len() as usize;
    let mut loader = ModelLoader::new(total);
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; 16 << 20];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("读取模型失败: {error}"))?;
        if read == 0 {
            break;
        }
        loader
            .write(&buffer[..read])
            .map_err(|error| error.to_string())?;
    }
    loader
        .finish()
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

fn engine(options: &Options) -> Result<(Arc<Model>, Tokenizer), String> {
    let model = load_model(&required(&options.model, "--model")?)?;
    let path = required(&options.tokenizer, "--tokenizer")?;
    let tokenizer = Tokenizer::from_file(&path)
        .map_err(|error| format!("无法加载分词器 {}: {error}", path.display()))?;
    Ok((model, tokenizer))
}

#[derive(Serialize)]
struct PromptRecord<'a> {
    id: &'a str,
    prompt: String,
}

fn prompts(options: &Options) -> Result<(), String> {
    let base = if options.config.reference || (options.config.prefetch && options.config.tools) {
        Some(KnowledgeBase::open(&required(&options.store, "--store")?)?)
    } else {
        None
    };
    let render = |question: &str| {
        let prompt = match &base {
            Some(base) if options.config.reference => {
                reference_prompt(base, &options.config, question, &mut Vec::new()).0
            }
            Some(base) => prefetch_prompt(base, &options.config, question, &mut Vec::new()).0,
            None => opening_prompt(&options.config, question),
        };
        prompt + &options.config.reply_prefix
    };
    if let Some(question) = &options.query {
        print!("{}", render(question));
        return Ok(());
    }
    let input = required(&options.input, "--input")?;
    let output = required(&options.output, "--output")?;
    let reader = BufReader::new(
        File::open(&input).map_err(|error| format!("无法打开 {}: {error}", input.display()))?,
    );
    let mut writer = BufWriter::new(
        File::create(&output).map_err(|error| format!("无法写入 {}: {error}", output.display()))?,
    );
    for (number, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let question: Question = serde_json::from_str(&line)
            .map_err(|error| format!("第 {} 行格式错误: {error}", number + 1))?;
        let record = PromptRecord {
            id: &question.id,
            prompt: render(&question.question),
        };
        writeln!(
            writer,
            "{}",
            serde_json::to_string(&record).map_err(|error| error.to_string())?
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn run(options: Options) -> Result<(), String> {
    if options.command == "prompt" {
        return prompts(&options);
    }
    let store = required(&options.store, "--store")?;
    let mut base = KnowledgeBase::open(&store)?;
    match options.command.as_str() {
        "ingest" => {
            let count = base.import(&required(&options.input, "--input")?)?;
            println!(
                "已导入 {count} 段，知识库共 {} 段、{} 条笔记",
                base.passages().len(),
                base.notes().len()
            );
        }
        "search" => {
            for hit in base.search(&required(&options.query, "--query")?, options.limit) {
                println!(
                    "{:.3}\t{}\t{}\t{}",
                    hit.score,
                    hit.id,
                    hit.source,
                    hit.text.replace('\n', " ")
                );
            }
        }
        "ask" => {
            let (model, tokenizer) = engine(&options)?;
            let question = required(&options.query, "--question")?;
            let answer =
                Agent::new(model, &tokenizer, &mut base, options.config.clone()).ask(&question)?;
            println!("{}", answer.reply);
            if let Some(confidence) = answer.confidence {
                eprintln!(
                    "不查资料时对最终选项的把握 {confidence:.3}，{}",
                    if answer.consulted {
                        "已查资料重答"
                    } else {
                        "未查资料"
                    }
                );
            }
            eprintln!(
                "检索 {:?}，网页 {:?}，新笔记 {:?}，生成 {} token",
                answer.searches, answer.pages, answer.notes, answer.generated
            );
        }
        "batch" => {
            let (model, tokenizer) = engine(&options)?;
            let input = required(&options.input, "--input")?;
            let output = required(&options.output, "--output")?;
            let reader = BufReader::new(
                File::open(&input)
                    .map_err(|error| format!("无法打开 {}: {error}", input.display()))?,
            );
            let mut writer = BufWriter::new(
                File::create(&output)
                    .map_err(|error| format!("无法写入 {}: {error}", output.display()))?,
            );
            for (number, line) in reader.lines().enumerate() {
                let line = line.map_err(|error| error.to_string())?;
                if line.trim().is_empty() {
                    continue;
                }
                let question: Question = serde_json::from_str(&line)
                    .map_err(|error| format!("第 {} 行格式错误: {error}", number + 1))?;
                let started = Instant::now();
                let answer =
                    Agent::new(model.clone(), &tokenizer, &mut base, options.config.clone())
                        .ask(&question.question)?;
                let record = Record {
                    id: &question.id,
                    reply: &answer.reply,
                    searches: &answer.searches,
                    pages: &answer.pages,
                    notes: &answer.notes,
                    rejected: answer.rejected,
                    generated: answer.generated,
                    seconds: started.elapsed().as_secs_f64(),
                    confidence: answer.confidence,
                    consulted: answer.consulted,
                    transcript: &answer.transcript,
                };
                writeln!(
                    writer,
                    "{}",
                    serde_json::to_string(&record).map_err(|error| error.to_string())?
                )
                .map_err(|error| error.to_string())?;
                writer.flush().map_err(|error| error.to_string())?;
                eprintln!(
                    "{} 完成：{:.1}s，检索 {} 次，新笔记 {:?}",
                    question.id,
                    record.seconds,
                    answer.searches.len(),
                    answer.notes
                );
            }
        }
        other => return Err(format!("未知命令 {other}\n\n{USAGE}")),
    }
    Ok(())
}

fn main() {
    let result = parse(std::env::args().skip(1)).and_then(run);
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
