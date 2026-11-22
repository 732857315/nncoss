use std::fs;
use std::io::Write;
use std::path::PathBuf;

use nncoss_edit::{Edit, ModelFile};
use nncoss_llm::Model;

const USAGE: &str = "用法：
  nncoss-edit inspect <模型>
  nncoss-edit apply --input <模型> --output <新模型> (--plan <改动.json> | --edit <JSON>)... [--no-verify]

改动计划是 JSON 数组，每项形如：
  {\"op\": \"scale\", \"target\": \"mlp\", \"layer\": 1, \"index\": [2997], \"factor\": 0.0}
target 可选 mlp、kv_head、q_norm、k_norm、attn_norm、ffn_norm、final_norm、token；
index 为空表示该张量的全部行或元素。
删除整层：
  {\"op\": \"drop_layers\", \"layers\": [20, 21]}
删除后层号从 0 重新连续编号，计划中排在它后面的改动使用新层号。
删除子层（输出投影置零，nncoss-llm 会整块跳过）：
  {\"op\": \"drop_sublayer\", \"part\": \"mlp\", \"layers\": [12, 13]}
part 可选 mlp、attention。";

fn inspect(path: &PathBuf) -> Result<(), String> {
    let file = ModelFile::parse(
        fs::read(path).map_err(|error| format!("无法读取 {}: {error}", path.display()))?,
    )?;
    let config = file.config();
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "vocab={} hidden={} intermediate={} layers={} heads={}/{} head_dim={} q4={} tied={}",
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.layer_count,
        config.head_count,
        config.kv_head_count,
        config.head_dim,
        config.quantized,
        config.tied_embeddings
    );
    for tensor in file.tensors() {
        let layer = tensor
            .layer
            .map(|layer| layer.to_string())
            .unwrap_or_else(|| "-".to_string());
        let line = writeln!(
            out,
            "{layer:>3} {:<26} offset={:<12} {}x{}{}",
            tensor.kind,
            tensor.offset,
            tensor.rows,
            tensor.columns,
            if tensor.quantized { " q4" } else { "" }
        );
        if line.is_err() {
            break;
        }
    }
    Ok(())
}

fn apply(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let (mut input, mut output, mut edits, mut verify) = (None, None, Vec::new(), true);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} 需要参数"));
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--plan" => {
                let path = value()?;
                let text = fs::read_to_string(&path)
                    .map_err(|error| format!("无法读取 {path}: {error}"))?;
                let plan: Vec<Edit> = serde_json::from_str(&text)
                    .map_err(|error| format!("{path} 格式错误: {error}"))?;
                edits.extend(plan);
            }
            "--edit" => edits.push(
                serde_json::from_str(&value()?)
                    .map_err(|error| format!("--edit 格式错误: {error}"))?,
            ),
            "--no-verify" => verify = false,
            other => return Err(format!("未知参数 {other}\n\n{USAGE}")),
        }
    }
    let input = input.ok_or(USAGE)?;
    let output = output.ok_or(USAGE)?;
    if edits.is_empty() {
        return Err(format!("没有指定任何改动\n\n{USAGE}"));
    }
    let mut file = ModelFile::parse(
        fs::read(&input).map_err(|error| format!("无法读取 {}: {error}", input.display()))?,
    )?;
    let mut total = 0;
    for (number, edit) in edits.iter().enumerate() {
        let changed = file
            .apply(edit)
            .map_err(|error| format!("第 {} 项改动失败：{error}", number + 1))?;
        total += changed;
    }
    if verify {
        Model::from_bytes(file.bytes())
            .map_err(|error| format!("改动后的模型无法加载：{error}"))?;
    }
    let temporary = output.with_extension("tmp");
    fs::write(&temporary, file.bytes())
        .map_err(|error| format!("无法写入 {}: {error}", temporary.display()))?;
    fs::rename(&temporary, &output)
        .map_err(|error| format!("无法写入 {}: {error}", output.display()))?;
    println!(
        "已应用 {} 项改动，修改 {total} 个数值，写入 {}",
        edits.len(),
        output.display()
    );
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let result = match args.next().as_deref() {
        Some("inspect") => args
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| USAGE.to_string())
            .and_then(|path| inspect(&path)),
        Some("apply") => apply(args),
        _ => Err(USAGE.to_string()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
