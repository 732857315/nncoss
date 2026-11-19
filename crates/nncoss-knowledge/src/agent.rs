use std::process::Command;
use std::sync::Arc;

use nncoss_llm::{Model, SamplingConfig, Session};
use serde::Serialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::store::{Kind, KnowledgeBase};

pub const EOS_TOKENS: [u32; 2] = [151643, 151645];
const NO_THINK: &str = "<think>\n\n</think>\n\n";
const PASSAGE_CHARS: usize = 360;
const PAGE_CHARS: usize = 4000;
const INSTRUCTION: &str = "你的知识保存在外部知识库里，知识库中有幼儿园到高中的课本、计算机与算法资料，以及你以前记下的笔记。回答需要事实、概念、公式、方法或课文内容的问题时，先用 search_knowledge 查找，再依据查到的内容作答；学到以后用得上的知识时，用 write_note 记下来。";
const BROWSE_INSTRUCTION: &str = "需要网页上的内容时，用 browse 打开网址阅读。";
const LEARN_PROMPT: &str = "请把刚才从资料中学到、以后解决同类问题时用得上的知识归纳成一条简洁的笔记。只写知识本身（概念、规律、公式、方法或事实），不要写本题的答案，也不要写推理过程。";
const SEARCH_TOOL: &str = r#"{"type": "function", "function": {"name": "search_knowledge", "description": "在知识库（课本、计算机与算法资料、你记下的笔记）中检索，返回最相关的几段内容。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "要检索的关键词或问题"}}, "required": ["query"]}}}"#;
const NOTE_TOOL: &str = r#"{"type": "function", "function": {"name": "write_note", "description": "把一条以后用得上的知识记入知识库，以后检索时可以找到。", "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "要记下的知识，写成简洁完整的陈述"}}, "required": ["content"]}}}"#;
const BROWSE_TOOL: &str = r#"{"type": "function", "function": {"name": "browse", "description": "打开一个网址，返回网页的正文文字。", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "以 http:// 或 https:// 开头的网址"}}, "required": ["url"]}}}"#;

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub thinking: bool,
    pub tools: bool,
    pub browse: bool,
    pub learn: bool,
    pub turn_tokens: usize,
    pub tool_rounds: usize,
    pub search_limit: usize,
    pub sampling: SamplingConfig,
    pub seed: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            thinking: false,
            tools: true,
            browse: false,
            learn: true,
            turn_tokens: 1024,
            tool_rounds: 6,
            search_limit: 4,
            sampling: SamplingConfig::default(),
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Answer {
    pub reply: String,
    pub transcript: String,
    pub searches: Vec<String>,
    pub pages: Vec<String>,
    pub notes: Vec<u64>,
    pub generated: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

pub fn system_prompt(config: &AgentConfig) -> String {
    if !config.tools {
        return String::new();
    }
    let mut tools = vec![SEARCH_TOOL, NOTE_TOOL];
    let mut instruction = INSTRUCTION.to_string();
    if config.browse {
        tools.push(BROWSE_TOOL);
        instruction.push_str(BROWSE_INSTRUCTION);
    }
    format!(
        "<|im_start|>system\n{instruction}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n",
        tools.join("\n")
    )
}

pub fn assistant_open(thinking: bool) -> &'static str {
    if thinking {
        "<|im_start|>assistant\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}

pub fn visible(text: &str) -> &str {
    if let Some(position) = text.rfind("</think>") {
        return text[position + "</think>".len()..].trim();
    }
    if text.trim_start().starts_with("<think>") {
        return "";
    }
    text.trim()
}

pub fn parse_tool_calls(text: &str) -> Vec<Result<ToolCall, String>> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let (body, next) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        let parsed = serde_json::from_str::<Value>(body.trim())
            .map_err(|error| format!("工具调用不是合法的 JSON：{error}"))
            .and_then(|value| {
                let name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("工具调用缺少 name")?
                    .to_string();
                let arguments = value.get("arguments").cloned().unwrap_or(Value::Null);
                Ok(ToolCall { name, arguments })
            });
        calls.push(parsed);
        rest = next;
    }
    calls
}

fn truncate(text: &str, limit: usize) -> String {
    let mut output: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        output.push('…');
    }
    output
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" => Some(' '),
        _ => {
            let number = entity.strip_prefix('#')?;
            let value = match number.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => number.parse().ok()?,
            };
            char::from_u32(value)
        }
    }
}

pub fn html_to_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut cleaned = String::with_capacity(html.len());
    let mut position = 0;
    while position < html.len() {
        let skip = ["<script", "<style", "<noscript", "<svg"]
            .iter()
            .filter_map(|tag| lower[position..].find(tag).map(|offset| (offset, *tag)))
            .min();
        let Some((offset, tag)) = skip else {
            cleaned.push_str(&html[position..]);
            break;
        };
        cleaned.push_str(&html[position..position + offset]);
        let closing = format!("</{}", &tag[1..]);
        position = match lower[position + offset..].find(&closing) {
            Some(end) => {
                let after = position + offset + end;
                lower[after..]
                    .find('>')
                    .map(|close| after + close + 1)
                    .unwrap_or(html.len())
            }
            None => html.len(),
        };
    }
    let mut text = String::with_capacity(cleaned.len());
    let mut chars = cleaned.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '<' => {
                let mut tag = String::new();
                for next in chars.by_ref() {
                    if next == '>' {
                        break;
                    }
                    tag.push(next);
                }
                let name = tag
                    .trim_start_matches('/')
                    .split(|c: char| c.is_whitespace() || c == '/')
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if matches!(
                    name.as_str(),
                    "br" | "p"
                        | "div"
                        | "li"
                        | "tr"
                        | "h1"
                        | "h2"
                        | "h3"
                        | "h4"
                        | "h5"
                        | "h6"
                        | "pre"
                        | "section"
                        | "article"
                ) {
                    text.push('\n');
                } else if matches!(name.as_str(), "td" | "th") {
                    text.push(' ');
                }
            }
            '&' => {
                let mut entity = String::new();
                while let Some(&next) = chars.peek() {
                    if next == ';' || entity.len() > 8 || next.is_whitespace() {
                        break;
                    }
                    entity.push(next);
                    chars.next();
                }
                if chars.peek() == Some(&';') {
                    chars.next();
                    match decode_entity(&entity) {
                        Some(decoded) => text.push(decoded),
                        None => {
                            text.push('&');
                            text.push_str(&entity);
                            text.push(';');
                        }
                    }
                } else {
                    text.push('&');
                    text.push_str(&entity);
                }
            }
            _ => text.push(ch),
        }
    }
    text.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn fetch_page(url: &str) -> Result<String, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("只支持 http:// 或 https:// 网址".to_string());
    }
    let output = Command::new("curl")
        .args([
            "-sSL",
            "--max-time",
            "30",
            "--max-filesize",
            "8000000",
            "-A",
            "Mozilla/5.0 (nncoss-knowledge)",
            url,
        ])
        .output()
        .map_err(|error| format!("无法启动 curl：{error}"))?;
    if !output.status.success() {
        return Err(format!(
            "打开网页失败：{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(html_to_text(&String::from_utf8_lossy(&output.stdout)))
}

pub struct Agent<'a> {
    model: Arc<Model>,
    tokenizer: &'a Tokenizer,
    base: &'a mut KnowledgeBase,
    config: AgentConfig,
}

impl<'a> Agent<'a> {
    pub fn new(
        model: Arc<Model>,
        tokenizer: &'a Tokenizer,
        base: &'a mut KnowledgeBase,
        config: AgentConfig,
    ) -> Self {
        Self {
            model,
            tokenizer,
            base,
            config,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }

    fn turn(
        &self,
        session: &mut Session,
        prompt: &str,
        limit: usize,
        answer: &mut Answer,
    ) -> Result<String, String> {
        let tokens = self.encode(prompt)?;
        let generated = session
            .generate(&tokens, limit, &EOS_TOKENS)
            .map_err(|error| error.to_string())?;
        answer.generated += generated.len();
        self.tokenizer
            .decode(&generated, false)
            .map_err(|error| error.to_string())
    }

    fn run_tool(
        &mut self,
        call: &ToolCall,
        question: &str,
        sources: &mut Vec<String>,
        answer: &mut Answer,
    ) -> String {
        let argument = |key: &str| {
            call.arguments
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string()
        };
        match call.name.as_str() {
            "search_knowledge" => {
                let query = argument("query");
                if query.is_empty() {
                    return "缺少 query 参数".to_string();
                }
                answer.searches.push(query.clone());
                let hits = self.base.search(&query, self.config.search_limit);
                if hits.is_empty() {
                    return "知识库中没有找到相关内容".to_string();
                }
                hits.iter()
                    .map(|hit| match hit.kind {
                        Kind::Note => format!("[笔记 {}] {}", hit.id, hit.text),
                        Kind::Passage => {
                            if !sources.contains(&hit.id) {
                                sources.push(hit.id.clone());
                            }
                            format!("[{}] {}", hit.source, truncate(&hit.text, PASSAGE_CHARS))
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            "write_note" => {
                let content = argument("content");
                match self
                    .base
                    .add_note(&content, sources.clone(), Some(question.to_string()))
                {
                    Ok(Some(id)) => {
                        answer.notes.push(id);
                        format!("已记下笔记 note-{id}")
                    }
                    Ok(None) => "笔记为空或已经存在，没有重复记录".to_string(),
                    Err(error) => format!("记笔记失败：{error}"),
                }
            }
            "browse" if self.config.browse => {
                let url = argument("url");
                answer.pages.push(url.clone());
                match fetch_page(&url) {
                    Ok(text) if text.is_empty() => "网页没有可读的文字".to_string(),
                    Ok(text) => truncate(&text, PAGE_CHARS),
                    Err(error) => error,
                }
            }
            other => format!("没有名为 {other} 的工具"),
        }
    }

    pub fn ask(&mut self, question: &str) -> Result<Answer, String> {
        let mut session = Session::new(self.model.clone(), 0).map_err(|error| error.to_string())?;
        session
            .set_sampler(self.config.sampling, self.config.seed)
            .map_err(|error| error.to_string())?;
        let mut answer = Answer::default();
        let mut sources = Vec::new();
        let mut prompt = format!(
            "{}<|im_start|>user\n{question}<|im_end|>\n{}",
            system_prompt(&self.config),
            assistant_open(self.config.thinking)
        );
        answer.transcript.push_str(&prompt);
        let mut rounds = 0;
        loop {
            let text = self.turn(&mut session, &prompt, self.config.turn_tokens, &mut answer)?;
            answer.transcript.push_str(&text);
            answer.transcript.push_str("<|im_end|>\n");
            let calls = if self.config.tools {
                parse_tool_calls(visible(&text))
            } else {
                Vec::new()
            };
            if calls.is_empty() || rounds >= self.config.tool_rounds {
                answer.reply = visible(&text).to_string();
                break;
            }
            rounds += 1;
            let responses: Vec<String> = calls
                .iter()
                .map(|call| match call {
                    Ok(call) => self.run_tool(call, question, &mut sources, &mut answer),
                    Err(error) => error.clone(),
                })
                .map(|output| format!("<tool_response>\n{output}\n</tool_response>"))
                .collect();
            let body = format!(
                "<|im_start|>user\n{}<|im_end|>\n{}",
                responses.join("\n"),
                assistant_open(self.config.thinking)
            );
            answer.transcript.push_str(&body);
            prompt = format!("<|im_end|>\n{body}");
        }
        if self.config.learn && !sources.is_empty() {
            let body = format!(
                "<|im_start|>user\n{LEARN_PROMPT}<|im_end|>\n<|im_start|>assistant\n{NO_THINK}"
            );
            let text = self.turn(
                &mut session,
                &format!("<|im_end|>\n{body}"),
                320,
                &mut answer,
            )?;
            answer.transcript.push_str(&body);
            answer.transcript.push_str(&text);
            let note = visible(&text);
            if parse_tool_calls(note).is_empty() {
                if let Some(id) =
                    self.base
                        .add_note(note, sources.clone(), Some(question.to_string()))?
                {
                    answer.notes.push(id);
                }
            }
        }
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_calls_are_parsed_outside_thinking_only() {
        let text = "<think>\n先查一下 <tool_call>{\"name\":\"x\"}</tool_call>\n</think>\n\n<tool_call>\n{\"name\": \"search_knowledge\", \"arguments\": {\"query\": \"光合作用\"}}\n</tool_call>";
        let calls = parse_tool_calls(visible(text));
        assert_eq!(calls.len(), 1);
        let call = calls[0].as_ref().unwrap();
        assert_eq!(call.name, "search_knowledge");
        assert_eq!(call.arguments["query"], "光合作用");
        assert!(parse_tool_calls("<tool_call>{bad json}</tool_call>")[0].is_err());
        assert!(parse_tool_calls("<tool_call>{\"arguments\": {}}</tool_call>")[0].is_err());
        assert!(parse_tool_calls("没有工具调用").is_empty());
    }

    #[test]
    fn visible_hides_unfinished_thinking() {
        assert_eq!(visible("<think>\n还在想"), "");
        assert_eq!(visible("<think>\nx\n</think>\n\n答案是 3"), "答案是 3");
        assert_eq!(visible("直接回答"), "直接回答");
    }

    #[test]
    fn prompts_follow_the_qwen_tool_template() {
        let mut config = AgentConfig::default();
        let prompt = system_prompt(&config);
        assert!(prompt.starts_with("<|im_start|>system\n你的知识保存在外部知识库里"));
        assert!(prompt.contains("\"name\": \"search_knowledge\""));
        assert!(prompt.contains("\"name\": \"write_note\""));
        assert!(!prompt.contains("browse"));
        assert!(prompt.ends_with("</tool_call><|im_end|>\n"));
        config.browse = true;
        assert!(system_prompt(&config).contains("\"name\": \"browse\""));
        config.tools = false;
        assert!(system_prompt(&config).is_empty());
        assert_eq!(
            assistant_open(false),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(assistant_open(true), "<|im_start|>assistant\n");
    }

    #[test]
    fn html_is_reduced_to_readable_text() {
        let html = "<html><head><style>p{color:red}</style><script>var a = '<p>';</script></head><body><h1>A + B &amp; C</h1><p>Input: two integers&nbsp;a, b.</p><p>Output a+b &lt; 10&#94;9 &#x4e2d;</p><div>x<br>y</div></body></html>";
        assert_eq!(
            html_to_text(html),
            "A + B & C\nInput: two integers a, b.\nOutput a+b < 10^9 中\nx\ny"
        );
        assert!(fetch_page("file:///etc/passwd").is_err());
    }
}
