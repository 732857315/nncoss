use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nncoss_llm::{Model, ModelLoader, SamplingConfig, Session};
use tokenizers::Tokenizer;

pub const ASSISTANT_PREFIX: &str = "<think>\n\n</think>\n\n";
const COOKIE_NAME: &str = "nncoss_sid";
const EOS_TOKENS: [u32; 2] = [151643, 151645];
const MAX_BODY: usize = 64 * 1024;
const MAX_HEADERS: usize = 16 * 1024;
const MAX_REQUEST_LINE: usize = 8 * 1024;
const MAX_HEADER_LINE: usize = 8 * 1024;
const MAX_PROMPT_BYTES: usize = 4096;
const CONTENT_SECURITY_POLICY: &str = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'";

const PAGE_START: &str = r#"<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nncoss 本地对话</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ctext y='13' font-size='14'%3En%3C/text%3E%3C/svg%3E">
<style>
:root {
  color-scheme: light dark;
  font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  line-height: 1.6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: Canvas;
  color: CanvasText;
}
.site-header, main {
  width: min(46rem, 100% - 2rem);
  margin: 0 auto;
}
.site-header { padding: 2rem 0 1rem; }
.site-header h1 { margin: 0 0 0.25rem; font-size: 1.6rem; }
.site-header p { margin: 0; opacity: 0.75; }
main { padding-bottom: 3rem; }
h2 { font-size: 1.1rem; }
.conversation {
  display: grid;
  gap: 0.75rem;
  margin: 0 0 1.5rem;
  padding: 0;
  list-style: none;
}
.turn { display: grid; gap: 0.5rem; }
.bubble {
  border: 1px solid color-mix(in srgb, CanvasText 18%, transparent);
  border-radius: 0.75rem;
  padding: 0.75rem 1rem;
}
.bubble.user { background: color-mix(in srgb, CanvasText 6%, Canvas); }
.bubble.assistant { background: color-mix(in srgb, CanvasText 12%, Canvas); }
.bubble h3 { margin: 0 0 0.35rem; font-size: 0.85rem; opacity: 0.7; }
.bubble p { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.empty { padding: 1rem; border: 1px dashed color-mix(in srgb, CanvasText 25%, transparent); border-radius: 0.75rem; }
.chat-form { display: grid; gap: 0.5rem; margin-bottom: 1rem; }
textarea {
  width: 100%;
  min-height: 6rem;
  padding: 0.75rem;
  border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
  border-radius: 0.5rem;
  font: inherit;
  resize: vertical;
}
button {
  justify-self: start;
  padding: 0.5rem 1.25rem;
  border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
  border-radius: 0.5rem;
  background: color-mix(in srgb, CanvasText 10%, Canvas);
  color: inherit;
  font: inherit;
  cursor: pointer;
}
button:hover { background: color-mix(in srgb, CanvasText 18%, Canvas); }
button:focus-visible, textarea:focus-visible { outline: 3px solid Highlight; outline-offset: 2px; }
.hint { margin: 0; font-size: 0.85rem; opacity: 0.7; }
.reset-form { border-top: 1px solid color-mix(in srgb, CanvasText 15%, transparent); padding-top: 1rem; }
</style>
</head>
<body>
<header class="site-header">
<h1>nncoss 本地对话</h1>
<p>在本地运行的 Qwen3 对话界面，对话只保存在当前浏览器会话中。</p>
</header>
<main>
<section aria-labelledby="conversation-heading">
<h2 id="conversation-heading">对话记录</h2>
"#;

const PAGE_END: &str = r#"
</section>
<form class="chat-form" method="post" action="/chat">
<label for="prompt">输入消息</label>
<textarea id="prompt" name="prompt" rows="4" required aria-describedby="prompt-hint" placeholder="输入后点击发送"></textarea>
<p id="prompt-hint" class="hint">空消息不会加入对话。</p>
<button type="submit">发送</button>
</form>
<form class="reset-form" method="post" action="/reset">
<button type="submit">清空对话</button>
</form>
</main>
</body>
</html>
"#;

#[derive(Debug, Clone)]
pub struct Options {
    pub host: String,
    pub port: u16,
    pub mock: bool,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub decode: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub chunk_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8765,
            mock: false,
            model: PathBuf::from("model/Qwen3-0.6B/qwen3-0.6b-f32.nncossllm"),
            tokenizer: PathBuf::from("model/Qwen3-0.6B/tokenizer.json"),
            decode: 32,
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seed: 42,
            chunk_bytes: 1_048_576,
        }
    }
}

enum Engine {
    Mock,
    Real {
        model: Arc<Model>,
        tokenizer: Arc<Tokenizer>,
        decode: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    },
}

#[derive(Default)]
struct ClientState {
    history: Vec<Turn>,
    session: Option<Session>,
}

#[derive(Clone)]
struct Turn {
    user: String,
    assistant: String,
}

struct Page {
    html: String,
    id: String,
    set_cookie: bool,
}

#[derive(Debug)]
enum ChatError {
    PromptTooLong,
    Generation(String),
}

pub struct App {
    engine: Engine,
    clients: Mutex<HashMap<String, Arc<Mutex<ClientState>>>>,
    counter: AtomicU64,
    inference: Mutex<()>,
}

impl App {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            clients: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(1),
            inference: Mutex::new(()),
        }
    }

    fn page(&self, cookie: Option<&str>) -> Page {
        let (id, set_cookie, client) = self.client(cookie);
        self.page_for(id, set_cookie, &client)
    }

    fn chat(&self, cookie: Option<&str>, prompt: Option<&str>) -> Result<Page, ChatError> {
        let (id, set_cookie, client) = self.client(cookie);
        if let Some(prompt) = prompt {
            if prompt.trim().is_empty() {
                return Ok(self.page_for(id, set_cookie, &client));
            }
            if prompt.len() > MAX_PROMPT_BYTES {
                return Err(ChatError::PromptTooLong);
            }
            if self.engine.uses_inference() {
                let _inference = lock_inference(&self.inference);
                let mut state = lock_state(&client);
                let reply = self
                    .engine
                    .reply(&mut state, prompt)
                    .map_err(ChatError::Generation)?;
                state.history.push(Turn {
                    user: prompt.to_string(),
                    assistant: reply,
                });
            } else {
                let mut state = lock_state(&client);
                let reply = self
                    .engine
                    .reply(&mut state, prompt)
                    .map_err(ChatError::Generation)?;
                state.history.push(Turn {
                    user: prompt.to_string(),
                    assistant: reply,
                });
            }
        }
        Ok(self.page_for(id, set_cookie, &client))
    }

    fn page_for(&self, id: String, set_cookie: bool, client: &Arc<Mutex<ClientState>>) -> Page {
        let history = {
            let state = lock_state(client);
            state.history.clone()
        };
        Page {
            html: render_page(&history),
            id,
            set_cookie,
        }
    }

    fn reset(&self, cookie: Option<&str>) -> (String, bool) {
        let (id, set_cookie, client) = self.client(cookie);
        {
            let mut state = lock_state(&client);
            state.history.clear();
            state.session = None;
        }
        (id, set_cookie)
    }

    fn lock_clients(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<Mutex<ClientState>>>> {
        self.clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn client(&self, cookie: Option<&str>) -> (String, bool, Arc<Mutex<ClientState>>) {
        let mut clients = self.lock_clients();
        if let Some(id) = cookie.and_then(valid_session_id) {
            let client = clients
                .entry(id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(ClientState::default())))
                .clone();
            return (id, false, client);
        }
        let id = self.next_id();
        let client = Arc::new(Mutex::new(ClientState::default()));
        clients.insert(id.clone(), Arc::clone(&client));
        (id, true, client)
    }

    fn next_id(&self) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("{nanos:016x}{count:016x}")
    }
}

fn lock_state(client: &Mutex<ClientState>) -> std::sync::MutexGuard<'_, ClientState> {
    client
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_inference(inference: &Mutex<()>) -> std::sync::MutexGuard<'_, ()> {
    inference
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Engine {
    fn uses_inference(&self) -> bool {
        matches!(self, Self::Real { .. })
    }

    fn reply(&self, state: &mut ClientState, prompt: &str) -> Result<String, String> {
        match self {
            Self::Mock => Ok(format!("MOCK_REPLY:{prompt}")),
            Self::Real {
                model,
                tokenizer,
                decode,
                temperature,
                top_k,
                top_p,
                seed,
            } => {
                let text = format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ASSISTANT_PREFIX}"
                );
                let tokens = encode_text(tokenizer, &text)?;
                if state.session.is_none() {
                    let mut session = new_session(model, *temperature, *top_k, *top_p, *seed)?;
                    replay_history(&mut session, &state.history, tokenizer)?;
                    state.session = Some(session);
                }
                let session = state.session.as_mut().expect("session created");
                let reply = match generate_turn(session, &tokens, *decode, tokenizer) {
                    Ok(reply) => reply,
                    Err(error) => {
                        state.session = None;
                        return Err(error);
                    }
                };
                let end = match encode_text(tokenizer, "<|im_end|>\n") {
                    Ok(end) => end,
                    Err(error) => {
                        state.session = None;
                        return Err(error);
                    }
                };
                let ended = state
                    .session
                    .as_mut()
                    .map(|session| session.prefill(&end).is_ok())
                    .unwrap_or(false);
                if !ended {
                    state.session = None;
                }
                Ok(reply)
            }
        }
    }
}

fn new_session(
    model: &Arc<Model>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> Result<Session, String> {
    let mut session = Session::new(model.clone(), 0).map_err(|error| error.to_string())?;
    session
        .set_sampler(
            SamplingConfig {
                temperature,
                top_k,
                top_p,
            },
            seed,
        )
        .map_err(|error| error.to_string())?;
    Ok(session)
}

fn generate_turn(
    session: &mut Session,
    tokens: &[u32],
    decode: usize,
    tokenizer: &Tokenizer,
) -> Result<String, String> {
    let generated = session
        .generate(tokens, decode, &EOS_TOKENS)
        .map_err(|error| error.to_string())?;
    tokenizer
        .decode(&generated, true)
        .map_err(|error| error.to_string())
}

fn replay_history(
    session: &mut Session,
    history: &[Turn],
    tokenizer: &Tokenizer,
) -> Result<(), String> {
    let text = history_text(history);
    if text.is_empty() {
        return Ok(());
    }
    let tokens = encode_text(tokenizer, &text)?;
    session.prefill(&tokens).map_err(|error| error.to_string())
}

fn history_text(history: &[Turn]) -> String {
    let mut text = String::new();
    for turn in history {
        text.push_str("<|im_start|>user\n");
        text.push_str(&turn.user);
        text.push_str("<|im_end|>\n<|im_start|>assistant\n");
        text.push_str(ASSISTANT_PREFIX);
        text.push_str(&turn.assistant);
        text.push_str("<|im_end|>\n");
    }
    text
}

fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| error.to_string())
}

fn escape_html(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for character in input.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn render_page(history: &[Turn]) -> String {
    let mut html = String::from(PAGE_START);
    if history.is_empty() {
        html.push_str("<p class=\"empty\">还没有消息。</p>");
    } else {
        html.push_str("<ol class=\"conversation\">");
        for turn in history {
            html.push_str("<li class=\"turn\"><div class=\"bubble user\"><h3>你</h3><p>");
            html.push_str(&escape_html(&turn.user));
            html.push_str("</p></div><div class=\"bubble assistant\"><h3>助手</h3><p>");
            html.push_str(&escape_html(&turn.assistant));
            html.push_str("</p></div></li>");
        }
        html.push_str("</ol>");
    }
    html.push_str(PAGE_END);
    html
}

fn valid_session_id(value: &str) -> Option<String> {
    let valid = !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_');
    valid.then(|| value.to_string())
}

fn set_cookie_header(id: &str) -> String {
    format!("{COOKIE_NAME}={id}; Path=/; HttpOnly; SameSite=Lax")
}

fn request_cookie(headers: &[(String, String)]) -> Option<String> {
    headers
        .iter()
        .filter(|(name, _)| name == "cookie")
        .flat_map(|(_, value)| value.split(';'))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(name, value)| {
            if name.trim() == COOKIE_NAME {
                valid_session_id(value.trim())
            } else {
                None
            }
        })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn percent_decode(input: &[u8]) -> Result<Vec<u8>, ()> {
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        match input[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' => {
                if index + 2 >= input.len() {
                    return Err(());
                }
                let high = hex_value(input[index + 1]).ok_or(())?;
                let low = hex_value(input[index + 2]).ok_or(())?;
                output.push((high << 4) | low);
                index += 3;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    Ok(output)
}

fn form_value(body: &[u8], key: &str) -> Result<Option<String>, ()> {
    for pair in body.split(|byte| *byte == b'&') {
        if pair.is_empty() {
            continue;
        }
        let mut parts = pair.splitn(2, |byte| *byte == b'=');
        let raw_key = parts.next().unwrap_or(&[]);
        let raw_value = parts.next().unwrap_or(&[]);
        if percent_decode(raw_key)? == key.as_bytes() {
            let value = percent_decode(raw_value)?;
            return String::from_utf8(value).map(Some).map_err(|_| ());
        }
    }
    Ok(None)
}

struct Request {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    fn new(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        Self {
            status,
            headers: vec![
                ("Content-Type".to_string(), content_type.to_string()),
                ("X-Content-Type-Options".to_string(), "nosniff".to_string()),
                ("Referrer-Policy".to_string(), "no-referrer".to_string()),
                (
                    "Content-Security-Policy".to_string(),
                    CONTENT_SECURITY_POLICY.to_string(),
                ),
                ("X-Frame-Options".to_string(), "DENY".to_string()),
            ],
            body,
        }
    }

    fn text(status: u16, body: &str) -> Self {
        Self::new(
            status,
            "text/plain; charset=utf-8",
            body.as_bytes().to_vec(),
        )
    }

    fn html(status: u16, body: String, cookie: Option<String>) -> Self {
        let mut response = Self::new(status, "text/html; charset=utf-8", body.into_bytes());
        response
            .headers
            .push(("Cache-Control".to_string(), "no-store".to_string()));
        if let Some(cookie) = cookie {
            response.headers.push(("Set-Cookie".to_string(), cookie));
        }
        response
    }

    fn redirect(location: &str, cookie: Option<String>) -> Self {
        let mut response = Self::new(303, "text/plain; charset=utf-8", Vec::new());
        response
            .headers
            .push(("Location".to_string(), location.to_string()));
        if let Some(cookie) = cookie {
            response.headers.push(("Set-Cookie".to_string(), cookie));
        }
        response
    }

    fn with_allow(mut self, allow: &str) -> Self {
        self.headers.push(("Allow".to_string(), allow.to_string()));
        self
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        303 => "See Other",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    }
}

fn write_response(stream: &mut TcpStream, response: Response) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\n",
        response.status,
        reason(response.status)
    );
    for (name, value) in &response.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str(&format!(
        "Content-Length: {}\r\nConnection: close\r\n\r\n",
        response.body.len()
    ));
    stream.write_all(head.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

fn read_line_bounded<R: BufRead>(
    reader: &mut R,
    limit: usize,
) -> Result<Option<Vec<u8>>, (u16, &'static str)> {
    let mut buffer = Vec::new();
    let read = reader
        .by_ref()
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut buffer)
        .map_err(|_| (400, "请求读取失败"))?;
    if read == 0 {
        return Ok(None);
    }
    if buffer.len() > limit {
        return Err((431, "请求行或请求头过长"));
    }
    Ok(Some(buffer))
}

fn read_request(stream: &mut TcpStream) -> Result<Request, (u16, &'static str)> {
    let mut reader = BufReader::new(stream);
    let request_line =
        read_line_bounded(&mut reader, MAX_REQUEST_LINE)?.ok_or((400, "请求为空"))?;
    let request_line = std::str::from_utf8(&request_line).map_err(|_| (400, "请求行编码无效"))?;
    let mut header_bytes = request_line.len();
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or((400, "请求行无效"))?
        .to_ascii_uppercase();
    let target = parts.next().ok_or((400, "请求行无效"))?.to_string();
    let version = parts.next().ok_or((400, "请求行无效"))?.to_string();
    if !version.starts_with("HTTP/1.") {
        return Err((400, "HTTP 版本无效"));
    }

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    loop {
        let header =
            read_line_bounded(&mut reader, MAX_HEADER_LINE)?.ok_or((400, "请求头不完整"))?;
        header_bytes += header.len();
        if header_bytes > MAX_HEADERS {
            return Err((431, "请求头超过 16 KiB"));
        }
        let header = std::str::from_utf8(&header).map_err(|_| (400, "请求头编码无效"))?;
        let header = header.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        let (name, value) = header.split_once(':').ok_or((400, "请求头无效"))?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
        if name == "content-length" {
            content_length = value
                .parse::<usize>()
                .map_err(|_| (400, "Content-Length 无效"))?;
        }
        headers.push((name, value));
    }

    if content_length > MAX_BODY {
        return Err((413, "请求体超过 64 KiB"));
    }
    let mut body = vec![0u8; content_length];
    reader
        .read_exact(&mut body)
        .map_err(|_| (400, "请求体读取失败"))?;
    Ok(Request {
        method,
        target,
        headers,
        body,
    })
}

fn route(app: &App, request: &Request) -> Response {
    let path = request.target.split('?').next().unwrap_or("/");
    let cookie = request_cookie(&request.headers);
    match request.method.as_str() {
        "GET" => match path {
            "/healthz" => Response::text(200, "ok"),
            "/favicon.ico" => Response::text(204, ""),
            "/" => {
                let page = app.page(cookie.as_deref());
                let set_cookie = page.set_cookie.then(|| set_cookie_header(&page.id));
                Response::html(200, page.html, set_cookie)
            }
            "/chat" | "/reset" => Response::text(405, "method not allowed").with_allow("POST"),
            _ => Response::text(404, "not found"),
        },
        "POST" => match path {
            "/chat" => {
                let prompt = match form_value(&request.body, "prompt") {
                    Ok(prompt) => prompt,
                    Err(_) => return Response::text(400, "invalid form"),
                };
                match app.chat(cookie.as_deref(), prompt.as_deref()) {
                    Ok(page) => {
                        let set_cookie = page.set_cookie.then(|| set_cookie_header(&page.id));
                        Response::html(200, page.html, set_cookie)
                    }
                    Err(ChatError::PromptTooLong) => Response::text(413, "提示词超过 4096 字节"),
                    Err(ChatError::Generation(error)) => {
                        Response::text(500, &format!("生成失败: {error}"))
                    }
                }
            }
            "/reset" => {
                let (id, set_cookie) = app.reset(cookie.as_deref());
                let cookie = set_cookie.then(|| set_cookie_header(&id));
                Response::redirect("/", cookie)
            }
            "/healthz" => Response::text(405, "method not allowed").with_allow("GET"),
            _ => Response::text(404, "not found"),
        },
        _ => Response::text(405, "method not allowed"),
    }
}

fn handle_connection(mut stream: TcpStream, app: &App) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let response = match read_request(&mut stream) {
        Ok(request) => route(app, &request),
        Err((status, message)) => Response::text(status, message),
    };
    let _ = write_response(&mut stream, response);
}

fn load_engine(options: &Options) -> Result<Engine, String> {
    if options.mock {
        return Ok(Engine::Mock);
    }
    if options.chunk_bytes == 0 {
        return Err("--chunk-bytes 必须大于 0".to_string());
    }
    let file = File::open(&options.model)
        .map_err(|error| format!("无法打开模型 {}: {error}", options.model.display()))?;
    let total = file
        .metadata()
        .map_err(|error| format!("无法读取模型信息: {error}"))?
        .len() as usize;
    let mut loader = ModelLoader::new(total);
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; options.chunk_bytes];
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
    let model = loader.finish().map_err(|error| error.to_string())?;
    let tokenizer = Tokenizer::from_file(&options.tokenizer)
        .map_err(|error| format!("无法加载分词器 {}: {error}", options.tokenizer.display()))?;
    Ok(Engine::Real {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
        decode: options.decode,
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        seed: options.seed,
    })
}

pub fn serve(options: Options) -> Result<(), String> {
    let engine = load_engine(&options)?;
    let app = Arc::new(App::new(engine));
    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .map_err(|error| format!("无法绑定 {}:{}: {error}", options.host, options.port))?;
    eprintln!(
        "nncoss-web 正在监听 http://{}:{}",
        options.host, options.port
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let app = Arc::clone(&app);
        std::thread::spawn(move || handle_connection(stream, &app));
    }
    Ok(())
}

pub fn usage() -> &'static str {
    "用法: nncoss-web [选项]\n\
     \n\
     --host <地址>          监听地址，默认 127.0.0.1\n\
     --port <端口>          监听端口，默认 8765\n\
     --mock                 使用确定性模拟回复，不加载模型\n\
     --model <路径>         模型文件，默认 model/Qwen3-0.6B/qwen3-0.6b-f32.nncossllm\n\
     --tokenizer <路径>     分词器文件，默认 model/Qwen3-0.6B/tokenizer.json\n\
     --decode <数量>        每次回复最多生成的 token 数，默认 32\n\
     --temperature <值>     采样温度，默认 0.6\n\
     --top-k <数量>         采样 top-k，默认 20\n\
     --top-p <值>           采样 top-p，默认 0.95\n\
     --seed <值>            随机种子，默认 42\n\
     --chunk-bytes <字节>   模型流式读取块大小，默认 1048576\n\
     -h, --help             显示帮助"
}

pub fn run_from_args<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let options = parse_args(args)?;
    serve(options)
}

fn parse_args<I>(args: I) -> Result<Options, String>
where
    I: IntoIterator<Item = String>,
{
    let mut options = Options::default();
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--mock" => options.mock = true,
            "--host" => options.host = next_value(&mut args, "--host")?,
            "--port" => {
                options.port = next_value(&mut args, "--port")?
                    .parse()
                    .map_err(|_| "--port 必须是有效端口".to_string())?;
            }
            "--model" => options.model = PathBuf::from(next_value(&mut args, "--model")?),
            "--tokenizer" => {
                options.tokenizer = PathBuf::from(next_value(&mut args, "--tokenizer")?)
            }
            "--decode" => {
                options.decode = next_value(&mut args, "--decode")?
                    .parse()
                    .map_err(|_| "--decode 必须是非负整数".to_string())?;
            }
            "--temperature" => {
                options.temperature = next_value(&mut args, "--temperature")?
                    .parse()
                    .map_err(|_| "--temperature 必须是数字".to_string())?;
            }
            "--top-k" => {
                options.top_k = next_value(&mut args, "--top-k")?
                    .parse()
                    .map_err(|_| "--top-k 必须是非负整数".to_string())?;
            }
            "--top-p" => {
                options.top_p = next_value(&mut args, "--top-p")?
                    .parse()
                    .map_err(|_| "--top-p 必须是数字".to_string())?;
            }
            "--seed" => {
                options.seed = next_value(&mut args, "--seed")?
                    .parse()
                    .map_err(|_| "--seed 必须是非负整数".to_string())?;
            }
            "--chunk-bytes" => {
                options.chunk_bytes = next_value(&mut args, "--chunk-bytes")?
                    .parse()
                    .map_err(|_| "--chunk-bytes 必须是正整数".to_string())?;
            }
            other => return Err(format!("未知参数: {other}")),
        }
    }
    if options.chunk_bytes == 0 {
        return Err("--chunk-bytes 必须大于 0".to_string());
    }
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err("--temperature 必须为非负有限值".to_string());
    }
    if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
        return Err("--top-p 必须在 (0, 1] 内".to_string());
    }
    Ok(options)
}

fn next_value<I>(args: &mut I, name: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next().ok_or_else(|| format!("{name} 缺少参数值"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_decodes_percent_and_plus_sequences() {
        let body = b"prompt=%E4%BD%A0%E5%A5%BD%2Bworld&extra=1";
        assert_eq!(
            form_value(body, "prompt").unwrap().as_deref(),
            Some("你好+world")
        );
        assert_eq!(
            form_value(b"prompt=hello+world", "prompt")
                .unwrap()
                .as_deref(),
            Some("hello world")
        );
        assert_eq!(form_value(b"other=1", "prompt").unwrap(), None);
        assert!(form_value(b"prompt=%zz", "prompt").is_err());
        assert!(form_value(b"prompt=%FF", "prompt").is_err());
    }

    #[test]
    fn html_escaping_covers_user_and_model_text() {
        assert_eq!(
            escape_html("<script>&\"'</script>"),
            "&lt;script&gt;&amp;&quot;&#39;&lt;/script&gt;"
        );
        let app = App::new(Engine::Mock);
        let page = app.chat(None, Some("<script>alert(1)</script>")).unwrap();
        assert!(!page.html.contains("<script>alert(1)</script>"));
        assert!(page.html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(page.html.contains("MOCK_REPLY:&lt;script&gt;"));
    }

    #[test]
    fn cookies_are_web_safe_and_parse_from_headers() {
        assert_eq!(
            set_cookie_header("abc123"),
            "nncoss_sid=abc123; Path=/; HttpOnly; SameSite=Lax"
        );
        assert_eq!(valid_session_id("abc123").as_deref(), Some("abc123"));
        assert!(valid_session_id("bad;id").is_none());
        assert!(valid_session_id("").is_none());
        let headers = vec![(
            "cookie".to_string(),
            "theme=dark; nncoss_sid=deadbeef; other=1".to_string(),
        )];
        assert_eq!(request_cookie(&headers).as_deref(), Some("deadbeef"));
        let invalid = vec![("cookie".to_string(), "nncoss_sid=bad%id".to_string())];
        assert_eq!(request_cookie(&invalid), None);
    }

    #[test]
    fn clients_reuse_cookie_and_keep_isolated_mock_history() {
        let app = App::new(Engine::Mock);
        let first = app.chat(None, Some("hello")).unwrap();
        assert!(first.set_cookie);
        assert!(first.html.contains("hello"));
        assert!(first.html.contains("MOCK_REPLY:hello"));

        let same = app.chat(Some(&first.id), Some("second-turn")).unwrap();
        assert!(!same.set_cookie);
        assert_eq!(same.id, first.id);
        assert!(same.html.contains("hello"));
        assert!(same.html.contains("MOCK_REPLY:hello"));
        assert!(same.html.contains("second-turn"));
        assert!(same.html.contains("MOCK_REPLY:second-turn"));

        let other = app.page(None);
        assert!(other.set_cookie);
        assert_ne!(other.id, first.id);
        assert!(!other.html.contains("hello"));

        let other_chat = app.chat(Some(&other.id), Some("CLIENT_B")).unwrap();
        assert!(other_chat.html.contains("CLIENT_B"));
        assert!(!other_chat.html.contains("hello"));
    }

    #[test]
    fn empty_prompt_does_not_change_history_and_reset_clears_it() {
        let app = App::new(Engine::Mock);
        let initial = app.page(None);
        let empty = app.chat(Some(&initial.id), Some("")).unwrap();
        assert_eq!(empty.html, initial.html);
        let whitespace = app.chat(Some(&initial.id), Some("   \n ")).unwrap();
        assert_eq!(whitespace.html, initial.html);

        app.chat(Some(&initial.id), Some("kept")).unwrap();
        let (id, set_cookie) = app.reset(Some(&initial.id));
        assert_eq!(id, initial.id);
        assert!(!set_cookie);
        let cleared = app.page(Some(&initial.id));
        assert!(!cleared.html.contains("kept"));
        assert!(cleared.html.contains("还没有消息。"));
    }

    #[test]
    fn oversized_prompts_are_rejected_without_changing_history() {
        let app = App::new(Engine::Mock);
        let initial = app.page(None);
        let too_long = "x".repeat(MAX_PROMPT_BYTES + 1);
        assert!(matches!(
            app.chat(Some(&initial.id), Some(&too_long)),
            Err(ChatError::PromptTooLong)
        ));
        assert_eq!(app.page(Some(&initial.id)).html, initial.html);

        let at_limit = "x".repeat(MAX_PROMPT_BYTES);
        assert!(app.chat(Some(&initial.id), Some(&at_limit)).is_ok());

        let request = Request {
            method: "POST".to_string(),
            target: "/chat".to_string(),
            headers: Vec::new(),
            body: format!("prompt={}", "x".repeat(MAX_PROMPT_BYTES + 1)).into_bytes(),
        };
        assert_eq!(route(&app, &request).status, 413);
    }

    #[test]
    fn responses_include_security_headers_and_favicon() {
        assert!(PAGE_START.contains("rel=\"icon\""));
        assert!(PAGE_START.contains("data:image/svg+xml"));
        let response = Response::text(200, "ok");
        let header = |name: &str| {
            response
                .headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(header("X-Content-Type-Options"), Some("nosniff"));
        assert_eq!(header("Referrer-Policy"), Some("no-referrer"));
        assert_eq!(header("X-Frame-Options"), Some("DENY"));
        assert!(
            header("Content-Security-Policy")
                .unwrap_or_default()
                .contains("frame-ancestors 'none'")
        );
        let app = App::new(Engine::Mock);
        let request = Request {
            method: "GET".to_string(),
            target: "/favicon.ico".to_string(),
            headers: Vec::new(),
            body: Vec::new(),
        };
        assert_eq!(route(&app, &request).status, 204);
    }

    #[test]
    fn request_lines_are_bounded_before_allocation() {
        let mut short = std::io::Cursor::new(b"GET / HTTP/1.1\r\n".to_vec());
        assert_eq!(
            read_line_bounded(&mut short, MAX_REQUEST_LINE)
                .unwrap()
                .unwrap(),
            b"GET / HTTP/1.1\r\n".to_vec()
        );
        let mut long = std::io::Cursor::new(vec![b'a'; MAX_REQUEST_LINE + 1]);
        assert!(read_line_bounded(&mut long, MAX_REQUEST_LINE).is_err());
    }

    #[test]
    fn page_loads_do_not_wait_for_the_inference_lock() {
        let app = Arc::new(App::new(Engine::Mock));
        let guard = lock_inference(&app.inference);
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = Arc::clone(&app);
        std::thread::spawn(move || {
            let _ = worker.page(None);
            let _ = sender.send(());
        });
        assert!(receiver.recv_timeout(Duration::from_secs(2)).is_ok());
        drop(guard);
    }

    #[test]
    fn replayed_history_uses_the_non_thinking_assistant_prefix() {
        let history = vec![Turn {
            user: "问".to_string(),
            assistant: "答".to_string(),
        }];
        assert_eq!(
            history_text(&history),
            "<|im_start|>user\n问<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n答<|im_end|>\n"
        );
    }
}
