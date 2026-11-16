from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import (
    HTTPCookieProcessor,
    ProxyHandler,
    Request,
    build_opener,
)


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "nncoss-web"
REQUEST_TIMEOUT = 5
STARTUP_TIMEOUT = 30
FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"


@dataclass(frozen=True)
class Response:
    status: int
    url: str
    text: str


class ChatFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "form":
            return
        values = {name.lower(): value or "" for name, value in attrs}
        if values.get("method", "").lower() == "post" and values.get("action") == "/chat":
            self.found = True


class Client:
    def __init__(self, base_url):
        self.base_url = base_url
        self.cookies = CookieJar()
        self.opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(self.cookies))

    def request(self, method, path, data=None):
        url = self.base_url + path
        headers = {"Content-Type": FORM_CONTENT_TYPE} if data is not None else {}
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=REQUEST_TIMEOUT) as response:
                body = response.read()
                return Response(response.status, response.geturl(), body.decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise AssertionError(
                f"{method} {url} returned HTTP {exc.code}; response body:\n{excerpt(body)}"
            ) from exc
        except (URLError, OSError) as exc:
            raise AssertionError(f"{method} {url} failed: {exc}") from exc
        except UnicodeError as exc:
            raise AssertionError(f"{method} {url} returned non-UTF-8 content: {exc}") from exc

    def get(self, path="/"):
        return self.request("GET", path)

    def post_prompt(self, prompt):
        data = urlencode({"prompt": prompt}).encode("utf-8")
        return self.request("POST", "/chat", data)

    def reset(self):
        return self.request("POST", "/reset", b"")


def excerpt(text, limit=4000):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated {len(text) - limit} characters ..."


def expect(condition, message, response=None):
    if condition:
        return
    if response is not None:
        message += (
            f"\nHTTP status: {response.status}\nFinal URL: {response.url}"
            f"\nResponse body:\n{excerpt(response.text)}"
        )
    raise AssertionError(message)


def expect_ok(response, operation):
    expect(response.status == 200, f"{operation} did not return HTTP 200", response)


def expect_content(response, *values):
    text = unescape(response.text)
    for value in values:
        expect(value in text, f"Conversation is missing {value!r}", response)


def expect_absent(response, *values):
    text = unescape(response.text)
    for value in values:
        expect(value not in text, f"Conversation unexpectedly contains {value!r}", response)


def build_server():
    command = ["cargo", "build", "-p", "nncoss-web", "--release"]
    print("Building: " + " ".join(command), flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"Could not execute cargo: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"cargo build failed with exit code {result.returncode}"
            f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    expect(BINARY.is_file(), f"Build succeeded but server binary is missing: {BINARY}")


def free_localhost_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_health(process, base_url):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_result = "no response received"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"nncoss-web exited during startup with code {process.returncode}")
        try:
            with opener.open(base_url + "/healthz", timeout=1) as response:
                body = response.read()
                if response.status == 200 and body == b"ok":
                    return
                last_result = f"HTTP {response.status}, body {body!r}"
        except HTTPError as exc:
            last_result = f"HTTP {exc.code}, body {exc.read()!r}"
        except (URLError, OSError) as exc:
            last_result = str(exc)
        time.sleep(0.1)
    raise TimeoutError(
        f"GET /healthz did not return HTTP 200 with body 'ok' within "
        f"{STARTUP_TIMEOUT} seconds; last result: {last_result}"
    )


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return process.returncode


def run_acceptance(base_url):
    client = Client(base_url)
    initial = client.get("/")
    expect_ok(initial, "GET /")
    parser = ChatFormParser()
    parser.feed(initial.text)
    expect(parser.found, "GET / does not contain a form with method=post and action=/chat", initial)
    print("PASS: GET / contains a POST form to /chat.")

    first = client.post_prompt("hello")
    expect_ok(first, "POST /chat prompt=hello")
    expect_content(first, "hello", "MOCK_REPLY:hello")

    second = client.post_prompt("second-turn")
    expect_ok(second, "second POST /chat")
    expect_content(second, "hello", "MOCK_REPLY:hello", "second-turn", "MOCK_REPLY:second-turn")
    print("PASS: two prompts and both deterministic mock replies persist.")

    unicode_prompt = "你好，世界 🌍"
    unicode_response = client.post_prompt(unicode_prompt)
    expect_ok(unicode_response, "Unicode POST /chat")
    expect_content(unicode_response, unicode_prompt, "MOCK_REPLY:" + unicode_prompt)
    print("PASS: Unicode prompt and mock reply round-trip.")

    xss = "<script>alert(1)</script>"
    xss_response = client.post_prompt(xss)
    expect_ok(xss_response, "XSS POST /chat")
    expect(xss not in xss_response.text, "XSS string appeared raw in the HTML", xss_response)
    expect_content(xss_response, xss, "MOCK_REPLY:" + xss)
    print("PASS: XSS strings are escaped and never appear raw.")

    before_empty = client.get("/")
    empty_response = client.post_prompt("")
    expect_ok(empty_response, "empty POST /chat")
    after_empty = client.get("/")
    expect_ok(after_empty, "GET / after empty prompt")
    expect(
        after_empty.text == before_empty.text,
        "Empty prompt changed the conversation",
        after_empty,
    )
    print("PASS: empty prompt does not create history.")

    reset = client.reset()
    expect_ok(reset, "POST /reset")
    expect(reset.url == base_url + "/", "POST /reset did not finish at / after redirects", reset)
    expect_absent(
        reset,
        "hello",
        "MOCK_REPLY:hello",
        "second-turn",
        "MOCK_REPLY:second-turn",
        unicode_prompt,
        "MOCK_REPLY:" + unicode_prompt,
        xss,
        "MOCK_REPLY:" + xss,
    )
    print("PASS: POST /reset follows the redirect to / and clears history.")

    client_a = Client(base_url)
    client_b = Client(base_url)
    prompt_a = "CLIENT_A_9f47"
    prompt_b = "CLIENT_B_c218"
    response_a = client_a.post_prompt(prompt_a)
    expect_ok(response_a, "client A POST /chat")
    expect_content(response_a, prompt_a, "MOCK_REPLY:" + prompt_a)
    isolated_b = client_b.get("/")
    expect_ok(isolated_b, "client B GET /")
    expect_absent(isolated_b, prompt_a, "MOCK_REPLY:" + prompt_a)
    response_b = client_b.post_prompt(prompt_b)
    expect_ok(response_b, "client B POST /chat")
    expect_content(response_b, prompt_b, "MOCK_REPLY:" + prompt_b)
    expect_absent(response_b, prompt_a, "MOCK_REPLY:" + prompt_a)
    isolated_a = client_a.get("/")
    expect_ok(isolated_a, "client A GET /")
    expect_content(isolated_a, prompt_a, "MOCK_REPLY:" + prompt_a)
    expect_absent(isolated_a, prompt_b, "MOCK_REPLY:" + prompt_b)
    print("PASS: independent CookieJar clients have isolated conversations.")


def print_server_output(log):
    log.flush()
    log.seek(0)
    output = log.read().decode("utf-8", errors="replace")
    print("--- nncoss-web output ---", file=sys.stderr)
    print(excerpt(output) if output else "<no output>", file=sys.stderr)


def main():
    if not (ROOT / "Cargo.toml").is_file():
        print(f"Cannot locate project root; missing {ROOT / 'Cargo.toml'}", file=sys.stderr)
        return 1
    try:
        build_server()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    port = free_localhost_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(BINARY),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mock",
        "--decode",
        "4",
    ]
    print(f"Starting: {' '.join(command)}", flush=True)
    failed = False
    process = None
    with tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            wait_for_health(process, base_url)
            print("PASS: GET /healthz returned HTTP 200 with body 'ok'.")
            run_acceptance(base_url)
        except Exception as exc:
            failed = True
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        finally:
            if process is not None:
                exit_code = stop_server(process)
                if not failed:
                    print(f"Server terminated (exit code {exit_code}).")
            if failed:
                print_server_output(log)
    print("Web UI acceptance passed: all checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
