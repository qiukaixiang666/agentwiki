from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://127.0.0.1:18082"


def _post_json(api_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(api_url: str, path: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(api_url.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _resolve_memory_root(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _health(api_url: str, timeout: float) -> dict[str, Any] | None:
    try:
        return _get_json(api_url, "/health", timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _validate_memory_root(api_url: str, requested_root: Path | None, timeout: float) -> None:
    if requested_root is None:
        return
    payload = _health(api_url, timeout)
    if not payload:
        return
    running_root = (
        payload.get("details", {}).get("memory_root")
        if isinstance(payload.get("details"), dict)
        else None
    )
    if running_root and Path(running_root).resolve() != requested_root:
        print(
            "当前端口上的 Memory API 使用的是另一个记忆库：\n"
            f"  running: {Path(running_root).resolve()}\n"
            f"  wanted : {requested_root}\n"
            "请先运行 `bash scripts/stop_services.sh` 后重试，或换一个 MEMORY_API_PORT。",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _print_help() -> None:
    print(
        """
可用命令：
  /help              显示帮助
  /exit              退出
  /new [session]     开启新会话；不填则自动生成
  /session <id>      切换 session_id
  /topk <n>          修改每轮召回记忆数量
  /debug on|off      是否显示召回记忆和记忆更新
  /memories [n]      查看记忆库中最近 n 条记忆，默认 10
  /card              查看 User Card
""".strip()
    )


def _show_memories(api_url: str, limit: int, timeout: float) -> None:
    payload = _get_json(api_url, f"/memories?limit={limit}", timeout)
    memories = payload.get("memories", [])
    if not memories:
        print("当前没有记忆。")
        return
    for index, memory in enumerate(memories, start=1):
        print(f"{index}. [{memory.get('memory_type', 'fact')}] {memory.get('content', '')}")
        print(
            "   "
            f"id={memory.get('id', '')} "
            f"topic={memory.get('topic', '')} "
            f"confidence={memory.get('confidence', '')} "
            f"observed_at={memory.get('observed_at') or '-'}"
        )


def _show_user_card(api_url: str, timeout: float) -> None:
    payload = _get_json(api_url, "/user-card", timeout)
    text = payload.get("profile_text", "")
    print(text if text else "User Card 还没有内容。")


def _print_debug(payload: dict[str, Any]) -> None:
    recalled = payload.get("recalled_memories", [])
    operations = payload.get("edit_operations", [])
    print("\n[Debug]")
    print(f"retrieval_query: {payload.get('retrieval_query') or '-'}")
    print(f"memory_update_status: {payload.get('memory_update_status') or '-'}")
    if recalled:
        print("recalled_memories:")
        for index, memory in enumerate(recalled, start=1):
            score = memory.get("score", 0.0)
            print(f"  {index}. ({score:.4f}) {memory.get('content', '')}")
    else:
        print("recalled_memories: none")
    if operations:
        print("edit_operations:")
        for index, operation in enumerate(operations, start=1):
            print(
                f"  {index}. {operation.get('op')} "
                f"[{operation.get('memory_type', 'fact')}] "
                f"{operation.get('content') or operation.get('memory_id') or ''}"
            )
    else:
        print("edit_operations: none")
    print()


def _next_auto_session() -> str:
    from datetime import datetime

    return "chat_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def run(args: argparse.Namespace) -> int:
    memory_root = _resolve_memory_root(args.memory_root)
    _validate_memory_root(args.api_url, memory_root, args.timeout)

    session_id = args.session
    top_k = args.top_k
    debug = args.debug

    payload = _health(args.api_url, args.timeout)
    if not payload:
        print(f"无法连接 Memory API：{args.api_url}", file=sys.stderr)
        return 1

    running_root = payload.get("details", {}).get("memory_root", "")
    print("Memory Chat")
    print(f"api: {args.api_url.rstrip('/')}")
    print(f"memory_root: {running_root or memory_root or '(unknown)'}")
    print(f"session_id: {session_id}")
    print(f"top_k: {top_k}")
    print("输入 /help 查看命令，/exit 退出。\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("已退出。")
            return 0
        if user_input == "/help":
            _print_help()
            continue
        if user_input.startswith("/new"):
            parts = user_input.split(maxsplit=1)
            session_id = parts[1].strip() if len(parts) == 2 else _next_auto_session()
            print(f"已切换到新会话：{session_id}")
            continue
        if user_input.startswith("/session "):
            session_id = user_input.split(maxsplit=1)[1].strip()
            print(f"当前 session_id: {session_id}")
            continue
        if user_input.startswith("/topk "):
            try:
                value = int(user_input.split(maxsplit=1)[1])
                if value <= 0:
                    raise ValueError
            except ValueError:
                print("用法：/topk 4")
                continue
            top_k = value
            print(f"top_k 已设为 {top_k}")
            continue
        if user_input.startswith("/debug"):
            value = user_input.split(maxsplit=1)[1].strip().lower() if " " in user_input else ""
            if value not in {"on", "off"}:
                print("用法：/debug on 或 /debug off")
                continue
            debug = value == "on"
            print(f"debug: {'on' if debug else 'off'}")
            continue
        if user_input.startswith("/memories"):
            parts = user_input.split(maxsplit=1)
            try:
                limit = int(parts[1]) if len(parts) == 2 else 10
            except ValueError:
                print("用法：/memories 10")
                continue
            try:
                _show_memories(args.api_url, limit, args.timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"读取记忆失败：{exc}")
            continue
        if user_input == "/card":
            try:
                _show_user_card(args.api_url, args.timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"读取 User Card 失败：{exc}")
            continue

        try:
            response = _post_json(
                args.api_url,
                "/chat",
                {
                    "session_id": session_id,
                    "user_input": user_input,
                    "top_k": top_k,
                    "debug_prompt": False,
                },
                args.timeout,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"请求失败：HTTP {exc.code} {detail}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"请求失败：{exc}")
            continue

        print(f"助手: {response.get('reply', '')}")
        if debug:
            _print_debug(response)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI for Memory Chat.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Memory API URL.")
    parser.add_argument("--session", default="default_chat", help="Chat session id.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of memories to recall.")
    parser.add_argument(
        "--memory-root",
        default=None,
        help="Expected memory bank root. Relative paths are resolved from the repo root.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--debug", action="store_true", help="Print recalled memories and updates.")
    return parser.parse_args(argv)


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
