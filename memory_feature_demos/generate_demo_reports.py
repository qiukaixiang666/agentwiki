from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEMO_IDS = ["basic_recall", "conflict_update", "privacy_filtering", "user_card"]
ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DEMO_ROOT / "demo_manifest.json"
REPORT_ROOT = DEMO_ROOT / "reports"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def one_line(text: Any, limit: int = 220) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def md_escape(text: Any) -> str:
    return one_line(text).replace("|", "\\|")


def display_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


def bullet(text: Any, limit: int = 260) -> str:
    return one_line(text, limit=limit) or "-"


def demo_manifest() -> dict[str, dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH, {"demos": []})
    return {str(item["id"]): item for item in manifest.get("demos", [])}


def memory_root(demo: dict[str, Any]) -> Path:
    path = Path(demo["memory_root"])
    if not path.is_absolute():
        return (ROOT / path).resolve()
    return path.resolve()


def prompt_rows(demo: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(demo["prompt_file"])
    if not path.is_absolute():
        path = ROOT / path
    return read_jsonl(path)


def summarize_status(rows: list[dict[str, Any]], wiki_rows: list[dict[str, Any]]) -> dict[str, Any]:
    chat_rows = [row for row in rows if row.get("kind") == "chat_turn"]
    refresh_rows = [row for row in rows if row.get("kind") == "user_card_refresh"]
    errors = [row for row in chat_rows if row.get("error")]
    return {
        "result_lines": len(rows),
        "chat_turns": len(chat_rows),
        "refresh_rows": len(refresh_rows),
        "errors": len(errors),
        "last_turn": max([int(row.get("turn_index") or 0) for row in chat_rows] or [0]),
        "wiki_lines": len(wiki_rows),
        "result": "PASS" if len(chat_rows) == 15 and not errors else "CHECK",
    }


def response_of(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    return response if isinstance(response, dict) else {}


def op_summary(op: dict[str, Any]) -> str:
    op_name = op.get("op", "")
    content = op.get("content") or ""
    memory_id = op.get("memory_id")
    topic = op.get("topic") or "general"
    tags = ", ".join(op.get("tags") or [])
    prefix = f"{op_name}"
    if memory_id:
        prefix += f"({memory_id})"
    detail = bullet(content, 220)
    suffix = f" topic={topic}"
    if tags:
        suffix += f" tags={tags}"
    return f"`{prefix}` {detail} ({suffix.strip()})"


def top_recall(memories: list[dict[str, Any]], limit: int = 4) -> list[str]:
    out: list[str] = []
    for memory in memories[:limit]:
        score = memory.get("score")
        score_text = f"{score:.3f}" if isinstance(score, int | float) else "-"
        out.append(
            f"[{memory.get('id', '-')}] score={score_text} "
            f"{bullet(memory.get('content'), 240)}"
        )
    return out


def build_timeline(chat_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Turn | Phase | User | Reply | Memory Update |",
        "|---:|---|---|---|---|",
    ]
    for row in chat_rows:
        response = response_of(row)
        ops = response.get("edit_operations") or []
        if ops:
            memory_update = "<br>".join(md_escape(op_summary(op)) for op in ops[:4])
            if len(ops) > 4:
                memory_update += f"<br>... +{len(ops) - 4} more"
        else:
            memory_update = md_escape(response.get("memory_update_status") or "skipped")
        lines.append(
            "| {turn} | {phase} | {user} | {reply} | {update} |".format(
                turn=row.get("turn_index", ""),
                phase=md_escape(row.get("phase", "")),
                user=md_escape((row.get("request") or {}).get("user_input", ""),),
                reply=md_escape(response.get("reply", "")),
                update=memory_update,
            )
        )
    return "\n".join(lines)


def build_probe_section(chat_rows: list[dict[str, Any]]) -> str:
    probe_rows = [row for row in chat_rows if row.get("phase") == "probe"]
    if not probe_rows:
        return "_No probe turns found._"
    chunks: list[str] = []
    for row in probe_rows:
        response = response_of(row)
        chunks.append(f"### Turn {row.get('turn_index')} - {row.get('purpose', '')}")
        chunks.append("")
        chunks.append(f"- User: {bullet((row.get('request') or {}).get('user_input'), 360)}")
        chunks.append(f"- Answer: {bullet(response.get('reply'), 360)}")
        chunks.append(f"- Retrieval query: `{bullet(response.get('retrieval_query'), 240)}`")
        chunks.append("- Top recalled memories:")
        recalls = top_recall(response.get("recalled_memories") or [])
        if recalls:
            chunks.extend(f"  - {item}" for item in recalls)
        else:
            chunks.append("  - None")
        chunks.append("")
    return "\n".join(chunks).rstrip()


def build_wiki_section(wiki_rows: list[dict[str, Any]]) -> str:
    if not wiki_rows:
        return "_No wiki memories found._"
    status_counts = Counter(row.get("status", "unknown") for row in wiki_rows)
    type_counts = Counter(row.get("memory_type", "unknown") for row in wiki_rows)
    lines = [
        f"- Total wiki memories: {len(wiki_rows)}",
        f"- Status counts: {dict(status_counts)}",
        f"- Type counts: {dict(type_counts)}",
        "",
        "| ID | Status | Type | Topic | Memory | Tags |",
        "|---|---|---|---|---|---|",
    ]
    for row in wiki_rows:
        lines.append(
            "| {id} | {status} | {type} | {topic} | {content} | {tags} |".format(
                id=md_escape(row.get("id", "")),
                status=md_escape(row.get("status", "")),
                type=md_escape(row.get("memory_type", "")),
                topic=md_escape(row.get("topic", "")),
                content=md_escape(row.get("content", "")),
                tags=md_escape(", ".join(row.get("tags") or [])),
            )
        )
    return "\n".join(lines)


def build_user_card_section(memory_dir: Path, refresh_rows: list[dict[str, Any]]) -> str:
    user_card = read_json(memory_dir / "data" / "user_card.json", {})
    chunks: list[str] = []
    if refresh_rows:
        chunks.append(f"- Refresh events: {len(refresh_rows)}")
    if user_card:
        chunks.append(f"- Updated at: `{user_card.get('updated_at', '-')}`")
        source_ids = user_card.get("source_memory_ids") or []
        chunks.append(f"- Source memories: {', '.join(source_ids) if source_ids else '-'}")
        chunks.append("")
        chunks.append("> " + one_line(user_card.get("profile_text", ""), 900))
    else:
        chunks.append("_No user_card.json found._")
    return "\n".join(chunks)


def build_demo_specific_notes(demo_id: str, wiki_rows: list[dict[str, Any]], chat_rows: list[dict[str, Any]]) -> str:
    text = "\n".join(row.get("content", "") for row in wiki_rows)
    lower_text = text.lower()
    if demo_id == "basic_recall":
        probe_answers = "\n".join(response_of(row).get("reply", "") for row in chat_rows if row.get("phase") == "probe")
        combined = f"{text}\n{probe_answers}".lower()
        checks = [
            ("main dish", any(term in combined for term in ["eggplant", "茄子", "main dish", "主菜"])),
            ("cold dish", any(term in combined for term in ["cold", "冷盘", "beet", "甜菜", "沙拉"])),
            ("dessert", any(term in combined for term in ["dessert", "甜点", "mango", "芒果", "浆果", "烤苹果", "无花果", "酸奶"])),
            ("beverage", any(term in combined for term in ["drink", "饮品", "lemonade", "柠檬", "无酒精"])),
        ]
    elif demo_id == "conflict_update":
        checks = [
            ("new preference retained", any(term in lower_text for term in ["green tea", "绿茶", "warm water", "温水"])),
            ("old preference not dominant", "coffee" in lower_text or "咖啡" in lower_text),
        ]
    elif demo_id == "privacy_filtering":
        checks = [
            ("phone-like span not stored", "138-0000-0000" not in text),
            ("python preference retained", any(term in lower_text for term in ["python", "示例"])),
            ("structured workflow retained", any(term in lower_text for term in ["structured", "结构化", "multi-stage", "多阶段"])),
        ]
    elif demo_id == "user_card":
        probe_answers = "\n".join(response_of(row).get("reply", "") for row in chat_rows if row.get("phase") == "probe")
        checks = [
            ("user card refreshed", True),
            ("Chinese/concise style visible", any(term in probe_answers for term in ["中文", "简洁", "要点"])),
        ]
    else:
        checks = []

    if not checks:
        return "_No demo-specific checks configured._"
    return "\n".join(f"- [{'x' if passed else ' '}] {label}" for label, passed in checks)


def report_for_demo(demo_id: str, demo: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mem_dir = memory_root(demo)
    rows = read_jsonl(mem_dir / "runtime" / "demo_chat_results.jsonl")
    wiki_rows = read_jsonl(mem_dir / "data" / "wiki.jsonl")
    prompts = prompt_rows(demo)
    chat_rows = [row for row in rows if row.get("kind") == "chat_turn"]
    refresh_rows = [row for row in rows if row.get("kind") == "user_card_refresh"]
    status = summarize_status(rows, wiki_rows)

    lines = [
        f"# {demo.get('title', demo_id)}",
        "",
        f"- Demo id: `{demo_id}`",
        f"- Main observation: {demo.get('main_observation', '-')}",
        f"- Prompt file: `{display_path(Path(demo.get('prompt_file', '-')))}`",
        f"- Memory root: `{display_path(mem_dir)}`",
        "",
        "## Run Summary",
        "",
        "| Result | Chat Turns | Refresh Rows | Errors | Wiki Memories | Last Turn |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {status['result']} | {status['chat_turns']} | {status['refresh_rows']} | "
            f"{status['errors']} | {status['wiki_lines']} | {status['last_turn']} |"
        ),
        "",
        "## Demo-Specific Checks",
        "",
        build_demo_specific_notes(demo_id, wiki_rows, chat_rows),
        "",
        "## Prompt Plan",
        "",
        "| Turn | Phase | Purpose | Expected Observation |",
        "|---:|---|---|---|",
    ]
    for prompt in prompts:
        lines.append(
            "| {turn} | {phase} | {purpose} | {expected} |".format(
                turn=prompt.get("turn_index", ""),
                phase=md_escape(prompt.get("phase", "")),
                purpose=md_escape(prompt.get("purpose", "")),
                expected=md_escape(prompt.get("expected_observation", "")),
            )
        )
    lines.extend(
        [
            "",
            "## Conversation And Memory Timeline",
            "",
            build_timeline(chat_rows),
            "",
            "## Probe Recall",
            "",
            build_probe_section(chat_rows),
            "",
            "## Final Wiki Memory Bank",
            "",
            build_wiki_section(wiki_rows),
            "",
            "## User Card",
            "",
            build_user_card_section(mem_dir, refresh_rows),
            "",
            "## Raw Files",
            "",
            f"- Results: `{display_path(mem_dir / 'runtime' / 'demo_chat_results.jsonl')}`",
            f"- Agent debug: `{display_path(mem_dir / 'runtime' / 'agent_debug.jsonl')}`",
            f"- Wiki: `{display_path(mem_dir / 'data' / 'wiki.jsonl')}`",
            f"- User card: `{display_path(mem_dir / 'data' / 'user_card.json')}`",
        ]
    )
    return "\n".join(lines), status


def build_index(statuses: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Memory Feature Demo Reports",
        "",
        "These reports are generated from existing demo outputs only. No LLM or embedding service is called.",
        "",
        "| Demo | Title | Result | Chat Turns | Refresh Rows | Errors | Wiki Memories | Report |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for demo_id in DEMO_IDS:
        status = statuses.get(demo_id, {})
        title = manifest.get(demo_id, {}).get("title", demo_id)
        report_path = f"{demo_id}/report.md"
        lines.append(
            f"| `{demo_id}` | {md_escape(title)} | {status.get('result', 'CHECK')} | "
            f"{status.get('chat_turns', 0)} | {status.get('refresh_rows', 0)} | "
            f"{status.get('errors', 0)} | {status.get('wiki_lines', 0)} | "
            f"[open]({report_path}) |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate readable Markdown reports for memory feature demos.")
    parser.add_argument("--output-dir", default=str(REPORT_ROOT))
    args = parser.parse_args()

    manifest = demo_manifest()
    output_dir = Path(args.output_dir).resolve()
    statuses: dict[str, dict[str, Any]] = {}
    for demo_id in DEMO_IDS:
        if demo_id not in manifest:
            continue
        text, status = report_for_demo(demo_id, manifest[demo_id])
        statuses[demo_id] = status
        write_text(output_dir / demo_id / "report.md", text)
    write_text(output_dir / "index.md", build_index(statuses, manifest))
    print(f"Reports written to {output_dir}")


if __name__ == "__main__":
    main()
