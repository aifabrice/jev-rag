#!/usr/bin/env python3
"""Minimal Jev smoke test for OpenRouter and TypeSafe's official API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROVIDERS = {
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/alpha/decisions",
        "model": "typesafe/jev-1.13",
    },
    "typesafe": {
        "env": "TYPESAFE_API_KEY",
        "url": "https://api.typesafe.ai/v1/systemone",
        "model": "jev-latest",
    },
}

DEFAULT_STATE = (
    "客户说：我昨天升级了 Pro 套餐并已扣款，但控制台仍显示 Free。"
    "我今天必须导出报表，请尽快处理。"
)

DEFAULT_QUESTIONS: dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "哪个团队最适合处理这条客户消息？",
        "criteria": {
            "billing": "扣款、账单、退款或订阅付款问题",
            "technical": "产品故障、报错或功能不可用",
            "sales": "售前咨询、报价或购买意向",
        },
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "这条消息是否表达了明确的紧迫性或时间限制？",
    },
    "frustration": {
        "type": "score",
        "instructions": "客户表现出的不满程度如何？",
        "criteria": ["平静陈述事实", "有些不满但保持克制", "非常愤怒或有攻击性"],
    },
}


def load_dotenv(path: Path) -> None:
    """Load a small, dependency-free subset of .env syntax."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def request_decision(
    provider: str,
    state: Any,
    questions: Any,
    timeout: float,
    retries: int = 2,
) -> dict[str, Any]:
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env"], "").strip()
    if not api_key:
        raise RuntimeError(f"缺少环境变量 {config['env']}")

    body = json.dumps(
        {"model": config["model"], "state": state, "questions": questions},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "jev-rag-smoke-test/0.2.0",
    }
    if provider == "openrouter":
        headers["X-Title"] = "Jev RAG Smoke Test"

    started = time.perf_counter()
    payload: dict[str, Any] = {}
    status = 0
    for attempt in range(retries + 1):
        request = urllib.request.Request(config["url"], data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                status = response.status
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
            if retryable and attempt < retries:
                time.sleep(0.4 * (2**attempt))
                continue
            try:
                detail = json.dumps(json.loads(detail), ensure_ascii=False)
            except json.JSONDecodeError:
                detail = detail[:1000]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                time.sleep(0.4 * (2**attempt))
                continue
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"网络请求失败（已重试 {retries} 次）: {reason}") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    if status != 200 or not isinstance(payload.get("answers"), dict):
        raise RuntimeError(f"响应格式异常: {json.dumps(payload, ensure_ascii=False)[:1000]}")
    # OpenRouter also returns an upstream `provider` (for example, TypeSafe).
    # Keep that field intact and report which gateway this script called separately.
    return {"gateway": provider, "elapsed_ms": elapsed_ms, **payload}


def selected_providers(value: str) -> list[str]:
    if value == "both":
        return ["openrouter", "typesafe"]
    if value != "auto":
        return [value]
    found = [name for name, cfg in PROVIDERS.items() if os.environ.get(cfg["env"], "").strip()]
    if not found:
        expected = " 或 ".join(cfg["env"] for cfg in PROVIDERS.values())
        raise RuntimeError(f"没有找到 API Key；请在 .env 中设置 {expected}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="本地测试 Jev（OpenRouter / TypeSafe 官方 API）")
    parser.add_argument("--provider", choices=["auto", "openrouter", "typesafe", "both"], default="auto")
    parser.add_argument("--text", help="要评估的文本；省略时使用内置示例")
    parser.add_argument("--state-file", help="从 JSON 文件读取 state（会覆盖 --text）")
    parser.add_argument("--questions-file", help="从 JSON 文件读取 questions")
    parser.add_argument("--env-file", default=".env", help="环境变量文件，默认 .env")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次请求超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只检查请求配置，不联网")
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))
    state = read_json(args.state_file) if args.state_file else (args.text or DEFAULT_STATE)
    questions = read_json(args.questions_file) if args.questions_file else DEFAULT_QUESTIONS

    if args.dry_run:
        preview = {
            "provider": args.provider,
            "state": state,
            "questions": questions,
            "configured": {
                name: bool(os.environ.get(cfg["env"], "").strip()) for name, cfg in PROVIDERS.items()
            },
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    try:
        providers = selected_providers(args.provider)
    except RuntimeError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for provider in providers:
        try:
            result = request_decision(provider, state, questions, args.timeout)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except RuntimeError as exc:
            failures += 1
            print(f"[{provider}] 调用失败: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
