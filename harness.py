#!/usr/bin/env python3
# harness.py
# MVP prompt-refinement runner for Bedrock (Anthropic Claude) + domain YAMLs

import os
import re
import csv
import json
import yaml
import time
import argparse
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# -----------------------------
# Config & setup
# -----------------------------
load_dotenv()

DEFAULT_REGION = os.getenv("AWS_REGION", "us-west-2")
DEFAULT_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
DEFAULT_PROFILE = os.getenv("AWS_PROFILE")  # optional

GLOBAL_PROMPT_PATH = Path("global_system_prompt.txt")
PROMPTS_DIR = Path("prompts")
CASES_PATH = Path("eval/cases.yaml")
MANIFEST_PATH = Path("manifest.yaml")  # optional

# Anthropic Messages API schema for Bedrock
ANTHROPIC_VERSION = "bedrock-2023-05-31"


# -----------------------------
# Utilities
# -----------------------------
def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ensure_exists(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")


def timestamp_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def bedrock_client(profile: str | None, region: str):
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
        return session.client("bedrock-runtime", region_name=region)
    return boto3.client("bedrock-runtime", region_name=region)


def to_sentence_count(text: str) -> int:
    # naive sentence count: ., !, ?
    return len(re.findall(r"[.!?]+(\s|$)", text))


def brief(text: str, max_sentences: int = 2) -> bool:
    return to_sentence_count(text) <= max_sentences


def no_question_stacking(text: str, max_q: int = 2) -> bool:
    return text.count("?") <= max_q


def no_unexpected_handoff(text: str) -> bool:
    return "HANDOFF_READY" not in text


def basic_traits_check(text: str, expect_traits: list[str] | None = None) -> dict:
    checks = {
        "brief": brief(text),
        "no_question_stacking": no_question_stacking(text),
        "no_unexpected_handoff": no_unexpected_handoff(text),
        "under_500_chars": len(text) <= 500,
    }
    # Optionally annotate trait expectations (heuristic)
    if expect_traits:
        for trait in expect_traits:
            if trait not in checks:
                # soft-true placeholder for clinician-scored traits
                checks[trait] = True
    return checks


def load_manifest_active_versions(env: str) -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    doc = load_yaml(MANIFEST_PATH)
    return doc.get("defaults", {}).get(env)


def resolve_prompt_path(domain: str, active_versions: dict | None) -> Path:
    if active_versions and domain in active_versions:
        prompt_id = active_versions[domain]  # e.g., "interpersonal-v2"
        version = prompt_id.split("-")[-1]   # crude split; expects id like "<domain>-vN"
        # allow both "v2" and "interpersonal-v2" naming; fallback to vN.yaml
        candidate = PROMPTS_DIR / domain / f"{version}.yaml"
        if candidate.exists():
            return candidate
        # fallback to exact id as filename
        candidate2 = PROMPTS_DIR / domain / f"{prompt_id}.yaml"
        if candidate2.exists():
            return candidate2
    # default to v1.yaml
    return PROMPTS_DIR / domain / "v1.yaml"


def load_cases(cases_path: Path):
    """
    Supports two formats:
    A) grouped by domain:
       interpersonal: ["...", "..."]
       emotion: ["...", "..."]
    B) list of objects:
       - id: INT_01
         domain: interpersonal
         input: "text"
         expect_traits: [brief]
    Returns: list of dicts with keys: id (auto if missing), domain, input, expect_traits
    """
    raw = load_yaml(cases_path)
    normalized = []
    if isinstance(raw, dict):
        for domain, inputs in raw.items():
            for i, text in enumerate(inputs, start=1):
                normalized.append({
                    "id": f"{domain[:3].upper()}_{i:02d}",
                    "domain": domain,
                    "input": str(text),
                    "expect_traits": []
                })
    elif isinstance(raw, list):
        for item in raw:
            normalized.append({
                "id": item.get("id") or f"{item['domain'][:3].upper()}_{len(normalized)+1:02d}",
                "domain": item["domain"],
                "input": item["input"],
                "expect_traits": item.get("expect_traits", []),
            })
    else:
        raise ValueError("Unsupported cases.yaml format.")
    return normalized


def anthropic_chat(client, model_id: str, system: str, user_text: str,
                   max_tokens: int = 220, temperature: float = 0.2,
                   retries: int = 2, retry_delay: float = 1.5) -> str:
    """
    Calls Anthropic Messages API on Bedrock. Returns text content.
    """
    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_text}
             ]}
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
                body=payload,
            )
            data = json.loads(resp["body"].read())
            parts = data.get("content", [])
            text = "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
            return text
        except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise
    raise last_err


def make_system_prompt(global_rules: str, agent_core: str, domain: str) -> str:
    # Keep it simple and stable
    return f"{global_rules}\n\nAGENT DOMAIN: {domain.upper()}\n\n{agent_core}"


def save_csv(rows: list[dict], path: Path):
    fields = ["timestamp", "env", "model_id", "domain", "prompt_file", "case_id", "input", "output", "checks"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_md(rows: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"### {r['case_id']} — {r['domain']}\n\n")
            f.write(f"**Prompt file:** `{r['prompt_file']}`  \n")
            f.write(f"**Model:** `{r['model_id']}`  \n")
            f.write(f"**Input:** {r['input']}\n\n")
            f.write(f"**Output:** {r['output']}\n\n")
            f.write(f"**Checks:** `{r['checks']}`\n\n")
            f.write("---\n\n")


def save_json(rows: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Prompt refinement harness for Bedrock (Anthropic).")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"], help="Environment for manifest selection.")
    parser.add_argument("--cases", default=str(CASES_PATH), help="Path to cases.yaml.")
    parser.add_argument("--domains", nargs="*", default=["interpersonal", "emotion", "mindfulness", "distress"],
                        help="Limit to specific domains (default: all).")
    parser.add_argument("--outdir", default="eval/results", help="Output directory for results.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Override model ID.")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="AWS profile (or omit for default creds).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=220, help="Max tokens for generation.")
    args = parser.parse_args()

    # Verify required files
    ensure_exists(GLOBAL_PROMPT_PATH, "global system prompt")
    ensure_exists(Path(args.cases), "cases file")
    ensure_exists(PROMPTS_DIR, "prompts directory")

    # Load resources
    global_rules = read_text(GLOBAL_PROMPT_PATH)
    cases = load_cases(Path(args.cases))
    active_versions = load_manifest_active_versions(args.env)

    # Bedrock client
    client = bedrock_client(args.profile, args.region)

    # Output dir
    out_root = Path(args.outdir)
    out_ts_dir = timestamp_dir(out_root)

    rows = []
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run
    print(f"[config] env={args.env} model={args.model} region={args.region} profile={args.profile}\n")
    print(f"Cases: {len(cases)} | Domains: {', '.join(args.domains)}")
    print(f"Saving results to: {out_ts_dir}\n")

    for case in cases:
        domain = case["domain"]
        if domain not in args.domains:
            continue

        prompt_path = resolve_prompt_path(domain, active_versions)
        ensure_exists(prompt_path, f"prompt file for {domain}")

        agent_doc = load_yaml(prompt_path)
        agent_core = agent_doc.get("core_instructions", "").strip()
        if not agent_core:
            raise ValueError(f"No `core_instructions` in {prompt_path}")

        system_prompt = make_system_prompt(global_rules, agent_core, domain)
        user_text = case["input"]

        try:
            output = anthropic_chat(
                client=client,
                model_id=args.model,
                system=system_prompt,
                user_text=user_text,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            output = f"[ERROR] {type(e).__name__}: {e}"

        checks = basic_traits_check(output, case.get("expect_traits"))
        row = {
            "timestamp": ts_str,
            "env": args.env,
            "model_id": args.model,
            "domain": domain,
            "prompt_file": str(prompt_path),
            "case_id": case["id"],
            "input": user_text,
            "output": output,
            "checks": json.dumps(checks, ensure_ascii=False),
        }
        rows.append(row)

        # Pretty console output
        print(f"=== {domain.upper()} | {case['id']} ===")
        print(f"User: {user_text}")
        print(f"Bot : {output}")
        print(f"Checks: {checks}\n")

    # Save artifacts
    save_csv(rows, out_ts_dir / "results.csv")
    save_md(rows, out_ts_dir / "results.md")
    save_json(rows, out_ts_dir / "results.json")

    print(f"Saved: {out_ts_dir/'results.csv'}")
    print(f"Saved: {out_ts_dir/'results.md'}")
    print(f"Saved: {out_ts_dir/'results.json'}")


if __name__ == "__main__":
    main()

