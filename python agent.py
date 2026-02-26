# agent.py
import getpass
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

# -----------------------
# Config / Memory
# -----------------------
MEMORY_PATH = "memory.json"

def load_memory() -> Dict[str, Any]:
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"history": []}

def save_memory(mem: Dict[str, Any]) -> None:
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

# -----------------------
# Safety
# -----------------------
DANGEROUS_SUBSTRINGS = [
    "rm -rf", "del /f", "format", "mkfs", "shutdown", "reboot", "poweroff",
]
ALLOWED_COMMAND_PREFIXES = [
    "python", "python3", "pip", "pip3",
    "ruff", "black", "pytest",
    "git",
]

def is_safe_command(cmd: str) -> bool:
    lower = cmd.lower()
    if any(x in lower for x in DANGEROUS_SUBSTRINGS):
        return False
    cmd_strip = cmd.strip()
    return any(cmd_strip.startswith(pfx) for pfx in ALLOWED_COMMAND_PREFIXES)

# -----------------------
# Tools
# -----------------------
def run_cmd(cmd: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    if not is_safe_command(cmd):
        return {"ok": False, "error": f"Blocked command: {cmd}"}
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": (p.stdout or "")[-8000:],
            "stderr": (p.stderr or "")[-8000:],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"

def backup_file(path: str) -> str:
    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)
    return backup_path

# -----------------------
# API Key (user enters if missing)
# -----------------------
def ensure_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    print("🔑 OPENAI_API_KEY bulunamadı.")
    print("Lütfen API key girin (gizli yazılır, kaydedilmez):")
    key = getpass.getpass("API Key: ").strip()

    if not key:
        raise SystemExit("❌ API key girilmedi. Çıkılıyor.")

    os.environ["OPENAI_API_KEY"] = key  # sadece bu çalıştırma için
    return key

# -----------------------
# Format/Lint/Test helpers
# -----------------------
def ensure_tools(project_dir: str) -> None:
    for pkg in ["ruff", "black", "pytest"]:
        res = run_cmd(f"python -m pip show {pkg}", cwd=project_dir)
        if not res.get("ok"):
            run_cmd(f"python -m pip install -U {pkg}", cwd=project_dir)

def format_and_lint(project_dir: str) -> Dict[str, Any]:
    r1 = run_cmd("ruff check . --fix", cwd=project_dir)
    r2 = run_cmd("black .", cwd=project_dir)
    return {"ruff": r1, "black": r2}

def run_tests(project_dir: str) -> Dict[str, Any]:
    return run_cmd("pytest -q", cwd=project_dir)

FILE_RE = re.compile(r"([A-Za-z0-9_\-./\\]+\.py):(\d+)")
def extract_suspect_file(pytest_output: str) -> Optional[Dict[str, Any]]:
    m = FILE_RE.search(pytest_output)
    if not m:
        return None
    return {"path": m.group(1).replace("\\", "/"), "line": int(m.group(2))}

def snippet_around(content: str, line: int) -> str:
    lines = content.splitlines()
    start = max(0, line - 12)
    end = min(len(lines), line + 12)
    return "\n".join(f"{i+1:4d} | {lines[i]}" for i in range(start, end))

# -----------------------
# LLM Fix (OpenAI-compatible)
# -----------------------
def llm_fix_file(
    *,
    failing_command: str,
    error_output: str,
    file_path: str,
    file_content: str,
) -> Dict[str, Any]:
    api_key = ensure_api_key()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()

    system = (
        "You are a senior Python engineer.\n"
        "Return ONLY valid JSON. No markdown.\n\n"
        "Goal: Make tests pass with minimal safe change.\n"
        "Output schema:\n"
        '{"type":"file_rewrite","path":"<same path>","content":"<full new file content>","notes":"<short>"}\n'
        "Rules:\n"
        "- Keep changes minimal.\n"
        "- Do not change unrelated behavior.\n"
        "- Preserve formatting (black).\n"
        "- Never change the target path.\n"
        "- Never delete files.\n"
    )

    user = (
        f"Failing command:\n{failing_command}\n\n"
        f"Pytest output:\n{error_output}\n\n"
        f"Target file path:\n{file_path}\n\n"
        f"Current file content:\n<<<\n{file_content}\n>>>\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
    }

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code >= 400:
            return {"ok": False, "error": f"LLM HTTP {r.status_code}: {r.text[:2000]}"}
        data = r.json()
        msg = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"ok": False, "error": f"LLM request failed: {e}"}

    try:
        obj = json.loads(msg)
    except Exception:
        return {"ok": False, "error": "LLM did not return valid JSON.", "raw": msg[:2000]}

    if obj.get("type") != "file_rewrite":
        return {"ok": False, "error": f"Unexpected JSON type: {obj.get('type')}", "raw": obj}

    if obj.get("path") != file_path:
        return {"ok": False, "error": f"Path mismatch. Expected {file_path}, got {obj.get('path')}", "raw": obj}

    new_content = obj.get("content", "")
    if not isinstance(new_content, str) or len(new_content) < 5:
        return {"ok": False, "error": "LLM content is empty/invalid.", "raw": obj}

    return {"ok": True, "rewrite": obj}

# -----------------------
# Main Agent Loop
# -----------------------
def agent_run(project_dir: str = ".", max_loops: int = 6) -> str:
    mem = load_memory()
    report: List[str] = []
    report.append(f"Project: {os.path.abspath(project_dir)}")

    ensure_tools(project_dir)

    # Initial format/lint
    fl = format_and_lint(project_dir)
    report.append("\n== ruff ==")
    report.append(((fl["ruff"].get("stdout", "") + "\n" + fl["ruff"].get("stderr", "")).strip()) or "(no output)")
    report.append("\n== black ==")
    report.append(((fl["black"].get("stdout", "") + "\n" + fl["black"].get("stderr", "")).strip()) or "(no output)")

    failing_cmd = "pytest -q"

    for i in range(1, max_loops + 1):
        t = run_tests(project_dir)
        out = (t.get("stdout", "") + "\n" + t.get("stderr", "")).strip()

        report.append(f"\n== pytest run {i} ==")
        report.append(out or "(no output)")

        if t.get("ok"):
            report.append("\n✅ Tests passed.")
            break

        suspect = extract_suspect_file(out)
        if not suspect:
            report.append("\n❌ Tests failed. No file:line found; cannot auto-fix safely.")
            break

        target_rel = suspect["path"]
        target_abs = os.path.join(project_dir, target_rel)
        if not os.path.exists(target_abs):
            report.append(f"\n❌ Suspect file not found: {target_abs}")
            break

        current = read_file(target_abs)
        report.append("\n🎯 Suspect snippet:")
        report.append(snippet_around(current, suspect["line"]))

        llm_res = llm_fix_file(
            failing_command=failing_cmd,
            error_output=out,
            file_path=target_rel,
            file_content=current,
        )
        if not llm_res.get("ok"):
            report.append("\n❌ LLM fix failed:")
            report.append(str(llm_res.get("error")))
            if "raw" in llm_res:
                report.append("\nLLM raw (truncated):")
                report.append(str(llm_res["raw"])[:1500])
            break

        rewrite = llm_res["rewrite"]
        bak = backup_file(target_abs)
        write_file(target_abs, rewrite["content"])
        report.append(f"\n🛠️ Rewrote {target_rel} (backup: {os.path.basename(bak)})")
        if rewrite.get("notes"):
            report.append(f"Notes: {rewrite['notes']}")

        # Post-fix format/lint
        fl2 = format_and_lint(project_dir)
        report.append("\n== ruff (post-fix) ==")
        report.append(((fl2["ruff"].get("stdout", "") + "\n" + fl2["ruff"].get("stderr", "")).strip()) or "(no output)")
        report.append("\n== black (post-fix) ==")
        report.append(((fl2["black"].get("stdout", "") + "\n" + fl2["black"].get("stderr", "")).strip()) or "(no output)")

        time.sleep(0.2)

    mem["history"].append({"project": os.path.abspath(project_dir), "ts": int(time.time()), "tail": report[-12:]})
    save_memory(mem)
    return "\n".join(report)

if __name__ == "__main__":
    print(agent_run(project_dir=".", max_loops=6))
