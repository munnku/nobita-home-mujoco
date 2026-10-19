"""TD から作った道具を、コマンドで見たり呼んだりする (Claude Code のセッションが LLM の代わりに使う)。

uv run python app/wot_cli.py list                          # 道具の一覧 (名前・説明・引数)
uv run python app/wot_cli.py call robot__checkRoom '{"room": "ousetsu"}'
uv run python app/wot_cli.py call read_property '{"target": "robot/state"}'
道具は app/llm_agent.py (Claude API) と同じもの (home/wot_tools.py)。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.wot_tools import THINGS, WoTTools  # noqa: E402


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "call"):
        print(__doc__)
        return 1
    tools = WoTTools(os.environ.get("WOT_THINGS", THINGS))
    if sys.argv[1] == "list":
        for t in tools.tools:
            props = t["input_schema"].get("properties", {})
            args = ", ".join(f"{k}: {v.get('type', '')}{' ' + str(v['enum']) if 'enum' in v and len(v['enum']) < 12 else ''}"
                             for k, v in props.items())
            print(f"{t['name']}({args})\n    {t['description']}")
        return 0
    name = sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    print(json.dumps(tools.call(name, args), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
