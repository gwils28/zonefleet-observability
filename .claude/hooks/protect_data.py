#!/usr/bin/env python3
"""Hook PreToolUse : refuse toute écriture par outil dans data/raw/ (lecture seule).

Les fichiers bruts TLC sont téléchargés par `make data` et vérifiés par leur manifeste ;
les modifier à la main casserait la reproductibilité et le data contract.
"""

import json
import sys
from pathlib import PurePath

PROTECTED = ("data/raw/",)


def main() -> None:
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    rel = PurePath(path).as_posix()
    if any(p in rel for p in PROTECTED):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "data/raw/ est en lecture seule : régénérer via `make data`.",
        }}))


if __name__ == "__main__":
    main()
