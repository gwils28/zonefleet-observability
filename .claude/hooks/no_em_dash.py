#!/usr/bin/env python3
"""Hook Claude Code : interdit le tiret cadratin (U+2014).

- PreToolUse (Edit, MultiEdit, Write, NotebookEdit, Bash) : refuse l'appel si le texte
  écrit (nouveau contenu ou commande) contient le caractère. Le texte remplacé
  (`old_string`) n'est pas contrôlé, pour pouvoir corriger les occurrences existantes.
- Stop : si la dernière réponse contient le caractère, demande de la réécrire.
"""

import json
import sys

DASH = "\u2014"  # tiret cadratin, écrit en échappement pour ne pas déclencher le hook
HINT = "Remplace-le par une virgule, deux-points, des parenthèses ou un tiret simple '-'."


def written_text(tool_name: str, tool_input: dict) -> str:
    """Renvoie le texte que l'outil va écrire ou exécuter."""
    if tool_name == "Edit":
        return tool_input.get("new_string", "")
    if tool_name == "MultiEdit":
        return "\n".join(e.get("new_string", "") for e in tool_input.get("edits", []))
    if tool_name == "Write":
        return tool_input.get("content", "")
    if tool_name == "NotebookEdit":
        return tool_input.get("new_source", "")
    if tool_name == "Bash":
        return tool_input.get("command", "")
    return ""


def last_reply(transcript_path: str) -> str:
    """Concatène le texte des messages assistant depuis le dernier message utilisateur."""
    texts: list[str] = []
    with open(transcript_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (entry.get("message") or {}).get("content")
            if entry.get("type") == "user":
                # Un résultat d'outil n'est pas un nouveau tour utilisateur.
                is_tool_result = isinstance(content, list) and any(
                    isinstance(c, dict) and c.get("type") == "tool_result" for c in content
                )
                if not is_tool_result:
                    texts = []
            elif entry.get("type") == "assistant":
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    texts += [c.get("text", "") for c in content if c.get("type") == "text"]
    return "\n".join(texts)


def main() -> None:
    data = json.load(sys.stdin)
    event = data.get("hook_event_name")

    if event == "PreToolUse":
        text = written_text(data.get("tool_name", ""), data.get("tool_input") or {})
        if DASH in text:
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Tiret cadratin (U+2014) interdit (CLAUDE.md, Typographie). {HINT}",
            }}))
        return

    if event == "Stop":
        if data.get("stop_hook_active"):
            return  # évite une boucle de blocage
        path = data.get("transcript_path")
        try:
            reply = last_reply(path) if path else ""
        except OSError:
            return
        if DASH in reply:
            print(json.dumps({
                "decision": "block",
                "reason": f"Ta réponse contient un tiret cadratin (U+2014), interdit par CLAUDE.md. "
                          f"Réécris-la sans. {HINT}",
            }))


if __name__ == "__main__":
    main()
