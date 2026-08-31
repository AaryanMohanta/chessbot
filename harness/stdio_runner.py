"""Subprocess entry point: loads an agent module and speaks the wire
protocol on stdio, one JSON object per line in each direction:

    stdin  -> {"fen": "<FEN>", "time_left_ms": <int>}
    stdout <- {"move": "<uci>"}                (success)
              {"error": "<ExceptionType>: msg"} (agent raised)

Before entering that loop it prints one readiness line,
``{"status": "ready"}``, immediately after the agent module finishes
importing (which is when ``Engine.__init__`` runs) — this lets the harness
measure init time separately from per-move time, mirroring the
competition's separate 60 s init budget vs. the per-move clock. This
readiness line is a harness convention for local testing, not part of the
real validator's wire protocol.

Usage: ``python stdio_runner.py <path-to-agent-module.py>``
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys


def load_agent(agent_path: str):
    """Import the agent module the same way the competition zip would see
    it: its own directory goes on sys.path so sibling ``cb_*`` imports
    resolve, mirroring "the zip goes first on sys.path"."""
    agent_dir = os.path.dirname(os.path.abspath(agent_path))
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load agent module from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: stdio_runner.py <path-to-agent-module.py>", file=sys.stderr)
        sys.exit(2)

    agent_path = sys.argv[1]
    module = load_agent(agent_path)

    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            move = module.get_move(request["fen"], int(request["time_left_ms"]))
            response = json.dumps({"move": move})
        except Exception as exc:  # the whole point: never let this crash the loop
            response = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        sys.stdout.write(response + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
