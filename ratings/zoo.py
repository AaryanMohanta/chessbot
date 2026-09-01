"""Version zoo: frozen, self-contained snapshots of the agent, one per
directory under zoo/, each directly runnable by the existing harness
(``harness.match.play_game('zoo/<tag>/agent.py', ...)``) exactly like any
other agent path.

A snapshot is just the current shipped files (agent.py + cb_*.py, the same
whitelist tools/build_zip.py uses) copied flat into zoo/<tag>/, after
running that same build-time validation — so a version can't enter the
zoo unless it would also pass the submission build checks.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZOO_DIR = REPO_ROOT / "zoo"
MANIFEST_PATH = ZOO_DIR / "manifest.json"

sys.path.insert(0, str(REPO_ROOT / "tools"))
import build_zip  # noqa: E402


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"versions": {}}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def snapshot_version(tag: str, notes: str = "") -> Path:
    """Validate the current working tree's shipped files (same checks as
    the submission build) and copy them into zoo/<tag>/. Raises if the
    tag already exists or if validation fails."""
    dest = ZOO_DIR / tag
    if dest.exists():
        raise FileExistsError(f"zoo tag already exists: {tag} ({dest})")

    errors: list[str] = []
    build_zip.check_imports(build_zip.SHIPPED_PY_FILES, errors)
    build_zip.check_filename_shadowing(build_zip.SHIPPED_PY_FILES, errors)
    shipped_paths = [REPO_ROOT / f for f in build_zip.SHIPPED_PY_FILES if (REPO_ROOT / f).exists()]
    build_zip.check_forbidden_files(shipped_paths, errors)
    if errors:
        raise ValueError("refusing to snapshot, same checks build_zip.py enforces failed:\n" + "\n".join(errors))

    dest.mkdir(parents=True)
    for path in shipped_paths:
        shutil.copy2(path, dest / path.name)

    manifest = _load_manifest()
    manifest["versions"][tag] = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": sorted(p.name for p in shipped_paths),
        "notes": notes,
    }
    _save_manifest(manifest)
    return dest


def list_versions() -> list[tuple[str, dict]]:
    """(tag, metadata) pairs in the order they were snapshotted (i.e.
    version history order), from the manifest."""
    manifest = _load_manifest()
    return sorted(manifest["versions"].items(), key=lambda kv: kv[1]["created_at"])


def agent_path_for(tag: str) -> str:
    path = ZOO_DIR / tag / "agent.py"
    if not path.exists():
        raise FileNotFoundError(f"no zoo snapshot named {tag!r} (looked for {path})")
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot the current agent into the version zoo.")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="freeze the current agent under a version tag")
    snap.add_argument("tag", help="version tag, e.g. v1-alphabeta")
    snap.add_argument("--notes", default="", help="free-text note about this version")

    sub.add_parser("list", help="list zoo versions in snapshot order")

    args = parser.parse_args()

    if args.command == "snapshot":
        dest = snapshot_version(args.tag, notes=args.notes)
        print(f"OK: snapshotted into {dest}")
        return 0

    if args.command == "list":
        for tag, meta in list_versions():
            note = f" - {meta['notes']}" if meta.get("notes") else ""
            print(f"{tag:30s} {meta['created_at']}{note}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
