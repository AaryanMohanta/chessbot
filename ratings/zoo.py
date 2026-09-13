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
    # Zoo tags and build tags (see cb_build_tag.py) are the same kind of
    # thing -- "which config is this" -- so a zoo snapshot's build tag
    # defaults to its own zoo tag rather than falling back to the git-SHA
    # default build() would otherwise pick. Generated in-memory and
    # written straight to dest/, never to the live repo root -- see
    # build_zip.generate_build_tag_source's docstring: this function used
    # to call generate_build_tag_file(tag, {}) with its default path
    # (REPO_ROOT/cb_build_tag.py), which is exactly the mutate-the-shared-
    # file bug that hit ratings/sprt.py's live testing twice in one
    # session. A snapshot running concurrently with local self-play
    # testing had the identical exposure.
    build_tag_source = build_zip.generate_build_tag_source(tag, {})
    build_zip.check_imports(build_zip.SHIPPED_PY_FILES, errors, source_overrides={"cb_build_tag.py": build_tag_source})
    build_zip.check_filename_shadowing(build_zip.SHIPPED_PY_FILES, errors)
    shipped_paths = [
        REPO_ROOT / f for f in build_zip.SHIPPED_PY_FILES
        if f != "cb_build_tag.py" and (REPO_ROOT / f).exists()
    ]
    build_zip.check_forbidden_files(shipped_paths, errors)
    if errors:
        raise ValueError("refusing to snapshot, same checks build_zip.py enforces failed:\n" + "\n".join(errors))

    dest.mkdir(parents=True)
    for path in shipped_paths:
        shutil.copy2(path, dest / path.name)
    (dest / "cb_build_tag.py").write_text(build_tag_source, encoding="utf-8")

    manifest = _load_manifest()
    manifest["versions"][tag] = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": sorted([p.name for p in shipped_paths] + ["cb_build_tag.py"]),
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
