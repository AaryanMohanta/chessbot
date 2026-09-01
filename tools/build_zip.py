"""Builds the submission zip and refuses to write it if any check fails.

Checks enforced (see repo README for the full constraint list this
encodes):
  - agent.py is present and will sit at the ZIP ROOT (no wrapping folder).
  - Every shipped .py file only imports stdlib modules, or the exact
    allowed third-party set, or another shipped module — found by
    AST-scanning imports, not by running the code.
  - No shipped file's module name shadows a stdlib module or an allowed
    third-party package (e.g. can't ship chess.py, random.py, numpy.py).
  - No native binaries (.pyd/.so/.dll/.dylib/.pyc) or __pycache__ dirs.
  - Total unzipped size stays under the 50 MB cap.

Run: python tools/build_zip.py [output_zip_path]
"""
from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "dist" / "submission.zip"

MAX_UNZIPPED_BYTES = 50 * 1024 * 1024

# Explicit whitelist, not a glob: adding a new module to the submission is
# a deliberate act, not something that happens by naming a file cb_*.py.
SHIPPED_PY_FILES = [
    "agent.py",
    "cb_engine.py",
    "cb_search.py",
    "cb_eval.py",
    "cb_time.py",
    "cb_order.py",
    "cb_tables.py",
    "cb_tt.py",
]

# Weight files are optional and picked up automatically if present at repo
# root, since the constraints explicitly allow shipping them.
WEIGHT_EXTENSIONS = {".onnx", ".safetensors", ".pt"}

ALLOWED_THIRD_PARTY = {"torch", "numpy", "chess", "onnxruntime", "numba"}

FORBIDDEN_SUFFIXES = {".pyd", ".so", ".dll", ".dylib", ".pyc"}


def _allowed_stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", ()))
    # builtins/sys are always fine to import even if not listed
    names |= {"builtins"}
    return names


def _shipped_module_basenames(shipped_files: list[str]) -> set[str]:
    return {Path(f).stem for f in shipped_files} | {"agent"}


def _collect_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Returns [(root_module_name, lineno), ...]. A relative import
    (``from . import x``) yields root_module_name "" so callers can flag it
    explicitly — flat submissions have no package to import from."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                found.append(("", node.lineno))
            elif node.module:
                found.append((node.module.split(".")[0], node.lineno))
    return found


def check_imports(shipped_files: list[str], errors: list[str]) -> None:
    allowed_stdlib = _allowed_stdlib_names()
    shipped_basenames = _shipped_module_basenames(shipped_files)

    for filename in shipped_files:
        path = REPO_ROOT / filename
        if not path.exists():
            errors.append(f"missing shipped file: {filename}")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as exc:
            errors.append(f"{filename}: syntax error: {exc}")
            continue

        for root_name, lineno in _collect_imports(tree):
            if root_name == "":
                errors.append(f"{filename}:{lineno}: relative import not allowed in a flat submission")
                continue
            if (
                root_name in allowed_stdlib
                or root_name in ALLOWED_THIRD_PARTY
                or root_name in shipped_basenames
            ):
                continue
            errors.append(
                f"{filename}:{lineno}: disallowed import '{root_name}' "
                f"(not stdlib, not in {sorted(ALLOWED_THIRD_PARTY)}, not a shipped module)"
            )


def check_filename_shadowing(shipped_files: list[str], errors: list[str]) -> None:
    allowed_stdlib = _allowed_stdlib_names()
    for filename in shipped_files:
        stem = Path(filename).stem
        if stem in allowed_stdlib:
            errors.append(f"{filename}: filename shadows stdlib module '{stem}'")
        if stem in ALLOWED_THIRD_PARTY:
            errors.append(f"{filename}: filename shadows allowed package '{stem}'")


def check_forbidden_files(all_files: list[Path], errors: list[str]) -> None:
    for path in all_files:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden native/compiled file: {path.name}")
        if "__pycache__" in path.parts:
            errors.append(f"__pycache__ must not be shipped: {path}")


def discover_weight_files() -> list[Path]:
    return sorted(
        p for p in REPO_ROOT.iterdir() if p.is_file() and p.suffix.lower() in WEIGHT_EXTENSIONS
    )


def build(output_path: Path = DEFAULT_OUTPUT) -> None:
    errors: list[str] = []

    if not (REPO_ROOT / "agent.py").exists():
        errors.append("agent.py not found at repo root")

    check_imports(SHIPPED_PY_FILES, errors)
    check_filename_shadowing(SHIPPED_PY_FILES, errors)

    weight_files = discover_weight_files()
    shipped_paths = [REPO_ROOT / f for f in SHIPPED_PY_FILES] + weight_files
    existing_paths = [p for p in shipped_paths if p.exists()]

    check_forbidden_files(existing_paths, errors)

    total_size = sum(p.stat().st_size for p in existing_paths)
    if total_size >= MAX_UNZIPPED_BYTES:
        errors.append(
            f"unzipped size {total_size} bytes exceeds the 50 MB cap ({MAX_UNZIPPED_BYTES} bytes)"
        )

    if errors:
        print("BUILD FAILED — refusing to write zip:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in existing_paths:
            zf.write(path, arcname=path.name)  # flat: no wrapping folder

    print(f"OK: wrote {output_path} ({total_size} bytes unzipped, {len(existing_paths)} files)")
    print("Files:")
    for path in existing_paths:
        print(f"  {path.name}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    build(out)
