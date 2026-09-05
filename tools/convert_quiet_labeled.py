"""Converts Zurichess's quiet-labeled.epd (FEN + `c9 "<result>";` opcode)
into the fen,result CSV format ratings/texel_tune.py's load_dataset()
already expects (same format tools/generate_texel_data.py produces),
so both data sources are interchangeable.

EPD lines carry a 4-field FEN (no halfmove/fullmove counters) -- appended
here as "0 1" since nothing in the eval or feature extraction depends on
their real values.

Usage: python tools/convert_quiet_labeled.py [in_epd] [out_csv]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_RESULT_TO_WHITE_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def convert(in_epd: Path, out_csv: Path) -> int:
    n = 0
    with open(in_epd, encoding="utf-8") as fin, open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["fen", "result"])
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # "<4-field FEN> c9 \"<result>\";"
            fen_part, _, rest = line.partition(" c9 ")
            result_str = rest.strip().strip(";").strip('"')
            score = _RESULT_TO_WHITE_SCORE.get(result_str)
            if score is None:
                continue
            fen = f"{fen_part} 0 1"
            writer.writerow([fen, score])
            n += 1
    return n


if __name__ == "__main__":
    in_epd = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "ratings" / "data" / "quiet-labeled.epd"
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "ratings" / "data" / "quiet_labeled.csv"
    count = convert(in_epd, out_csv)
    print(f"Wrote {count} positions to {out_csv}")
