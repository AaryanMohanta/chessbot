"""Report command: the current ladder (every version, rating, 95% CI,
games played, sorted) plus a plot of rating over version history.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings import zoo  # noqa: E402
from ratings.db import DB_PATH, load_games  # noqa: E402
from ratings.elo import fit_ratings  # noqa: E402

DEFAULT_PLOT_PATH = REPO_ROOT / "ratings" / "ladder.png"


def print_ladder(db_path: Path = DB_PATH, anchor: str | None = None) -> None:
    games = load_games(db_path)
    if not games:
        print("No games recorded yet.")
        return

    if anchor is None:
        history = zoo.list_versions()
        anchor = history[0][0] if history else games[0].white_version

    fit = fit_ratings(games, anchor)

    print(f"{'version':30s} {'rating':>8s} {'95% CI':>10s} {'games':>7s} {'W-D-L':>14s}  flags")
    for e in fit.entries:
        wdl = f"{e.wins:.1f}-{e.draws}-{e.losses:.1f}"
        flags = ",".join(e.flags)
        print(f"{e.version:30s} {e.rating:8.1f} {'+/-' + f'{e.ci95:.1f}':>10s} {e.games:7d} {wdl:>14s}  {flags}")


def plot_history(db_path: Path = DB_PATH, anchor: str | None = None, out_path: Path = DEFAULT_PLOT_PATH) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    games = load_games(db_path)
    if not games:
        return None

    history = zoo.list_versions()
    if not history:
        return None
    if anchor is None:
        anchor = history[0][0]

    fit = fit_ratings(games, anchor)
    by_version = fit.by_version()

    tags = [tag for tag, _meta in history if tag in by_version]
    ratings = [by_version[tag].rating for tag in tags]
    errors = [by_version[tag].ci95 for tag in tags]

    fig, ax = plt.subplots(figsize=(max(6, len(tags) * 0.6), 4))
    ax.errorbar(range(len(tags)), ratings, yerr=errors, fmt="o-", capsize=3)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel(f"rating (anchor: {anchor} = 0)")
    ax.set_title("Rating over version history")
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the ratings ladder and plot rating over version history.")
    parser.add_argument("--anchor", default=None, help="version tag to pin at 0 (default: oldest zoo snapshot)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-path", default=str(DEFAULT_PLOT_PATH))
    args = parser.parse_args()

    print_ladder(anchor=args.anchor)

    if not args.no_plot:
        path = plot_history(anchor=args.anchor, out_path=Path(args.plot_path))
        if path:
            print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
