"""Rating tooling: version zoo, game database, joint MLE rating fits,
gauntlets, SPRT, and reporting.

Independent of engine internals — everything here only talks to a version
through the harness's existing wire-protocol subprocess interface
(``harness.match.play_game``), so it works on any agent conforming to that
protocol, not just this repo's own agent.py.
"""
