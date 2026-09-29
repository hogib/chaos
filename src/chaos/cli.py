"""The ``chaos`` command line.

Every setting comes from ``config.json``; anything given on the command line
overrides it for that invocation only (``--save`` writes the result back).
Order of precedence, lowest first: the config file, ``--set KEY=VALUE``, the
dedicated flags.

    chaos                                  # run exactly what config.json says
    chaos run --station ELBA --start 2025-04-16 --days 7
    chaos run --quiet --days 7 --samples 15 --min-mag 3
    chaos run --event 20250423_M6.3_Silivri --mean-period 2.6
    chaos run --set feature_extraction.win_sec=300 --dry-run
    chaos quiet --station ELBA --days 7    # list the quiet periods only
    chaos mean-period --station ELBA --refresh
    chaos config --days 3 --save           # edit config.json from the shell
"""

import argparse
import copy
import json
import re
import shlex
import sys
from pathlib import Path

# Sub-dicts that accept keys the shipped config does not list, because they
# pass straight through to another tool's options.
_OPEN_SECTIONS = ("mean_period_estimation.options",)

# argparse runs ``type`` on string defaults, so "not given" needs an object.
_UNSET = object()


def _parse_value(text: str):
    """Reads a ``--set`` value as JSON, falling back to a plain string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_set(config: dict, assignment: str) -> None:
    """Applies one ``dotted.key=value`` override in place.

    Raises:
        ValueError: If the assignment is malformed or names a key the config
            does not have (typos would otherwise be silently ignored).
    """
    if "=" not in assignment:
        raise ValueError(f"--set expects KEY=VALUE, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    keys = dotted.strip().split(".")
    node = config
    for i, key in enumerate(keys[:-1]):
        if not isinstance(node.get(key), dict):
            if ".".join(keys[:i + 1]) in _OPEN_SECTIONS and key not in node:
                node[key] = {}
            else:
                raise ValueError(f"--set: no section {'.'.join(keys[:i + 1])!r}"
                                 " in the config")
        node = node[key]
    leaf = keys[-1]
    parent = ".".join(keys[:-1])
    if leaf not in node and parent not in _OPEN_SECTIONS:
        raise ValueError(f"--set: unknown key {dotted!r}; known keys in "
                         f"{parent or 'the top level'}: {sorted(node)}")
    node[leaf] = _parse_value(raw)


def _put(config: dict, dotted: str, value) -> None:
    """Sets a key, creating sections as needed (for the dedicated flags)."""
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _mean_period(text: str):
    if text.lower() in ("auto", "null", "none", "estimate"):
        return None
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--mean-period takes seconds or 'auto', got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("--mean-period must be > 0")
    return value


def apply_overrides(config: dict, args) -> dict:
    """Returns a copy of ``config`` with the command line applied."""
    config = copy.deepcopy(config)
    for assignment in getattr(args, "set", None) or []:
        apply_set(config, assignment)

    def flag(name):
        return getattr(args, name, None)

    if flag("station") is not None:
        _put(config, "single_run.station", flag("station"))
    if flag("event") is not None:
        _put(config, "single_run.earthquake_name", flag("event"))
        # Naming an event means the old folder mode, whatever target says.
        _put(config, "target.start_date", None)
        _put(config, "target.quiet_window", False)
    if flag("start") is not None:
        _put(config, "target.start_date", flag("start"))
        _put(config, "target.quiet_window", False)
    if flag("days") is not None:
        _put(config, "target.duration_days", flag("days"))
    if flag("quiet") is not None:
        _put(config, "target.quiet_window", flag("quiet"))
    for name, key in (("samples", "quiet_window.samples"),
                      ("min_mag", "quiet_window.min_mag"),
                      ("max_dist", "quiet_window.max_dist_km"),
                      ("seed", "quiet_window.seed"),
                      ("jobs", "feature_extraction.n_jobs")):
        if flag(name) is not None:
            _put(config, key, flag(name))
    if flag("all") is not None:
        config["process_all"] = flag("all")
    if getattr(args, "mean_period", _UNSET) is not _UNSET:
        _put(config, "feature_extraction.features.rosenstein.mean_period",
             args.mean_period)
    return config


def format_config(config: dict) -> str:
    """JSON with short lists of scalars kept on one line, like the original."""
    text = json.dumps(config, indent=2, ensure_ascii=False)

    def collapse(match):
        items = [x.strip() for x in match.group(1).split(",")]
        return "[" + ", ".join(items) + "]"

    text = re.sub(r"\[\s*((?:[^\[\]{}\s][^\[\]{}]*?))\s*\]", collapse,
                  text)
    # A blank line between top-level sections, as in the hand-written file.
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if i > 1 and re.match(r'^  "', line):
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def _common(p):
    g = p.add_argument_group("configuration")
    g.add_argument("--config", type=Path, default=None,
                   help="config file (default: the project's config.json)")
    g.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config key, e.g. "
                        "feature_extraction.win_sec=300 (repeatable; value "
                        "is JSON)")
    g.add_argument("--save", action="store_true",
                   help="write the overridden settings back to the config "
                        "file")


def _target(p, quiet_flag=True):
    g = p.add_argument_group("what to run on")
    g.add_argument("--station", help="station code")
    if quiet_flag:
        g.add_argument("--event", help="raw/<station>/<EVENT> folder (event "
                                       "mode)")
        g.add_argument("--start", help="target period start, YYYY-MM-DD (UTC)")
    g.add_argument("--days", type=float, help="target period length in days")
    if quiet_flag:
        q = g.add_mutually_exclusive_group()
        q.add_argument("--quiet", dest="quiet", action="store_true",
                       default=None,
                       help="run on quiet periods picked from the catalog")
        q.add_argument("--no-quiet", dest="quiet", action="store_false")
    g = p.add_argument_group("quiet period selection")
    g.add_argument("--samples", type=int, help="how many periods to pick")
    g.add_argument("--min-mag", type=float,
                   help="events at or above this magnitude break quiet")
    g.add_argument("--max-dist", type=float,
                   help="only events within this many km count")
    g.add_argument("--seed", type=int, help="random seed for the pick")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chaos", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("run", help="run the pipeline (the default)")
    _common(p)
    _target(p)
    g = p.add_argument_group("run")
    a = g.add_mutually_exclusive_group()
    a.add_argument("--all", dest="all", action="store_true", default=None,
                   help="every station (period modes) or every event folder")
    a.add_argument("--single", dest="all", action="store_false",
                   help="only the configured/given station or event")
    g.add_argument("--jobs", type=int, help="worker processes (-1 = all)")
    g.add_argument("--mean-period", type=_mean_period, default=_UNSET,
                   metavar="SEC|auto",
                   help="Rosenstein mean period, or 'auto' to estimate it")
    g.add_argument("--refresh-mean-period", action="store_true",
                   help="re-estimate even if a cached value exists")
    g.add_argument("--dry-run", action="store_true",
                   help="show what would run, then stop")

    p = sub.add_parser("quiet", help="list quiet periods without running")
    _common(p)
    _target(p, quiet_flag=False)

    p = sub.add_parser("mean-period", help="estimate the mean period only")
    _common(p)
    p.add_argument("--station", help="station code")
    p.add_argument("--refresh", action="store_true",
                   help="re-estimate even if a cached value exists")

    p = sub.add_parser("config", help="print the effective configuration")
    _common(p)
    _target(p)
    p.add_argument("--jobs", type=int)
    p.add_argument("--mean-period", type=_mean_period, default=_UNSET,
                   metavar="SEC|auto")
    a = p.add_mutually_exclusive_group()
    a.add_argument("--all", dest="all", action="store_true", default=None)
    a.add_argument("--single", dest="all", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> None:
    from chaos import pipeline

    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("run", "quiet", "mean-period", "config")
    if not argv or (argv[0] not in commands and argv[0] not in ("-h",
                                                                "--help")):
        argv = ["run", *argv]
    args = build_parser().parse_args(argv)

    path = args.config or pipeline.CONFIG_PATH
    try:
        config = apply_overrides(pipeline.load_config(path), args)
    except ValueError as exc:
        raise SystemExit(f"chaos: {exc}") from None

    if args.save:
        path.write_text(format_config(config), encoding="utf-8")
        print(f"[config] saved to {path}")

    command = "chaos " + shlex.join(argv)
    try:
        if args.command == "run":
            pipeline.run(config, command=command, dry_run=args.dry_run,
                         refresh_mean_period=args.refresh_mean_period)
        elif args.command == "quiet":
            _quiet(pipeline, config)
        elif args.command == "mean-period":
            _mean_period_cmd(pipeline, config, args.refresh)
        elif args.command == "config":
            if not args.save:
                sys.stdout.write(format_config(config))
    except ValueError as exc:
        raise SystemExit(f"chaos: {exc}") from None


def _quiet(pipeline, config):
    from chaos.periods import discover_stations, quiet_periods

    days = (config.get("target") or {}).get("duration_days")
    if not days:
        raise ValueError("give --days or set target.duration_days")
    stations = (discover_stations(pipeline.SCRIPT_DIR, config)
                if config.get("process_all")
                else [config["single_run"]["station"]])
    for station in stations:
        periods, info = quiet_periods(pipeline.SCRIPT_DIR, config, station,
                                      days)
        print(f"\n{station}: {info['samples_found']} period(s) from "
              f"{info['candidates']} quiet start days "
              f"(M>={info['min_mag']:g}, {info['max_dist_km']} km, "
              f"seed {info['seed']})")
        for p in periods:
            print(f"  {p.label.split('/')[-1]}  {p.t0} -> {p.t1}")


def _mean_period_cmd(pipeline, config, refresh):
    from obspy import UTCDateTime

    from chaos.periods import Period

    station = config["single_run"]["station"]
    config = copy.deepcopy(config)
    # Estimate even when config.json pins a value: that is what was asked.
    config["feature_extraction"]["features"]["rosenstein"]["mean_period"] = None
    # The estimate is per station; any period makes a valid Settings.
    period = Period(UTCDateTime(0), UTCDateTime(86400), "_mean_period")
    cfg = pipeline.Settings(config, station, None, period=period,
                            make_dirs=False)
    cfg.resolve_mean_period(refresh=refresh)
    print(f"{station}: mean period "
          f"{cfg.FEATURES['rosenstein']['mean_period']:.3f} s")


if __name__ == "__main__":
    main()
