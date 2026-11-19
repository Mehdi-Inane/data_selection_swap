#!/usr/bin/env python3
"""
Run an experiment described by a YAML file in configs/.

A config names, for each stage, the script to run, a grid of values and the
fixed arguments. Task i of a stage is the i-th point of the grid's cartesian
product, the last axis varying fastest, so a Slurm array index is one run.
Every argument becomes a flag of the stage's script:

    key: value      ->  --key value
    key: true       ->  --key          (false or null: the flag is left out)
    key: [a, b]     ->  --key a b

Grid axes become flags the same way, except an axis listed under `choices`,
whose value names a bundle that sets the `script` and/or adds `args`.
Strings may use {placeholders}: grid axes, `constants`, {results} (the
results root the scripts write under: RESULTS_ROOT, else SCRATCH, else
./results) and {data_root} (--data-root). A config with `status: planned`
describes an experiment whose code does not exist yet: it can be validated
and listed, not run.

    python scripts/experiments/run_config.py validate configs/*/*.yaml
    python scripts/experiments/run_config.py stages CONFIG
    python scripts/experiments/run_config.py count CONFIG STAGE
    python scripts/experiments/run_config.py list CONFIG [STAGE]
    python scripts/experiments/run_config.py run-task CONFIG STAGE TASK [--dry-run]
"""

import argparse
import functools
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from data_selection_swap.paths import default_results_root  # noqa: E402

TOP_LEVEL_KEYS = {"schema_version", "name", "description", "status", "requires",
                  "open_questions", "constants", "stages"}
STAGE_KEYS = {"script", "grid", "choices", "args"}
BUNDLE_KEYS = {"script", "args"}
RUNTIME_KEYS = ("results", "data_root")


class ConfigError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Loading and validation
# ─────────────────────────────────────────────────────────────────────────────

def load(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate(config, path)
    return config


def _scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def _check_args(args, where):
    if not isinstance(args, dict):
        raise ConfigError(f"{where}: args must be a mapping")
    for key, value in args.items():
        values = value if isinstance(value, list) else [value]
        if not isinstance(key, str) or not all(_scalar(v) for v in values):
            raise ConfigError(f"{where}: {key!r} must be a scalar or a list of scalars")


def validate(config, path="config"):
    """Raise ConfigError unless every task of every stage resolves."""
    def fail(message):
        raise ConfigError(f"{path}: {message}")

    if not isinstance(config, dict):
        fail("the file must hold a mapping")
    if set(config) - TOP_LEVEL_KEYS:
        fail(f"unknown keys {sorted(set(config) - TOP_LEVEL_KEYS)}")
    if config.get("schema_version") != 1:
        fail("schema_version must be 1")
    for key in ("name", "description"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            fail(f"{key} must be a non-empty string")
    status = config.get("status", "ready")
    if status not in ("ready", "planned"):
        fail("status must be 'ready' or 'planned'")
    if status == "planned" and not config.get("requires"):
        fail("a planned config must say what it requires")
    constants = config.get("constants") or {}
    if not isinstance(constants, dict) or not all(_scalar(v) for v in constants.values()):
        fail("constants must map names to scalars")
    stages = config.get("stages")
    if not isinstance(stages, dict) or not stages:
        fail("stages must be a non-empty mapping")

    for name, stage in stages.items():
        where = f"stage {name!r}"
        if not isinstance(stage, dict) or set(stage) - STAGE_KEYS:
            fail(f"{where} must be a mapping of {sorted(STAGE_KEYS)}")
        grid = stage.get("grid")
        if (not isinstance(grid, dict) or not grid
                or not all(isinstance(v, list) and v and all(_scalar(x) for x in v)
                           for v in grid.values())):
            fail(f"{where}: grid must map each axis to a non-empty list of scalars")
        _check_args(stage.get("args") or {}, f"{path} {where}")
        choices = stage.get("choices") or {}
        scripted = []
        for axis, bundles in choices.items():
            if axis not in grid:
                fail(f"{where}: choices for {axis!r}, which is not a grid axis")
            missing = [v for v in grid[axis] if v not in (bundles or {})]
            if missing:
                fail(f"{where}: no choice for {axis}={missing}")
            for label, bundle in bundles.items():
                if not isinstance(bundle, dict) or set(bundle) - BUNDLE_KEYS:
                    fail(f"{where}: choice {axis}={label} must be a mapping of script/args")
                _check_args(bundle.get("args") or {}, f"{path} {where} {axis}={label}")
            if any("script" in bundle for bundle in bundles.values()):
                scripted.append(axis)
        if len(scripted) > 1:
            fail(f"{where}: only one choice axis may set the script, not {scripted}")
        if not stage.get("script") and not scripted:
            fail(f"{where} has no script")
        bundle_keys = {key for bundles in choices.values() for bundle in bundles.values()
                       for key in (bundle.get("args") or {})}
        for axis in grid:
            if axis not in choices and (axis in (stage.get("args") or {}) or axis in bundle_keys):
                fail(f"{where}: {axis!r} is both a grid axis and an argument")

        runtime = {key: f"<{key}>" for key in RUNTIME_KEYS}
        for index in range(task_count(config, name)):
            script, _ = resolve(config, name, index, runtime)
            if status == "ready" and not (REPO_ROOT / script).is_file():
                fail(f"{where}: {script} does not exist")


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────

def task_count(config, stage):
    return math.prod(len(values) for values in config["stages"][stage]["grid"].values())


def grid_point(config, stage, index):
    """The index-th point of the stage's grid, the last axis varying fastest
    (the order the Slurm launchers decode SLURM_ARRAY_TASK_ID in)."""
    grid = config["stages"][stage]["grid"]
    if not 0 <= index < task_count(config, stage):
        raise ConfigError(f"task {index} is outside stage {stage!r} "
                          f"(0..{task_count(config, stage) - 1})")
    point = {}
    for axis, values in reversed(grid.items()):
        index, position = divmod(index, len(values))
        point[axis] = values[position]
    return {axis: point[axis] for axis in grid}


def _fill(value, values, where):
    if isinstance(value, list):
        return [_fill(v, values, where) for v in value]
    if not isinstance(value, str):
        return value
    try:
        return value.format_map(values)
    except KeyError as missing:
        raise ConfigError(f"{where}: unknown placeholder {missing}") from None


class _Values(dict):
    """Grid and runtime values; a constant is filled in when first used.

    Constants may refer to grid axes and runtime values, so one that names
    an axis only some stages have is fine as long as only those use it.
    """

    def __init__(self, base, constants):
        super().__init__(base)
        self._base = dict(base)
        self._constants = constants

    def __missing__(self, key):
        if key not in self._constants:
            raise KeyError(key)
        value = _fill(self._constants[key], self._base, f"constant {key!r}")
        self[key] = value
        return value


def resolve(config, stage, index, runtime):
    """(script, args) of one task, every placeholder filled in."""
    spec = config["stages"][stage]
    point = grid_point(config, stage, index)
    values = _Values({**runtime, **point}, config.get("constants") or {})
    choices = spec.get("choices") or {}
    script = spec.get("script")
    args = {axis: value for axis, value in point.items() if axis not in choices}
    args.update(spec.get("args") or {})
    for axis in choices:
        bundle = choices[axis][point[axis]]
        script = bundle.get("script", script)
        args.update(bundle.get("args") or {})
    where = f"{stage}[{index}]"
    return (_fill(script, values, where),
            {key: _fill(value, values, f"{where} {key}") for key, value in args.items()})


def to_flags(args):
    flags = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flags.append(f"--{key}")
        if value is not True:
            flags.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return flags


def command(config, stage, index, data_root):
    runtime = {"results": default_results_root(), "data_root": str(data_root)}
    script, args = resolve(config, stage, index, runtime)
    return [sys.executable, str(REPO_ROOT / script), *to_flags(args)]


def _environment():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT / "src"),
                                                      env.get("PYTHONPATH")]))
    return env


@functools.lru_cache(maxsize=None)
def _script_flags(script):
    """Flags listed by ``script --help``; empty when the script cannot run."""
    result = subprocess.run([sys.executable, str(REPO_ROOT / script), "--help"],
                            capture_output=True, text=True, env=_environment())
    return frozenset(re.findall(r"--([A-Za-z0-9_]+)", result.stdout))


def check_flags(config):
    """{script: [arguments its --help does not list]}; needs the full environment."""
    used = {}
    runtime = {key: f"<{key}>" for key in RUNTIME_KEYS}
    for stage in config["stages"]:
        for index in range(task_count(config, stage)):
            script, args = resolve(config, stage, index, runtime)
            used.setdefault(script, set()).update(args)
    missing = {}
    for script, keys in used.items():
        absent = sorted(keys - _script_flags(script))
        if absent:
            missing[script] = absent
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────

def _stage(config, stage):
    if stage not in config["stages"]:
        raise ConfigError(f"no stage {stage!r}; stages: {', '.join(config['stages'])}")
    return stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_cmd = commands.add_parser("validate", help="Check configs.")
    validate_cmd.add_argument("configs", nargs="+")
    validate_cmd.add_argument("--check-flags", action="store_true",
                              help="Also check every argument against its script's --help "
                                   "(needs the full environment).")
    stages_cmd = commands.add_parser("stages", help="Print the stage names.")
    count_cmd = commands.add_parser("count", help="Number of tasks in a stage.")
    list_cmd = commands.add_parser("list", help="Print the command of every task.")
    field_cmd = commands.add_parser("task-field", help="Print one argument of a task.")
    run_cmd = commands.add_parser("run-task", help="Run one task.")
    for sub in (stages_cmd, count_cmd, list_cmd, field_cmd, run_cmd):
        sub.add_argument("config")
    for sub in (count_cmd, field_cmd, run_cmd):
        sub.add_argument("stage")
    list_cmd.add_argument("stage", nargs="?")
    for sub in (field_cmd, run_cmd):
        sub.add_argument("task", type=int)
    field_cmd.add_argument("field")
    for sub in (list_cmd, field_cmd, run_cmd):
        sub.add_argument("--data-root", default="data",
                         help="Directory holding <dataset>_data folders (default: data).")
    run_cmd.add_argument("--dry-run", action="store_true", help="Print the command only.")
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            failed = False
            for path in args.configs:
                config = load(path)
                sizes = ", ".join(f"{s}={task_count(config, s)}" for s in config["stages"])
                print(f"ok  {path}  [{config.get('status', 'ready')}]  {sizes}")
                if args.check_flags:
                    for script, absent in check_flags(config).items():
                        planned = config.get("status") == "planned"
                        failed |= not planned
                        print(f"    {script}: {'to add' if planned else 'unknown'} "
                              f"{', '.join(absent)}")
            return 1 if failed else 0

        config = load(args.config)
        if args.command == "stages":
            print(" ".join(config["stages"]))
        elif args.command == "count":
            print(task_count(config, _stage(config, args.stage)))
        elif args.command == "list":
            for stage in [_stage(config, args.stage)] if args.stage else config["stages"]:
                for index in range(task_count(config, stage)):
                    print(f"{stage}[{index}]  "
                          f"{shlex.join(command(config, stage, index, args.data_root))}")
        elif args.command == "task-field":
            runtime = {"results": default_results_root(), "data_root": args.data_root}
            _, task_args = resolve(config, _stage(config, args.stage), args.task, runtime)
            if args.field not in task_args:
                raise ConfigError(f"{args.stage}[{args.task}] has no argument {args.field!r}")
            value = task_args[args.field]
            print(" ".join(map(str, value)) if isinstance(value, list) else value)
        elif args.command == "run-task":
            if config.get("status") == "planned":
                raise ConfigError(f"{args.config} is planned and cannot run yet; it "
                                  f"requires: {config['requires'].strip()}")
            argv = command(config, _stage(config, args.stage), args.task, args.data_root)
            print(shlex.join(argv), flush=True)
            if not args.dry_run:
                return subprocess.run(argv, env=_environment()).returncode
        return 0
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
