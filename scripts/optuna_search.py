#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import optuna
except ImportError:
    optuna = None


VALID_RE = re.compile(
    r"step\s+(?P<step>\d+).*?"
    r"valid_expected_runtime\s+(?P<valid_expected_runtime>[0-9.]+)")
TEST_RE = re.compile(
    r"test_loss\s+(?P<test_loss>[0-9.]+).*?"
    r"test_expected_runtime\s+(?P<test_expected_runtime>[0-9.]+)")


def parse_args():
    bundle_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Optuna search for CSP Rank training/loss hyperparameters.")
    parser.add_argument("--models-dir", default=str(bundle_dir / "models/optuna_search"))
    parser.add_argument("--study-name", default="csp_rank_optuna")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL. Defaults to sqlite:///<models-dir>/study.db.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional Optuna wall-clock timeout in seconds.")
    parser.add_argument("--pruner-warmup-steps", type=int, default=2000,
                        help="Do not prune a trial before this training step.")
    parser.add_argument("--results-csv", default=None,
                        help="Defaults to <models-dir>/optuna_results.csv.")
    parser.add_argument("--best-json", default=None,
                        help="Defaults to <models-dir>/best_config.json.")
    parser.add_argument("--train-python", default="python3",
                        help="Python executable used for training. Defaults to python3.")

    parser.add_argument("--prop-embedding-size", type=int, default=64)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--activation", default="relu", choices=["relu", "elu"])
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)

    parser.add_argument("--learning-rate-min", type=float, default=2e-4)
    parser.add_argument("--learning-rate-max", type=float, default=6e-4)
    parser.add_argument("--pl-margin-min", type=float, default=1.5)
    parser.add_argument("--pl-margin-max", type=float, default=3.0)
    parser.add_argument("--dropout-rate-min", type=float, default=0.0)
    parser.add_argument("--dropout-rate-max", type=float, default=0.05)
    parser.add_argument("--min-pair-runtime-ratio-min", type=float, default=0.03)
    parser.add_argument("--min-pair-runtime-ratio-max", type=float, default=0.12)
    parser.add_argument("--pair-keep-floor-min", type=float, default=0.25)
    parser.add_argument("--pair-keep-floor-max", type=float, default=0.40)
    parser.add_argument("--pair-sampling-strategies", default="weighted",
                        help=("Comma-separated pair mining strategies Optuna may use. "
                              "Choices: weighted,threshold,all_pairs,topk_bottomk."))
    parser.add_argument("--pair-top-k-values", default="3",
                        help="Comma-separated K values Optuna may use with topk_bottomk mining.")
    parser.add_argument("--timeout-penalty-multipliers", default="1,2,5,10",
                        help="Comma-separated PAR multipliers Optuna may use for training timeouts.")
    parser.add_argument("--validation-timeout-penalty-multiplier", type=float, default=1.0,
                        help="Fixed validation PAR multiplier used to compare all trials.")
    parser.add_argument("--split-strategy", default="within_class_stratified",
                        choices=[
                            "random",
                            "within_class_stratified",
                            "within_class_bucket_stratified",
                        ])
    parser.add_argument("--sampler-seed", type=int, default=0)
    return parser.parse_args()


def require_optuna():
    if optuna is None:
        raise SystemExit(
            "Optuna is not installed. Install it with: python3 -m pip install --user optuna")


def hparam_string(params, args):
    values = {
        "learning_rate": params["learning_rate"],
        "pl_margin": params["pl_margin"],
        "hidden_units": args.hidden_units,
        "num_blocks": args.num_blocks,
        "prop_embedding_size": args.prop_embedding_size,
        "num_heads": args.num_heads,
        "dropout_rate": params["dropout_rate"],
        "activation": args.activation,
        "max_gradient_norm": args.max_gradient_norm,
    }
    return ",".join("%s=%s" % (key, values[key]) for key in [
        "learning_rate",
        "pl_margin",
        "hidden_units",
        "num_blocks",
        "prop_embedding_size",
        "num_heads",
        "dropout_rate",
        "activation",
        "max_gradient_norm",
    ])


def append_result(results_csv, row):
    first_write = not results_csv.is_file()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_number",
        "status",
        "best_valid_expected_runtime",
        "best_step",
        "test_expected_runtime",
        "test_loss",
        "elapsed_sec",
        "learning_rate",
        "pl_margin",
        "dropout_rate",
        "min_pair_runtime_ratio",
        "pair_keep_floor",
        "pair_sampling_strategy",
        "pair_top_k",
        "timeout_penalty_multiplier",
        "prop_embedding_size",
        "hidden_units",
        "num_heads",
        "num_blocks",
        "activation",
        "max_gradient_norm",
        "train_dir",
        "log_file",
    ]
    with open(results_csv, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()
        writer.writerow(row)


def update_best(best_json, row):
    if row.get("status") != "ok":
        return
    value = row.get("best_valid_expected_runtime")
    if value in (None, ""):
        return

    current = None
    if best_json.is_file():
        with open(best_json) as fin:
            current = json.load(fin)
    if current is not None:
        try:
            if float(current["best_valid_expected_runtime"]) <= float(value):
                return
        except (KeyError, TypeError, ValueError):
            pass

    best_json.parent.mkdir(parents=True, exist_ok=True)
    with open(best_json, "w") as fout:
        json.dump(row, fout, indent=2, sort_keys=True)


def suggest_params(trial, args):
    pair_sampling_strategies = [
        value.strip()
        for value in args.pair_sampling_strategies.split(",")
        if value.strip()
    ]
    valid_pair_sampling_strategies = set([
        "weighted",
        "threshold",
        "all_pairs",
        "topk_bottomk",
    ])
    invalid_strategies = [
        value for value in pair_sampling_strategies
        if value not in valid_pair_sampling_strategies
    ]
    if invalid_strategies:
        raise ValueError("Invalid --pair-sampling-strategies values: %s" %
                         ",".join(invalid_strategies))
    if not pair_sampling_strategies:
        raise ValueError("--pair-sampling-strategies must contain at least one value")
    pair_top_k_values = [
        int(value.strip())
        for value in args.pair_top_k_values.split(",")
        if value.strip()
    ]
    if not pair_top_k_values:
        raise ValueError("--pair-top-k-values must contain at least one value")
    if any(value < 1 for value in pair_top_k_values):
        raise ValueError("--pair-top-k-values must contain positive integers")
    timeout_penalty_multipliers = [
        float(value.strip())
        for value in args.timeout_penalty_multipliers.split(",")
        if value.strip()
    ]
    if not timeout_penalty_multipliers:
        raise ValueError("--timeout-penalty-multipliers must contain at least one value")
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            args.learning_rate_min,
            args.learning_rate_max,
            log=True),
        "pl_margin": trial.suggest_float(
            "pl_margin",
            args.pl_margin_min,
            args.pl_margin_max),
        "dropout_rate": trial.suggest_float(
            "dropout_rate",
            args.dropout_rate_min,
            args.dropout_rate_max),
        "min_pair_runtime_ratio": trial.suggest_float(
            "min_pair_runtime_ratio",
            args.min_pair_runtime_ratio_min,
            args.min_pair_runtime_ratio_max),
        "pair_keep_floor": trial.suggest_float(
            "pair_keep_floor",
            args.pair_keep_floor_min,
            args.pair_keep_floor_max),
        "pair_sampling_strategy": trial.suggest_categorical(
            "pair_sampling_strategy",
            pair_sampling_strategies),
        "timeout_penalty_multiplier": trial.suggest_categorical(
            "timeout_penalty_multiplier",
            timeout_penalty_multipliers),
    }
    if len(pair_top_k_values) == 1:
        params["pair_top_k"] = pair_top_k_values[0]
    else:
        params["pair_top_k"] = trial.suggest_categorical(
            "pair_top_k",
            pair_top_k_values)
    return params


def run_trial(args, trial, params, train_dir, log_file):
    bundle_dir = Path(__file__).resolve().parent.parent
    data_dir = bundle_dir / "data"
    command = [
        args.train_python,
        str(bundle_dir / "src/train_model.py"),
        "--dataset", str(data_dir / "dataset.json"),
        "--instance_features", str(data_dir / "instance_features.csv"),
        "--train_dir", str(train_dir),
        "--hparams", hparam_string(params, args),
        "--max_steps", str(args.max_steps),
        "--eval_every", str(args.eval_every),
        "--min_pair_runtime_ratio", str(params["min_pair_runtime_ratio"]),
        "--pair_sampling_strategy", params["pair_sampling_strategy"],
        "--pair_keep_floor", str(params["pair_keep_floor"]),
        "--pair_sampling_seed", str(trial.number),
        "--pair_top_k", str(params["pair_top_k"]),
        "--timeout_penalty_multiplier", str(params["timeout_penalty_multiplier"]),
        "--validation_timeout_penalty_multiplier", str(args.validation_timeout_penalty_multiplier),
        "--split_strategy", args.split_strategy,
        "--valid_ratio", "0.15",
        "--test_ratio", "0.15",
        "--difficulty_buckets", "3",
        "--timeout_ms", "1200000",
        "--min_class_instances", "7",
        "--small_class_policy", "test",
    ]

    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("AUTOGRAPH_VERBOSITY", "0")
    env.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning,ignore::FutureWarning")
    env.setdefault("MPLCONFIGDIR", "/tmp/rank-matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    best_valid = None
    best_step = None
    test_expected_runtime = ""
    test_loss = ""

    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(bundle_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            match = VALID_RE.search(line)
            if match:
                value = float(match.group("valid_expected_runtime"))
                step = int(match.group("step"))
                if best_valid is None or value < best_valid:
                    best_valid = value
                    best_step = step
                trial.report(value, step)
                if trial.should_prune():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise optuna.TrialPruned()
            match = TEST_RE.search(line)
            if match:
                test_loss = match.group("test_loss")
                test_expected_runtime = match.group("test_expected_runtime")
        return_code = process.wait()

    elapsed = time.time() - start
    status = "ok" if return_code == 0 and best_valid is not None else "error"
    metrics = {
        "status": status,
        "best_valid_expected_runtime": "" if best_valid is None else best_valid,
        "best_step": "" if best_step is None else best_step,
        "test_expected_runtime": test_expected_runtime,
        "test_loss": test_loss,
        "elapsed_sec": "%.1f" % elapsed,
    }
    if status != "ok":
        raise RuntimeError("trial failed; see %s" % log_file)
    return metrics


def main():
    require_optuna()
    args = parse_args()
    if args.hidden_units % args.num_heads != 0:
        raise SystemExit("hidden-units must be divisible by num-heads")

    models_dir = Path(args.models_dir).resolve()
    storage = args.storage or ("sqlite:///" + str(models_dir / "study.db"))
    results_csv = Path(args.results_csv).resolve() if args.results_csv else models_dir / "optuna_results.csv"
    best_json = Path(args.best_json).resolve() if args.best_json else models_dir / "best_config.json"
    models_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=max(args.pruner_warmup_steps, 1))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True)

    def objective(trial):
        params = suggest_params(trial, args)
        train_dir = models_dir / ("trial_%04d" % trial.number)
        log_file = train_dir / "train.log"
        print("\n[trial %d] %s" % (trial.number, hparam_string(params, args)))
        metrics = run_trial(args, trial, params, train_dir, log_file)
        row = {
            "trial_number": trial.number,
            "prop_embedding_size": args.prop_embedding_size,
            "hidden_units": args.hidden_units,
            "num_heads": args.num_heads,
            "num_blocks": args.num_blocks,
            "activation": args.activation,
            "max_gradient_norm": args.max_gradient_norm,
            "train_dir": str(train_dir),
            "log_file": str(log_file),
        }
        row.update(params)
        row.update(metrics)
        append_result(results_csv, row)
        update_best(best_json, row)
        return float(metrics["best_valid_expected_runtime"])

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    print("best_value=%s" % study.best_value)
    print("best_params=%s" % study.best_params)
    print("results=%s" % results_csv)
    print("best_json=%s" % best_json)


if __name__ == "__main__":
    main()
