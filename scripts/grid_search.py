#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


LEARNING_RATES = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]
PROP_EMBEDDING_SIZES = [32, 64, 128]
HIDDEN_UNITS = [64, 128, 256]
NUM_HEADS = [2, 4, 8]
NUM_BLOCKS = [1, 2, 3]

VALID_RE = re.compile(
    r"step\s+(?P<step>\d+).*?"
    r"valid_expected_runtime\s+(?P<valid_expected_runtime>[0-9.]+)")
TEST_RE = re.compile(
    r"test_loss\s+(?P<test_loss>[0-9.]+).*?"
    r"test_expected_runtime\s+(?P<test_expected_runtime>[0-9.]+)")


def parse_args():
    bundle_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Grid-search CSP Rank hyperparameters.")
    parser.add_argument("--models-dir", default=str(bundle_dir / "models/grid_search"))
    parser.add_argument("--results-csv", default=None,
                        help="Defaults to <models-dir>/grid_results.csv.")
    parser.add_argument("--best-json", default=None,
                        help="Defaults to <models-dir>/best_config.json.")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--pl-margin", type=float, default=0.5)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--min-pair-runtime-ratio", type=float, default=0.3)
    parser.add_argument("--pair-sampling-strategy", default="weighted",
                        choices=[
                            "weighted",
                            "threshold",
                            "all_pairs",
                            "topk_bottomk",
                        ])
    parser.add_argument("--pair-keep-floor", type=float, default=0.1)
    parser.add_argument("--pair-top-k", type=int, default=3)
    parser.add_argument("--split-strategy", default="within_class_stratified",
                        choices=[
                            "random",
                            "within_class_stratified",
                            "within_class_bucket_stratified",
                        ])
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N pending configurations.")
    parser.add_argument("--force", action="store_true",
                        help="Rerun configurations already present in the results CSV.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_name(config):
    lr_text = ("%g" % config["learning_rate"]).replace(".", "p")
    return (
        "lr%s_emb%d_hid%d_heads%d_blocks%d" %
        (lr_text, config["prop_embedding_size"], config["hidden_units"],
         config["num_heads"], config["num_blocks"]))


def hparam_string(config, pl_margin, dropout_rate):
    values = {
        "learning_rate": config["learning_rate"],
        "pl_margin": pl_margin,
        "hidden_units": config["hidden_units"],
        "num_blocks": config["num_blocks"],
        "prop_embedding_size": config["prop_embedding_size"],
        "num_heads": config["num_heads"],
        "dropout_rate": dropout_rate,
    }
    return ",".join("%s=%s" % (key, values[key]) for key in [
        "learning_rate",
        "pl_margin",
        "hidden_units",
        "num_blocks",
        "prop_embedding_size",
        "num_heads",
        "dropout_rate",
    ])


def iter_configs():
    for learning_rate, prop_embedding_size, hidden_units, num_heads, num_blocks in itertools.product(
            LEARNING_RATES, PROP_EMBEDDING_SIZES, HIDDEN_UNITS, NUM_HEADS, NUM_BLOCKS):
        if hidden_units % num_heads != 0:
            continue
        yield {
            "learning_rate": learning_rate,
            "prop_embedding_size": prop_embedding_size,
            "hidden_units": hidden_units,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
        }


def read_done(results_csv):
    if not results_csv.is_file():
        return set(), []
    rows = []
    with open(results_csv, newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            rows.append(row)
    return set(row["config_name"] for row in rows if row.get("status") == "ok"), rows


def append_result(results_csv, row):
    first_write = not results_csv.is_file()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_name",
        "status",
        "best_valid_expected_runtime",
        "best_step",
        "test_expected_runtime",
        "test_loss",
        "elapsed_sec",
        "learning_rate",
        "prop_embedding_size",
        "hidden_units",
        "num_heads",
        "num_blocks",
        "pl_margin",
        "dropout_rate",
        "min_pair_runtime_ratio",
        "pair_sampling_strategy",
        "pair_keep_floor",
        "pair_top_k",
        "train_dir",
        "log_file",
    ]
    with open(results_csv, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()
        writer.writerow(row)


def update_best(best_json, rows):
    ok_rows = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = row.get("best_valid_expected_runtime")
        if value in (None, ""):
            continue
        try:
            row = dict(row)
            row["best_valid_expected_runtime"] = float(value)
            ok_rows.append(row)
        except ValueError:
            pass
    if not ok_rows:
        return
    best = min(ok_rows, key=lambda row: row["best_valid_expected_runtime"])
    best_json.parent.mkdir(parents=True, exist_ok=True)
    with open(best_json, "w") as fout:
        json.dump(best, fout, indent=2, sort_keys=True)


def run_config(args, config, train_dir, log_file):
    bundle_dir = Path(__file__).resolve().parent.parent
    data_dir = bundle_dir / "data"
    command = [
        sys.executable,
        str(bundle_dir / "src/train_model.py"),
        "--dataset", str(data_dir / "dataset.json"),
        "--instance_features", str(data_dir / "instance_features.csv"),
        "--train_dir", str(train_dir),
        "--hparams", hparam_string(config, args.pl_margin, args.dropout_rate),
        "--max_steps", str(args.max_steps),
        "--eval_every", str(args.eval_every),
        "--min_pair_runtime_ratio", str(args.min_pair_runtime_ratio),
        "--pair_sampling_strategy", args.pair_sampling_strategy,
        "--pair_keep_floor", str(args.pair_keep_floor),
        "--pair_top_k", str(args.pair_top_k),
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
            match = TEST_RE.search(line)
            if match:
                test_loss = match.group("test_loss")
                test_expected_runtime = match.group("test_expected_runtime")
        return_code = process.wait()

    elapsed = time.time() - start
    status = "ok" if return_code == 0 and best_valid is not None else "error"
    return {
        "status": status,
        "best_valid_expected_runtime": "" if best_valid is None else best_valid,
        "best_step": "" if best_step is None else best_step,
        "test_expected_runtime": test_expected_runtime,
        "test_loss": test_loss,
        "elapsed_sec": "%.1f" % elapsed,
    }


def main():
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()
    results_csv = Path(args.results_csv).resolve() if args.results_csv else models_dir / "grid_results.csv"
    best_json = Path(args.best_json).resolve() if args.best_json else models_dir / "best_config.json"

    done, existing_rows = read_done(results_csv)
    configs = list(iter_configs())
    pending = []
    for config in configs:
        name = config_name(config)
        if not args.force and name in done:
            continue
        pending.append((name, config))

    if args.limit is not None:
        pending = pending[:args.limit]

    print("total_configs=%d pending=%d results=%s" %
          (len(configs), len(pending), results_csv))
    if args.dry_run:
        for name, config in pending:
            print(name, hparam_string(config, args.pl_margin, args.dropout_rate))
        return

    rows = list(existing_rows)
    for index, (name, config) in enumerate(pending, start=1):
        train_dir = models_dir / name
        log_file = train_dir / "train.log"
        print("\n[%d/%d] %s" % (index, len(pending), name))
        metrics = run_config(args, config, train_dir, log_file)
        row = {
            "config_name": name,
            "pl_margin": args.pl_margin,
            "dropout_rate": args.dropout_rate,
            "min_pair_runtime_ratio": args.min_pair_runtime_ratio,
            "pair_sampling_strategy": args.pair_sampling_strategy,
            "pair_keep_floor": args.pair_keep_floor,
            "pair_top_k": args.pair_top_k,
            "train_dir": str(train_dir),
            "log_file": str(log_file),
        }
        row.update(config)
        row.update(metrics)
        append_result(results_csv, row)
        rows.append(dict((key, str(value)) for key, value in row.items()))
        update_best(best_json, rows)
        print("result %s best_valid_expected_runtime=%s best_step=%s" %
              (metrics["status"], metrics["best_valid_expected_runtime"], metrics["best_step"]))


if __name__ == "__main__":
    main()
