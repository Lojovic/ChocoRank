# ChocoRank

ChocoRank is a self-contained bundle for learning and using propagator
priority orderings for Choco on XCSP instances.

At runtime, ChocoRank builds the Choco model, extracts the concrete
propagator classes, extracts XCSP instance features, predicts a priority
ordering with Rank, applies those priorities, and solves the instance.
Inference is NumPy-only; TensorFlow is needed only for training.

## Requirements

Runtime solving:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy
export RANK_PYTHON="$PWD/.venv/bin/python"
```

Training and hyperparameter search:

```bash
python3 -m pip install numpy tensorflow optuna
```

## Main Files

```text
choco/dist/choco-rank.jar        Java runner used for probing, solving, and evaluation
feature_extractor/xcsp2feat         XCSP feature extractor
src/                                Python runtime and training code
data/dataset.json                   Runtime data from random propagator orderings
data/instance_features.csv          XCSP feature rows
models/                             Packaged trained models
training_instances.txt              Instance basenames used for the bundled data
```

The original XML files used to build the bundled data are not included.
They were downloaded from the
[XCSP3 instances page](https://xcsp.org/instances/) using the full series
download list shown there:

```bash
wget https://xcsp.org/assets/all-xcsp3-series.txt
wget -i all-xcsp3-instances.txt
```

Only a filtered subset of those downloaded instances was used for the
training/evaluation data.

## Scripts

`scripts/run_default.sh`
Solves one XCSP instance with normal Choco priorities and prints only the XCSP
solution.

```bash
./scripts/run_default.sh /path/to/instance.xml
```

`scripts/run_model.sh`
Solves one XCSP instance with a trained Rank model and prints only the XCSP
solution. It uses gated mode; if the model has no `class_policy.json`, gated
mode behaves like ordinary model mode.

```bash
./scripts/run_model.sh /path/to/instance.xml models/new_threshold/model
```

`scripts/extract_instance_features.py`
Recursively extracts `xcsp2feat` features for a directory of `.xml` instances
and writes `instance_features.csv`.

```bash
python3 scripts/extract_instance_features.py \
  --instances-dir /path/to/instances \
  --output data/instance_features.csv
```

`scripts/collect_random_ordering_data.py`
Probes each instance for concrete propagator classes, runs random priority
orderings, and writes training data in the same format as
`data/dataset.json`. Timed-out solves are stored with
`runtime_ms = timeout`.

```bash
python3 scripts/collect_random_ordering_data.py \
  --instances-dir /path/to/instances \
  --output data/dataset.json \
  --orderings 20 \
  --timeout-sec 1200 \
  --seed 0
```

`scripts/grid_search.py`
Runs a fixed architecture grid search over `src/train_model.py`.
It writes trial directories, a CSV summary, and `best_config.json`.

```bash
python3 scripts/grid_search.py \
  --models-dir models/grid_search \
  --max-steps 5000 \
  --eval-every 100
```

`scripts/optuna_search.py`
Runs Optuna over training/loss/pair-mining parameters while keeping the
architecture values supplied on the command line.

```bash
python3 scripts/optuna_search.py \
  --models-dir models/optuna_search \
  --n-trials 20 \
  --max-steps 5000 \
  --eval-every 100 \
  --prop-embedding-size 64 \
  --hidden-units 256 \
  --num-heads 2 \
  --num-blocks 3
```

`scripts/evaluate_test_walltimes.py`
Runs default Choco and model Choco on a saved split from a model directory.
Each worker evaluates both default and model for one instance. The only
persistent output is the walltime CSV.

```bash
python3 scripts/evaluate_test_walltimes.py \
  --train-dir models/new_threshold/model \
  --instances-dir /path/to/instances \
  --split test \
  --jobs 8 \
  --output-dir test_results/new_threshold
```

It writes:

```text
<output-dir>/<split>_walltimes.csv
```

`scripts/build_validation_class_policy.py`
Evaluates validation instances and writes a per-problem-class policy deciding
whether runtime should use the model or default Choco for each class. The policy
uses PAR walltime aggregation. The only persistent output is
`class_policy.json`; temporary validation walltimes are discarded unless an
existing CSV is supplied with `--walltimes-csv`.

```bash
python3 scripts/build_validation_class_policy.py \
  --train-dir models/new_threshold/model \
  --instances-dir /path/to/instances \
  --jobs 8 \
  --par 1 \
  --min-improvement 0.05
```

By default it writes:

```text
models/new_threshold/model/class_policy.json
```

`scripts/filter_supported_test_instances.py`
Probes split instances and removes instances whose propagator classes are not
all present in a model's `metadata.json`. Use `--update-split` only when you
want to rewrite the model split files.

```bash
python3 scripts/filter_supported_test_instances.py \
  --train-dir models/new_threshold/model \
  --split test \
  --instances-dir /path/to/instances \
  --update-split
```

## Start To Finish

### 1. Prepare An Instance Directory

Put the XCSP files you want to use in one directory. The scripts search
recursively for `.xml` files, so nested benchmark folders are fine:

```text
/path/to/instances/
  AllInterval-035.xml
  subdir/SomeOtherInstance.xml
```

### 2. Extract Features

```bash
python3 scripts/extract_instance_features.py \
  --instances-dir /path/to/instances \
  --output data/instance_features.csv
```

This creates one feature row per instance. The first column is the instance
basename, so the runtime data and feature data must use matching basenames.

### 3. Collect Runtime Data

Run each instance with random propagator priority orderings:

```bash
python3 scripts/collect_random_ordering_data.py \
  --instances-dir /path/to/instances \
  --output data/dataset.json \
  --orderings 20 \
  --timeout-sec 1200 \
  --seed 0
```

For a large benchmark set this is the expensive step. `--orderings 20` matches
the usual experiment setup; 

### 4. Train A Model

Train directly with the bundled training entry point:

```bash
python3 src/train_model.py \
  --dataset data/dataset.json \
  --instance_features data/instance_features.csv \
  --train_dir models/my_model \
  --hparams "learning_rate=0.0005,pl_margin=0.5,hidden_units=256,num_blocks=2,prop_embedding_size=64,num_heads=4,dropout_rate=0.2" \
  --max_steps 2000 \
  --eval_every 100 \
  --min_pair_runtime_ratio 0.3 \
  --pair_sampling_strategy weighted \
  --pair_keep_floor 0.1 \
  --split_strategy within_class_stratified \
  --valid_ratio 0.15 \
  --test_ratio 0.15 \
  --difficulty_buckets 3 \
  --timeout_ms 1200000 \
  --min_class_instances 7 \
  --small_class_policy test
```

Training writes:

```text
models/my_model/metadata.json
models/my_model/numpy_weights.npz
models/my_model/split_instances.json
models/my_model/split_instances.csv
models/my_model/CspRank.ckpt-*
```

The runtime solver needs `metadata.json` and `numpy_weights.npz`. The split
files are used by evaluation and validation-policy scripts.

### 5. Optionally Run Hyperparameter Search

Use Optuna when you want to tune training parameters:

```bash
python3 scripts/optuna_search.py \
  --models-dir models/optuna_search \
  --n-trials 20 \
  --max-steps 5000 \
  --eval-every 100 \
  --prop-embedding-size 64 \
  --hidden-units 256 \
  --num-heads 2 \
  --num-blocks 3 \
  --pair-sampling-strategies weighted,threshold,all_pairs,topk_bottomk \
  --timeout-penalty-multipliers 1,2,5,10 \
  --split-strategy within_class_stratified
```

The validation objective is `valid_expected_runtime`, where lower is better.
Each trial directory is a normal model directory if it contains
`metadata.json` and `numpy_weights.npz`.

### 6. Evaluate A Model On A Split

Evaluate the model on the test split that was created during training:

```bash
python3 scripts/evaluate_test_walltimes.py \
  --train-dir models/my_model \
  --instances-dir /path/to/instances \
  --split test \
  --jobs 8 \
  --output-dir test_results/my_model
```

Before launching a long run, you can use `--dry-run` to confirm that instance names
resolve correctly:

```bash
python3 scripts/evaluate_test_walltimes.py \
  --train-dir models/my_model \
  --instances-dir /path/to/instances \
  --split test \
  --dry-run
```

### 7. Optionally Build A Gated Policy

If validation shows that some problem classes are better with default Choco,
build a class policy:

```bash
python3 scripts/build_validation_class_policy.py \
  --train-dir models/my_model \
  --instances-dir /path/to/instances \
  --jobs 8 \
  --par 1 \
  --min-improvement 0.05 \
  --unseen-class-decision model
```

After this, `scripts/run_model.sh` automatically uses the policy. If no policy
file exists, it uses the model for all instances.

### 8. Solve A New Instance

Default Choco:

```bash
./scripts/run_default.sh /path/to/new_instance.xml
```

Rank model:

```bash
./scripts/run_model.sh /path/to/new_instance.xml models/my_model
```

The instance may be outside the training/evaluation directory. It only needs to
be an XCSP XML instance whose propagator classes are supported by the model.

## Direct Java Commands

All wrapper scripts call the unified runner:

```text
choco/dist/choco-rank.jar
```

Useful direct commands:

```bash
java -jar choco/dist/choco-rank.jar --probe <probe-json> <instance.xml>
java -jar choco/dist/choco-rank.jar --run-default <result-json> <instance.xml>
java -jar choco/dist/choco-rank.jar --run <result-json> <priorities-json> <instance.xml>
java -jar choco/dist/choco-rank.jar --run-model <result-json> <project-root> <train-dir> <xcsp2feat> <instance.xml>
java -jar choco/dist/choco-rank.jar --run-gated-model <result-json> <project-root> <train-dir> <xcsp2feat> <instance.xml> [class-policy.json]
java -jar choco/dist/choco-rank.jar --solve-default <instance.xml>
java -jar choco/dist/choco-rank.jar --solve-model <project-root> <train-dir> <xcsp2feat> <instance.xml>
java -jar choco/dist/choco-rank.jar --solve-gated-model <project-root> <train-dir> <xcsp2feat> <instance.xml> [class-policy.json]
```

Use the shell scripts for ordinary solving; use direct Java commands when you
need result JSONs, probing, or custom integration.

## Packaged Models

The bundle currently includes these trained model directories:

```text
models/all_pairs/model
models/top_k_bottom_k/model
models/stratified/model
models/gated_threshold/model
models/weighted/model
```

Each model directory contains runtime metadata, NumPy weights, and split files.
Experiment metadata such as `best_config.json`, `optuna_results.csv`, and
`study.db` is stored next to each model family.

## Split Strategies

Training supports:

```text
random
within_class_stratified
within_class_bucket_stratified
```

`within_class_stratified` sorts instances inside each problem class by
difficulty, puts easier instances in training, medium instances in validation,
and harder instances in test. `within_class_bucket_stratified` keeps easy,
medium, and hard buckets represented in each split. `random` ignores problem
class difficulty.

## Notes

`src/train_model.py` is the training entry point.
`src/choco_priority_pipeline.py` is the runtime bridge called from Java. The
project does not depend on a parent Rank checkout.
