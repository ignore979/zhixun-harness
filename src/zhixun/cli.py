from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from zhixun.data import ROOT, verify_inputs


def doctor() -> dict:
    packages = {}
    for name in ["numpy", "swanlab", "pytrio"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not_installed"
    locks = ["requirements.txt"]
    if packages["pytrio"] != "not_installed":
        locks.append("requirements-pytrio.txt")
    try:
        version("matplotlib")
        locks.append("requirements-plots.txt")
    except PackageNotFoundError:
        pass
    pinned = {}
    for name in locks:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            if "==" not in line or line.lstrip().startswith("#"):
                continue
            package, expected = line.strip().split("==")
            try:
                installed = version(package)
            except PackageNotFoundError as exc:
                raise ValueError(f"Locked dependency is missing: {package}") from exc
            if installed != expected:
                raise ValueError(
                    f"Dependency drift: {package} expected {expected}, got {installed}"
                )
            pinned[package] = expected
    checked = {profile: verify_inputs(profile)["archive_sha256"] for profile in "ABC"}
    return {
        "status": "local_inputs_verified", "python": platform.python_version(),
        "packages": packages, "archive_sha256": checked,
        "dependency_pins_verified": len(pinned),
        "credential_environment_present": {
            "swanlab": bool(os.environ.get("SWANLAB_API_KEY")),
            "pytrio": bool(os.environ.get("PYTRIO_API_KEY")),
        },
        "organizer_authenticity": "download_received; organizer_signature_not_supplied",
        "formal_hidden_evaluation": "not_integrated",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="zhixun")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check environment and pinned input hashes")
    diagnosis = commands.add_parser("diagnose", help="read-only saturation and error diagnostics")
    diagnosis.add_argument("--run", type=Path, required=True)
    targeted = commands.add_parser("adaptation-study", help="prediction-only temporal ablations")
    targeted.add_argument("--mode", choices=["offline", "online"], default="offline")
    targeted.add_argument("--project", default="zhixun-school")
    ensemble = commands.add_parser("ensemble-study", help="temporal ensemble and CORAL search")
    ensemble.add_argument("--mode", choices=["offline", "online"], default="offline")
    ensemble.add_argument("--project", default="zhixun-school")
    domain = commands.add_parser("domain-study", help="separated domain alignment follow-up")
    domain.add_argument("--mode", choices=["offline", "online"], default="offline")
    domain.add_argument("--project", default="zhixun-school")
    selector = commands.add_parser("selector-study", help="causal domain-view selection follow-up")
    selector.add_argument("--mode", choices=["offline", "online"], default="offline")
    selector.add_argument("--project", default="zhixun-school")
    guarded = commands.add_parser("prepare-guarded", help="same-state accuracy-floor policy labels")
    guarded.add_argument("--source", type=Path, required=True)
    guarded.add_argument("--mode", choices=["offline", "online"], default="offline")
    guarded.add_argument("--extra-run", action="append", type=Path, default=[])
    guarded.add_argument("--destination", default="harness-guarded-v4",
                         choices=["harness-guarded-v4", "harness-guarded-v4-expanded"])
    oracle = commands.add_parser("prepare-oracle",
                                  help="offline result-verified finite-action corpus")
    oracle.add_argument("--mode", choices=["offline", "online"], default="offline")
    fair = commands.add_parser("fair-replay", help="equal epoch-cap rule/LLM end-to-end replay")
    fair.add_argument("--strategy", choices=["flat", "soft", "agent_base", "agent_trained"],
                      required=True)
    fair.add_argument("--training-run", type=Path)
    fair.add_argument("--resume", type=Path)
    fair.add_argument("--mode", choices=["offline", "online"], default="offline")
    fair.add_argument("--config", type=Path)
    fair.add_argument("--project", default="zhixun-school")
    fair.add_argument("--prediction-config", type=Path)
    training = commands.add_parser("train", help="continue the original CPU checkpoint")
    training.add_argument("--task", choices=list("ABC"), required=True)
    training.add_argument("--epochs", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--mode", choices=["offline", "online"], default="offline")
    training.add_argument("--project", default="zhixun-school")
    training.add_argument("--workspace")
    training.add_argument("--parent", type=Path)
    streaming = commands.add_parser("stream", help="run the public-validation drift loop")
    streaming.add_argument("--task", choices=list("ABC"), required=True)
    streaming.add_argument("--mode", choices=["offline", "online"], default="offline")
    streaming.add_argument("--project", default="zhixun-school")
    streaming.add_argument("--workspace")
    streaming.add_argument("--resume", type=Path, help="existing stream experiment directory")
    streaming.add_argument("--config", type=Path)
    replay = commands.add_parser("replay", help="UCI labeled replay, separate from official splits")
    replay.add_argument("--strategy", choices=["static", "always", "performance"], required=True)
    replay.add_argument("--mode", choices=["offline", "online"], default="offline")
    replay.add_argument("--project", default="zhixun-school")
    replay.add_argument("--workspace")
    replay.add_argument("--config", type=Path)
    replay.add_argument("--resume", type=Path)
    capacity = commands.add_parser("capacity-train", help="controlled public B width/epoch study")
    capacity.add_argument("--hidden", type=int, choices=[24, 96], default=96)
    capacity.add_argument("--epochs", type=int, default=5)
    capacity.add_argument("--learning-rate", type=float, default=0.001)
    capacity.add_argument("--lr-decay", type=float, default=1.0)
    capacity.add_argument("--mode", choices=["offline", "online"], default="offline")
    capacity.add_argument("--project", default="zhixun-school")
    capacity.add_argument("--workspace")
    corpus = commands.add_parser("prepare-harness", help="derive user-approved real policy traces")
    corpus.add_argument("--run", action="append", type=Path, required=True)
    prediction = commands.add_parser("predict", help="predict from JSONL stdin")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    batch_prediction = commands.add_parser("predict-batch",
                                           help="experimental label-free CORAL batch")
    batch_prediction.add_argument("--checkpoint", type=Path, required=True)
    batch_prediction.add_argument("--adapter", type=Path, required=True)
    commands.add_parser("package", help="build a source/evidence archive excluding guidance")
    commands.add_parser("pytrio-models", help="query supported models from a logged-in account")
    cloud = commands.add_parser("verify-cloud", help="verify saved SwanLab runs read-only")
    cloud.add_argument(
        "--url", action="append", help="synced experiment URL; repeat for multiple runs"
    )
    evidence = commands.add_parser("record-cloud", help="record a manually reviewed cloud run link")
    evidence.add_argument("--run", required=True)
    evidence.add_argument("--url", required=True)
    evidence.add_argument("--reviewer", required=True)
    sft = commands.add_parser("pytrio-train", help="train on explicitly supplied approved LLM data")
    sft.add_argument("--data", type=Path, required=True)
    sft.add_argument("--provenance", type=Path, required=True)
    sft.add_argument("--model", required=True)
    sft.add_argument("--steps", type=int)
    sft.add_argument("--epochs", type=int, default=3)
    sft.add_argument("--batch-size", type=int, default=3)
    sft.add_argument("--evaluation", type=Path)
    sft.add_argument("--learning-rate", type=float, default=0.0001)
    sft.add_argument("--project", default="zhixun-school")
    sft.add_argument("--workspace")
    evaluation = commands.add_parser("pytrio-evaluate",
                                     help="compare base and saved policy on holdout")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--project", default="zhixun-school")
    evaluation.add_argument("--workspace")
    rescore = commands.add_parser("rescore-harness", help="decode and rescore saved model outputs")
    rescore.add_argument("--run", type=Path, required=True)
    rescore.add_argument("--project", default="zhixun-school")
    rescore.add_argument("--workspace")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            output = doctor()
        elif args.command == "diagnose":
            from zhixun.diagnosis import diagnose_run

            output = diagnose_run(args.run)
        elif args.command == "adaptation-study":
            from zhixun.adaptation_study import run_adaptation_study

            output = run_adaptation_study(mode=args.mode, project=args.project)
        elif args.command == "ensemble-study":
            from zhixun.ensemble_study import run_ensemble_study

            output = run_ensemble_study(mode=args.mode, project=args.project)
        elif args.command == "domain-study":
            from zhixun.domain_study import run_domain_study

            output = run_domain_study(mode=args.mode, project=args.project)
        elif args.command == "selector-study":
            from zhixun.selector_study import run_selector_study

            output = run_selector_study(mode=args.mode, project=args.project)
        elif args.command == "prepare-guarded":
            from zhixun.guarded_corpus import prepare_guarded_corpus

            output = prepare_guarded_corpus(args.source, mode=args.mode,
                                            extra_runs=args.extra_run,
                                            destination_name=args.destination)
        elif args.command == "prepare-oracle":
            from zhixun.oracle_data import prepare_oracle

            output = prepare_oracle(mode=args.mode)
        elif args.command == "fair-replay":
            from zhixun.fair_replay import run_fair

            output = run_fair(args.strategy, training_run=args.training_run,
                              resume=args.resume, mode=args.mode, config_path=args.config,
                              project=args.project, prediction_config=args.prediction_config)
        elif args.command == "train":
            from zhixun.train import RoundPlan, run_round

            config = json.loads((ROOT / "configs/round1.json").read_text(encoding="utf-8"))
            plan = RoundPlan(
                args.task, args.epochs if args.epochs is not None else config["epochs"],
                args.learning_rate if args.learning_rate is not None else config["learning_rate"],
                config["max_seconds"],
            )
            output = run_round(plan, mode=args.mode, project=args.project,
                               workspace=args.workspace, parent=args.parent)
        elif args.command == "stream":
            from zhixun.loop import run_stream

            output = run_stream(args.task, mode=args.mode, project=args.project,
                                workspace=args.workspace, resume=args.resume,
                                config_path=args.config)
        elif args.command == "replay":
            from zhixun.replay import run_replay

            output = run_replay(args.strategy, mode=args.mode, project=args.project,
                                workspace=args.workspace, config_path=args.config,
                                resume=args.resume)
        elif args.command == "capacity-train":
            from zhixun.efficiency import run_capacity

            output = run_capacity(hidden=args.hidden, epochs=args.epochs,
                                   learning_rate=args.learning_rate, decay=args.lr_decay,
                                   mode=args.mode, project=args.project, workspace=args.workspace)
        elif args.command == "prepare-harness":
            from zhixun.harness_data import prepare_harness

            output = prepare_harness(args.run)
        elif args.command in {"pytrio-evaluate", "rescore-harness"}:
            from zhixun.harness_evaluation import evaluate_harness, rescore_harness

            operation = evaluate_harness if args.command == "pytrio-evaluate" else rescore_harness
            output = operation(args.run, project=args.project, workspace=args.workspace)
        elif args.command in {"predict", "predict-batch"}:
            from zhixun.predict import predict_batch, predict_stream

            if args.command == "predict-batch":
                predict_batch(args.checkpoint, args.adapter, sys.stdin, sys.stdout)
            else:
                predict_stream(args.checkpoint, sys.stdin, sys.stdout)
            return 0
        elif args.command == "package":
            from zhixun.packaging import build_package

            output = build_package()
        elif args.command == "verify-cloud":
            from zhixun.cloud import verify_cloud_runs

            output = verify_cloud_runs(args.url)
        elif args.command == "record-cloud":
            from zhixun.evidence import record_cloud

            output = record_cloud(args.run, args.url, args.reviewer)
        else:
            from zhixun.pytrio_training import list_models, train_sft

            output = list_models() if args.command == "pytrio-models" else train_sft(
                args.data, args.provenance, args.model, args.steps, args.learning_rate,
                project=args.project, workspace=args.workspace,
                epochs=args.epochs, batch_size=args.batch_size, evaluation_path=args.evaluation,
            )
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, ImportError, RuntimeError, TimeoutError) as exc:
        print(f"zhixun: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
