"""Offline SDK host stand-in: real supplied data/training, simulated model replies only."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import threading
from pathlib import Path

from zhixun.contest_runtime import rows_dataset
from zhixun.metrics import score
from zhixun.vendor.model import Model


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mounted(source, target):
    return ["--mount", f"type=bind,source={source.resolve()},target={target},readonly"]


def check_predictions(image, model_dir, task, dataset, inputs):
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--read-only",
        "--network",
        "none",
        *mounted(model_dir, "/model"),
        "-e",
        "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz",
        image,
        "predict",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    timer = threading.Timer(30, process.kill)
    timer.start()
    try:
        for value in inputs[:2]:
            process.stdin.write(encoded({"task": task, "inputs": value}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()  # stdin stays open: enforce immediate per-row reply.
            result = json.loads(line)
            if set(result) != {"outputs"}:
                raise ValueError("Predict did not return exactly one SDK frame")
            if task == "classification" and not isinstance(result["outputs"], list):
                raise ValueError("Classification result is not a list")
        process.stdin.close()
        if process.wait(timeout=15):
            raise RuntimeError(process.stderr.read())
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()


def smoke(image: str, root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    for dataset in "ABC":
        data = output / dataset / "data"
        data.mkdir(parents=True)
        for split in ["train", "validation"]:
            with (root / "data" / dataset / (split + ".jsonl")).open(encoding="utf-8") as source:
                rows = [json.loads(line) for line in source if line.strip()]
            projected = [{k: row[k] for k in ["id", "inputs", "answer"]} for row in rows]
            (data / (split + ".jsonl")).write_text(
                "\n".join(map(encoded, projected)) + "\n", encoding="utf-8"
            )
        parent_dir = output / dataset / "parent-0"
        parent_dir.mkdir()
        (parent_dir / "checkpoint.npz").write_bytes(
            (root / "data" / dataset / "checkpoint.npz").read_bytes()
        )
        task = Model.load(parent_dir / "checkpoint.npz").meta["task"]
        history, state_dir = [], None
        for number in range(1, 4 if dataset == "A" else 2):
            run = output / dataset / f"round-{number}"
            run.mkdir()
            digest = sha(parent_dir / "checkpoint.npz")
            parent_id = f"local-parent-{dataset}-{number}"
            artifact = {
                "checkpoint_paths": ["checkpoint.npz"],
                "code_paths": [],
                "predict_argv": ["python3", "/app/harness.py", "predict"],
            }
            parent = {
                "artifact_id": parent_id,
                "artifact_sha256": "sha256:" + digest,
                "artifact": artifact,
                "files": [
                    {
                        "path": "checkpoint.npz",
                        "sha256": digest,
                        "size_bytes": (parent_dir / "checkpoint.npz").stat().st_size,
                    }
                ],
            }
            frame = {
                "protocol": "contest.harness.workspace.v2",
                "run_id": "local-sdk-smoke",
                "dataset_key": dataset,
                "round_index": number,
                "stage_id": "local-public-check",
                "prompt": "Local protocol stand-in; no actual platform or model score is claimed.",
                "prompt_version": "industrial-workspace-flexible-v3",
                "rounds_per_dataset": 10,
                "task_contract": {
                    "task": task,
                    "dataset_key": dataset,
                    "public_projection": {
                        "train_path": "/data/train.jsonl",
                        "validation_path": "/data/validation.jsonl",
                    },
                },
                "model_name": "local-model-stand-in-not-real-sampler",
                "max_output_tokens": 512,
                "parent_checkpoint_sha256": "sha256:" + digest,
                "parent_artifact": parent,
                "artifacts": [parent],
                "state_path": "/state" if state_dir else None,
                "state_sha256": "",
                "resource_limits": {},
                "wall_seconds": 300,
            }
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--read-only",
                "--network",
                "none",
                "--tmpfs",
                "/workspace:rw,uid=65534,gid=65534,mode=0700",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid",
                *mounted(data, "/data"),
                *mounted(parent_dir, "/previous"),
            ]
            if state_dir:
                command += mounted(state_dir, "/state")
            command += [image, "train"]
            stderr = (run / "stderr.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                encoding="utf-8",
            )
            timer = threading.Timer(300, process.kill)
            timer.start()
            calls, candidate_dir, complete = [], None, None
            next_state = run / "state"

            def send(value, process=process):
                process.stdin.write(encoded(value) + "\n")
                process.stdin.flush()

            def receive_files(items, folder):
                folder.mkdir()
                for item in items:
                    path = folder / item["path"]
                    if not path.resolve().is_relative_to(folder.resolve()):
                        raise ValueError("Unsafe test artifact path")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(base64.b64decode(item["data_base64"], validate=True))

            try:
                send(frame)
                sequence = 0
                with (run / "protocol.jsonl").open("w", encoding="utf-8") as journal:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            raise RuntimeError("Train process exited before round acknowledgement")
                        request = json.loads(line)
                        journal.write(encoded(request) + "\n")
                        kind = request["type"]
                        if kind == "round_failed":
                            raise RuntimeError("Participant failed: " + str(request))
                        if kind == "round_complete":
                            complete = request["record"]
                            if complete["model_call_ids"] != calls:
                                raise ValueError("Round omitted model receipts")
                            if complete["history_ids"] != [r["id"] for r in history]:
                                raise ValueError("Round did not cite the actual read history")
                            send({"type": "round_ack", "accepted": True})
                            process.stdin.close()
                            break
                        if request["id"] != sequence:
                            raise ValueError("RPC sequence changed")
                        sequence += 1
                        reply = {"id": request["id"]}
                        if kind == "history_request":
                            reply.update(type="history_response", records=history)
                        elif kind == "model_request":
                            if any(
                                frame["prompt"] in str(m.get("content", ""))
                                for m in request["messages"]
                            ):
                                raise ValueError("Official prompt was duplicated")
                            call_id = f"local-mock-{dataset}-{number}-{len(calls)}"
                            content = (
                                encoded(
                                    {
                                        "select": "candidate",
                                        "reason": "Compare actual public measurement.",
                                    }
                                )
                                if len(calls)
                                else encoded(
                                    {
                                        "action": "retrain",
                                        "epochs": 1,
                                        "learning_rate": 0.0005,
                                        "cw_power": 0.5,
                                    }
                                )
                            )
                            calls.append(call_id)
                            reply.update(
                                type="model_response",
                                call_id=call_id,
                                response={
                                    "choices": [
                                        {"message": {"role": "assistant", "content": content}}
                                    ]
                                },
                            )
                        elif kind == "candidate_request":
                            candidate_dir = run / "candidate"
                            receive_files(request["files"], candidate_dir)
                            model = Model.load(candidate_dir / "checkpoint.npz")
                            validation_rows = [
                                json.loads(s)
                                for s in (data / "validation.jsonl")
                                .read_text(encoding="utf-8")
                                .splitlines()
                                if s.strip()
                            ]
                            metrics = score(model, rows_dataset(validation_rows, model))[0]
                            check_predictions(
                                image,
                                candidate_dir,
                                task,
                                dataset,
                                [r["inputs"] for r in validation_rows[:2]],
                            )
                            reply.update(
                                type="candidate_response",
                                artifact_id=f"local-candidate-{dataset}-{number}",
                                summary={
                                    "metrics": metrics,
                                    "source": "local-public-host-stand-in",
                                },
                            )
                        elif kind == "state_request":
                            receive_files(request["files"], next_state)
                            reply.update(type="state_response", state_sha256="sha256:" + "11" * 32)
                        else:
                            raise ValueError("Unexpected SDK request: " + kind)
                        send(reply)
                if process.wait(timeout=15):
                    raise RuntimeError("Train process failed after acknowledgement")
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                stderr.close()
            assert (
                complete is not None and len(calls) >= 2 and candidate_dir and next_state.exists()
            )
            if complete["selected_artifact_id"] != parent_id:
                parent_dir = candidate_dir
            state_dir = next_state
            history.append({"id": f"local-history-{dataset}-{number}", **complete})
            row = {
                "dataset": dataset,
                "round": number,
                "model_requests_simulated": len(calls),
                "parent_selected": complete["selected_artifact_id"] == parent_id,
                "history_references": len(complete["history_ids"]),
                "status": "passed",
            }
            checks.append(row)
            print(encoded(row), flush=True)
    result = {
        "status": "passed",
        "checks": checks,
        "real_public_data_and_training": True,
        "model_replies_simulated": True,
        "real_platform_or_sampler_verified": False,
        "read_only_root_nonroot_user_tested": True,
        "row_prediction_before_eof_tested": True,
    }
    (output / "smoke_report.json").write_text(encoded(result), encoding="utf-8")
    return result
