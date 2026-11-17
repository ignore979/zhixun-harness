from __future__ import annotations

import json
import math
from pathlib import Path


class Recorder:
    """Actual SwanLab recording plus an append-only local scalar journal."""

    def __init__(
        self, folder: Path, config: dict, *, mode: str, project: str, workspace: str | None,
    ) -> None:
        if mode not in {"offline", "online"}:
            raise ValueError("Experiments require SwanLab offline or online recording")
        import swanlab

        self.sdk = swanlab
        self.folder = folder
        self.mode = mode
        self.run = swanlab.init(
            project=project, workspace=workspace, name=folder.name,
            config=config, log_dir=str(folder / "swanlog"), mode=mode,
            on_online_error="raise", public=False,
        )

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Only finite real metrics can be recorded")
        with (self.folder / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, "metrics": metrics}, allow_nan=False) + "\n")
        self.sdk.log(metrics, step=step)

    def finish(self, *, failed: bool = False) -> None:
        self.sdk.finish(state="crashed" if failed else "success")

    def text(self, step: int, key: str, content: dict) -> None:
        encoded = json.dumps(content, ensure_ascii=False, allow_nan=False)
        with (self.folder / "texts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, "key": key, "text": content},
                                    ensure_ascii=False, allow_nan=False) + "\n")
        self.sdk.log({key: self.sdk.Text(encoded)}, step=step)

    def reference(self) -> dict:
        return {
            "mode": self.mode,
            "cloud_record_verified": False,
            "cloud_url": getattr(self.run, "url", None) if self.mode == "online" else None,
            "log_directory": "swanlog",
        }
