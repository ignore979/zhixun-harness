"""Build and verify the official SDK image locally. Never submits a platform evaluation."""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime

from fetch_offline_python_base import LAYOUT, verify_layout
from prepare_official_image import CONTEXT, ROOT
from validate_image_archive import validate_archive

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from check_official_image import smoke

OUTPUT = ROOT / "指导/交付/正式SDK镜像-20261117"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    status = OUTPUT / f"build-{stamp}.json"
    log = OUTPUT / f"build-{stamp}.log"
    tag = "zhixun-official-sdk:" + stamp.lower()
    result = {
        "status": "running",
        "phase": "context_check",
        "tag": tag,
        "archive": None,
        "log": str(log),
        "sdk_protocol": "contest.harness.workspace.v2",
        "platform_evaluation_started": False,
        "real_platform_sampler_verified": False,
    }

    def save(phase):
        result["phase"] = phase
        result["updated_at"] = datetime.now(UTC).isoformat()
        status.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT / "latest_build_status.json").write_text(
            json.dumps(
                {"status_file": str(status), "status": result["status"], "phase": phase},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def docker(args):
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMMAND " + json.dumps(["docker", *args], ensure_ascii=False) + "\n")
            process = subprocess.Popen(
                ["docker", *args],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            lines = []
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
                lines.append(line)
            if process.wait():
                raise RuntimeError(f"Docker command failed; see {log}")
            return "".join(lines).strip()

    try:
        files = json.loads((CONTEXT / "CONTEXT_SHA256.json").read_text(encoding="utf-8"))
        actual = {p.relative_to(CONTEXT).as_posix() for p in CONTEXT.rglob("*") if p.is_file()}
        if actual != set(files) | {"CONTEXT_SHA256.json"}:
            raise ValueError("Unexpected files in official context")
        for name, expected in files.items():
            if sha(CONTEXT / name) != expected:
                raise ValueError("Context changed: " + name)
        base = verify_layout(LAYOUT)
        save("docker_engine_check")
        result["server"] = json.loads(docker(["version", "--format", "{{json .Server}}"]))
        save("build")
        named = "python_base=oci-layout://./" + LAYOUT.relative_to(ROOT).as_posix()
        named += "@" + base["manifest_digest"]
        docker(
            [
                "buildx",
                "build",
                "--load",
                "--platform",
                "linux/amd64",
                "--provenance=false",
                "--network",
                "none",
                "--pull=false",
                "--build-context",
                named,
                "--progress",
                "plain",
                "--tag",
                tag,
                str(CONTEXT),
            ]
        )
        save("describe")
        described = json.loads(
            docker(["run", "--rm", "--read-only", "--network", "none", tag, "describe"])
        )
        if (
            described["protocol"] != "contest.harness.workspace.v2"
            or described["model_access"] != "host_gateway_rpc"
        ):
            raise ValueError("Official describe contract mismatch")
        result["describe"] = described
        save("readonly_sdk_round_and_prediction_checks")
        result["smoke"] = smoke(tag, ROOT, OUTPUT / ("smoke-" + stamp))
        save("docker_save")
        temporary = OUTPUT / f"zhixun-official-linux-amd64-{stamp}.building.tar"
        archive = OUTPUT / f"zhixun-official-linux-amd64-{stamp}.tar"
        docker(["image", "save", "--platform", "linux/amd64", "--output", str(temporary), tag])
        save("archive_validation")
        checked = validate_archive(temporary)
        temporary.rename(archive)
        checked["archive"] = str(archive)
        result.update(status="completed", archive=str(archive), archive_validation=checked)
        save("completed")
        (OUTPUT / (archive.name + ".sha256")).write_text(
            checked["sha256"] + "  " + archive.name + "\n", encoding="utf-8"
        )
        print("\nOFFICIAL SDK IMAGE TAR READY: " + str(archive))
        print(
            "Local SDK checks passed. Bound Sampler and hidden scores await platform evaluation."
        )
        return 0
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        save(result["phase"])
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
