import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from zhixun.cloud import compare_config, expected_points, parse_cloud_url, verify_cloud_runs
from zhixun.data import ROOT


class CloudTests(unittest.TestCase):
    def fixture(self, directory):
        source = sorted((ROOT / "artifacts").glob("run-A-*/run.json"))[0].parent
        folder = Path(directory) / "artifacts" / source.name
        folder.mkdir(parents=True)
        for filename in ["run.json", "registration.json", "history.jsonl"]:
            (folder / filename).write_bytes((source / filename).read_bytes())
        registration = json.loads((folder / "registration.json").read_text(encoding="utf-8"))
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        response = {"list": [
            {"key": key, "metrics": [{"step": step, "value": value}
                                     for step, value in values.items()]}
            for key, values in expected_points(history).items()
        ]}
        run = SimpleNamespace(
            name=source.name,
            profile={"config": {key: {"value": value, "desc": "", "sort": i}
                                for i, (key, value) in enumerate(registration.items())}},
            metrics=lambda **kwargs: response,
        )
        return run, response, source.name

    def temp(self):
        folder = ROOT / "指导/临时/cloud-tests"
        folder.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=folder)

    def test_wrapped_cloud_config_and_all_scalar_points_match(self):
        with self.temp() as directory:
            run, _, name = self.fixture(directory)
            result = verify_cloud_runs(["https://swanlab.cn/@unit/project/runs/example"],
                                       root=Path(directory), api=SimpleNamespace(run=lambda _: run))
            self.assertTrue(result["all_current_completed_runs_verified"])
            self.assertEqual(result["verified_experiments"][name]["scalar_points_checked"], 28)

    def test_altered_cloud_config_is_not_verified(self):
        with self.temp() as directory:
            run, _, _ = self.fixture(directory)
            run.profile = copy.deepcopy(run.profile)
            run.profile["config"]["epochs"]["value"] += 1
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                verify_cloud_runs(["https://swanlab.cn/@unit/project/runs/example"],
                                  root=Path(directory), api=SimpleNamespace(run=lambda _: run))
            self.assertFalse((Path(directory) / "evidence/cloud_records.json").exists())

    def test_missing_or_changed_cloud_scalar_is_not_verified(self):
        with self.temp() as directory:
            run, response, _ = self.fixture(directory)
            response["list"][0]["metrics"][0]["value"] += 1
            with self.assertRaisesRegex(ValueError, "scalar differs"):
                verify_cloud_runs(["https://swanlab.cn/@unit/project/runs/example"],
                                  root=Path(directory), api=SimpleNamespace(run=lambda _: run))

    def test_foreign_hosts_and_credentials_in_urls_are_rejected(self):
        for url in ["http://swanlab.cn/@unit/project/runs/id",
                    "https://example.com/@unit/project/runs/id",
                    "https://key@swanlab.cn/@unit/project/runs/id"]:
            with self.assertRaises(ValueError):
                parse_cloud_url(url)

    def test_observed_sdk_null_serialization_is_explicit_and_missing_keys_still_fail(self):
        self.assertEqual(compare_config({"monetary_limit": None}, {"monetary_limit": {}}),
                         ["monetary_limit"])
        with self.assertRaises(ValueError):
            compare_config({"monetary_limit": None}, {})
        with self.assertRaises(ValueError):
            compare_config({"epochs": 3}, {"epochs": {}})

    def test_nested_null_compatibility_does_not_allow_missing_or_extra_config_fields(self):
        self.assertEqual(compare_config({"upgrade": {"cap": None}}, {"upgrade": {"cap": {}}}),
                         ["upgrade.cap"])
        with self.assertRaises(ValueError):
            compare_config({"upgrade": {"cap": None}}, {"upgrade": {}})
        with self.assertRaises(ValueError):
            compare_config({"upgrade": {"cap": None}}, {"upgrade": {"cap": {}, "extra": 1}})
