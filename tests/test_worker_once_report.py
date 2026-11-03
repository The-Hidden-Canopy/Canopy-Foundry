from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import neural_forge_worker as worker
from scripts.neural_forge_worker import ONCE_REPORT_SCHEMA, WorkerError, write_once_report


class _Client:
    def __init__(self, job=None, status="succeeded", status_error=False):
        self.job = job
        self.status = status
        self.status_error = status_error
        self.claims = 0

    def __call__(self, config):  # stands in for the HubClient class
        return self

    def claim(self):
        self.claims += 1
        return self.job

    def get_worker_run(self, run_id):
        if self.status_error:
            raise WorkerError("Hub worker status is incomplete")
        return {"run": {"run_id": run_id, "status": self.status}}


JOB = {"run_id": "nf-0123456789", "worker_manifest": {"secret": "never-reported"}}


class OnceReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.report = Path(self.temp.name) / "out" / "once.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def read(self) -> dict:
        return json.loads(self.report.read_text(encoding="utf-8"))

    def test_no_queued_job_is_reported_as_unclaimed(self) -> None:
        write_once_report(self.report, _Client(), None)
        self.assertEqual(
            self.read(),
            {"schema": ONCE_REPORT_SCHEMA, "claimed": False, "run_id": None, "hub_status": None},
        )

    def test_claimed_job_reports_hub_status_without_private_manifest(self) -> None:
        write_once_report(self.report, _Client(status="failed"), JOB)
        report = self.read()
        self.assertEqual((report["claimed"], report["run_id"], report["hub_status"]), (True, JOB["run_id"], "failed"))
        self.assertNotIn("never-reported", self.report.read_text(encoding="utf-8"))
        self.assertFalse(self.report.with_name("once.json.tmp").exists())

    def test_unconfirmed_hub_status_is_null_not_inferred(self) -> None:
        write_once_report(self.report, _Client(status_error=True), JOB)
        self.assertIsNone(self.read()["hub_status"])
        self.assertTrue(self.read()["claimed"])

    def run_main(self, client, *extra: str) -> int:
        argv = ["neural_forge_worker.py", "--hub-url", "https://hub.invalid", "--hub-allowed-host", "hub.invalid", "--worker-token", "t", "--org", "o", *extra]
        with patch.object(sys, "argv", argv), patch.object(worker, "HubClient", client), \
             patch.object(worker, "execute_job") as execute:
            code = worker.main()
        self.executed = execute.call_count
        return code

    def test_main_once_claims_executes_and_reports(self) -> None:
        client = _Client(job=JOB)
        self.assertEqual(self.run_main(client, "--once", "--once-report", str(self.report)), 0)
        self.assertEqual((client.claims, self.executed), (1, 1))
        self.assertEqual(self.read()["hub_status"], "succeeded")

    def test_main_once_without_job_reports_and_exits(self) -> None:
        client = _Client(job=None)
        self.assertEqual(self.run_main(client, "--once", "--once-report", str(self.report)), 0)
        self.assertEqual((client.claims, self.executed), (1, 0))
        self.assertFalse(self.read()["claimed"])

    def test_report_requires_once(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_main(_Client(job=JOB), "--once-report", str(self.report))
        self.assertIn("--once-report requires --once", str(raised.exception))
        self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()
