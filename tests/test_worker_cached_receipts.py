"""Cached receipt reconciliation must preserve current Hub evidence and authority."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import Mock, patch

from scripts import neural_forge_worker as worker
from scripts import local_receipt
from scripts.local_receipt import LocalRunReceiptStore, build_execution_receipt_ref

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cached_receipt_fixture", ROOT / "tests/test_neural_forge_worker.py"
)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def no_network(*_args, **_kwargs):
    raise AssertionError("Network blocked in cached-receipt fixtures")


class PrivateEvidenceAdapter:
    @staticmethod
    def is_request_schema(schema):
        return schema == "opaque-private-schema.v1"

    @staticmethod
    def validate_execution_evidence(evidence, descriptor):
        if set(evidence) != {"evidence_id", "evidence_sha256"}:
            raise ValueError("invalid synthetic private evidence")
        if evidence["evidence_sha256"] != descriptor.get("request_sha256"):
            raise ValueError("descriptor mismatch")
        return dict(evidence)


class GovernanceBoundaries(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(socket, "create_connection", side_effect=no_network))
        self.patches.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
        self.patches.enter_context(patch.object(socket.socket, "connect_ex", side_effect=no_network))
        self.launch = self.patches.enter_context(patch.object(
            worker.subprocess, "Popen", side_effect=AssertionError("Native launch blocked in audit")
        ))
        self.fixture = fixture_module.NeuralForgeWorkerBoundaryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.job = self.fixture.manifest_job()
        self.config = self.fixture.config

    def reject(self, pattern):
        before = self.config.run_root.exists()
        with self.assertRaisesRegex(worker.WorkerError, pattern):
            worker.build_command(self.job, self.config)
        self.assertEqual(self.config.run_root.exists(), before)
        self.launch.assert_not_called()

    def terminal(self, status="succeeded", complete=True, receipt_hash=None):
        identity = worker.receipt_manifest(self.job)
        store = LocalRunReceiptStore(self.config.run_root)
        store.claim(self.job["run_id"], identity)
        store.complete(self.job["run_id"], identity, status=status, metrics_available=(status == "succeeded"))
        reference = build_execution_receipt_ref(identity, status=status, metrics_available=(status == "succeeded"))
        client = Mock(spec=worker.HubClient)
        client.get_worker_run.return_value = {
            "run": {
                "run_id": self.job["run_id"], "status": status,
                "evaluation": {"complete": complete},
                "execution_receipt_sha256": receipt_hash or reference["receipt_sha256"],
            },
            "cancel_requested": False,
        }
        path = self.config.run_root / ".receipts" / (self.job["run_id"] + ".json")
        return client, path, path.read_bytes(), reference

    def test_valid_hub_manifest_builds_bound_request_without_launch(self):
        command, output = worker.build_command(self.job, self.config)
        self.assertEqual(Path(command[0]), self.fixture.binary_path)
        self.assertTrue((output / "native-request.json").is_file())
        self.launch.assert_not_called()

    def test_cross_org_is_blocked_before_output_mutation(self):
        self.job["worker_manifest"]["org"] = "other-org"
        self.reject("organization")

    def test_viewer_role_is_blocked(self):
        self.job["worker_manifest"]["authority"]["role"] = "viewer"
        self.reject("authority role")

    def test_unissued_admin_role_is_blocked(self):
        self.job["worker_manifest"]["authority"]["role"] = "admin"
        self.reject("authority role")

    def test_missing_justification_is_blocked(self):
        self.job["worker_manifest"]["authority"].pop("justification_hash")
        self.reject("authority")

    def test_empty_justification_is_blocked(self):
        self.job["worker_manifest"]["authority"]["justification_hash"] = ""
        self.reject("justification_hash")

    def test_stale_claim_is_blocked(self):
        now = datetime.now(timezone.utc)
        self.job["worker_manifest"].update(
            issued_at=(now - timedelta(minutes=4)).isoformat(),
            claimed_at=(now - timedelta(minutes=3)).isoformat(),
            deadline_at=(now - timedelta(minutes=2)).isoformat(),
        )
        self.reject("deadline has elapsed")

    def test_naive_deadline_is_blocked(self):
        self.job["worker_manifest"]["deadline_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=60)
        ).replace(tzinfo=None).isoformat()
        self.reject("deadline")

    def test_inverted_claim_window_is_blocked(self):
        self.job["worker_manifest"]["issued_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=10)
        ).isoformat()
        self.reject("claim window")

    def test_local_request_cannot_replace_hub_manifest(self):
        self.job.pop("worker_manifest")
        self.reject("worker manifest is required")

    def test_busy_retry_preserves_controller_cancel_and_receipt(self):
        identity = worker.receipt_manifest(self.job)
        store = LocalRunReceiptStore(self.config.run_root)
        store.claim(self.job["run_id"], identity)
        path = self.config.run_root / ".receipts" / (self.job["run_id"] + ".json")
        before = path.read_bytes()
        client = Mock(spec=worker.HubClient)
        client.get_worker_run.return_value = {
            "run": {"status": "cancel_requested", "evaluation": {"complete": False}},
            "cancel_requested": True,
        }
        with self.assertRaisesRegex(worker.WorkerError, "operator reconciliation"):
            worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_matching_cached_cancel_is_idempotent(self):
        client, path, before, _ = self.terminal("cancelled", complete=False)
        worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_cached_cancel_cannot_override_different_hub_terminal(self):
        client, path, before, _ = self.terminal("cancelled", complete=False)
        client.get_worker_run.return_value["run"]["status"] = "failed"
        with self.assertRaisesRegex(worker.WorkerError, "conflicts with Hub"):
            worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_matching_evaluated_cached_success_is_idempotent(self):
        client, path, before, _ = self.terminal()
        worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_cached_success_must_not_accept_incomplete_hub_evaluation(self):
        client, path, before, _ = self.terminal(complete=False)
        with self.assertRaises(worker.WorkerError):
            worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_cached_success_must_not_accept_mismatched_hub_receipt(self):
        client, path, before, _ = self.terminal(receipt_hash="f" * 64)
        with self.assertRaises(worker.WorkerError):
            worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_cached_success_replay_must_include_bound_receipt(self):
        client, path, before, reference = self.terminal()
        client.get_worker_run.return_value["run"]["status"] = "running"
        client.get_worker_run.return_value["run"]["evaluation"]["complete"] = False
        try:
            worker.execute_job(self.job, self.config, client)
        except worker.WorkerError:
            client.update.assert_not_called()
        else:
            client.update.assert_called_once()
            self.assertEqual(client.update.call_args.kwargs.get("execution_receipt"), reference)
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def private_terminal(self):
        identity = worker.receipt_manifest(self.job)
        store = LocalRunReceiptStore(self.config.run_root)
        store.claim(self.job["run_id"], identity)
        descriptor = {"schema_version": "opaque-private-schema.v1", "request_sha256": "b" * 64}
        evidence = {"evidence_id": "synthetic-private-evidence", "evidence_sha256": "b" * 64}
        with patch.object(local_receipt, "load_private_adapter", return_value=PrivateEvidenceAdapter()):
            private = store.prepare_private_evidence(
                self.job["run_id"], identity, native_execution=evidence, native_descriptor=descriptor
            )
        store.complete(self.job["run_id"], identity, status="succeeded", metrics_available=True, **private)
        reference = worker.build_worker_execution_receipt(
            identity, **private, descriptor_sha256=descriptor["request_sha256"]
        )
        self.job["native_execution_descriptor"] = descriptor
        client = Mock(spec=worker.HubClient)
        client.get_worker_run.return_value = {
            "run": {"run_id": self.job["run_id"], "status": "succeeded", "evaluation": {"complete": True},
                    "execution_receipt_sha256": reference["receipt_sha256"]},
            "cancel_requested": False,
        }
        path = self.config.run_root / ".receipts" / (self.job["run_id"] + ".json")
        return client, path, path.read_bytes()

    def assert_reconciliation_refusal(self, client, path, before):
        with self.assertRaises(worker.WorkerError):
            worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_private_descriptor_receipt_matches_without_status_descriptor(self):
        client, path, before = self.private_terminal()
        self.assertNotIn("native_execution_descriptor", client.get_worker_run.return_value)
        worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_private_cached_success_refuses_missing_issued_descriptor(self):
        client, path, before = self.private_terminal()
        self.job.pop("native_execution_descriptor")
        self.assert_reconciliation_refusal(client, path, before)

    def test_private_cached_success_refuses_changed_issued_descriptor(self):
        client, path, before = self.private_terminal()
        self.job["native_execution_descriptor"]["request_sha256"] = "c" * 64
        self.assert_reconciliation_refusal(client, path, before)

    def test_private_cached_success_refuses_missing_descriptor_hash(self):
        client, path, before = self.private_terminal()
        self.job["native_execution_descriptor"].pop("request_sha256")
        self.assert_reconciliation_refusal(client, path, before)

    def test_cached_success_refuses_missing_hub_receipt_hash(self):
        client, path, before, _ = self.terminal()
        client.get_worker_run.return_value["run"].pop("execution_receipt_sha256")
        self.assert_reconciliation_refusal(client, path, before)

    def test_cached_success_refuses_conflicting_controller_cancel(self):
        client, path, before, _ = self.terminal()
        client.get_worker_run.return_value["cancel_requested"] = True
        self.assert_reconciliation_refusal(client, path, before)

    def test_cached_cancel_cannot_publish_over_running_hub_state(self):
        client, path, before, _ = self.terminal("cancelled", complete=False)
        client.get_worker_run.return_value["run"]["status"] = "running"
        self.assert_reconciliation_refusal(client, path, before)

    def test_stale_claim_reentry_only_observes_matching_terminal_evidence(self):
        client, path, before, _ = self.terminal()
        now = datetime.now(timezone.utc)
        self.job["worker_manifest"]["claimed_at"] = (now - timedelta(minutes=3)).isoformat()
        self.job["worker_manifest"]["deadline_at"] = (now - timedelta(minutes=2)).isoformat()
        worker.execute_job(self.job, self.config, client)
        client.update.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        self.launch.assert_not_called()

    def test_cached_success_refuses_missing_metrics_attestation(self):
        client, path, _, _ = self.terminal()
        receipt = json.loads(path.read_text())
        receipt["metrics_available"] = False
        path.write_text(json.dumps(receipt), encoding="utf-8")
        self.assert_reconciliation_refusal(client, path, path.read_bytes())
