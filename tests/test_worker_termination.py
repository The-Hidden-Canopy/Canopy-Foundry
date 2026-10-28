import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import neural_forge_worker as worker

@unittest.skipUnless(worker.os.name == "nt", "Windows ownership semantics")
class WindowsTerminationTests(unittest.TestCase):
    def check_unknown(self, child):
        with patch.object(worker.subprocess, "run") as taskkill:
            with self.assertRaises(worker.TerminationUncertain):
                worker.terminate_process_tree(child)
        taskkill.assert_not_called()

    def test_owned_root_stopped_does_not_prove_tree_stopped(self):
        child = Mock(); child.poll.return_value = None; child.wait.return_value = 1
        self.check_unknown(child)
        child.kill.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=3)

    def test_failed_root_stop_stays_unknown(self):
        child = Mock(); child.poll.return_value = None; child.kill.side_effect = OSError("failure")
        child.wait.return_value = 1
        self.check_unknown(child)
        child.kill.assert_called_once_with()

    def test_wait_timeout_stays_unknown(self):
        child = Mock(); child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("child", 3)
        self.check_unknown(child)
        child.kill.assert_called_once_with()

    def test_already_exited_child_never_targets_recycled_pid(self):
        child = Mock(); child.poll.return_value = 0
        self.check_unknown(child)
        child.kill.assert_not_called(); child.wait.assert_not_called()

    def test_unreadable_child_state_does_not_target_other_process(self):
        child = Mock(); child.poll.side_effect = OSError("unknown")
        self.check_unknown(child)
        child.kill.assert_not_called(); child.wait.assert_not_called()

    def test_controller_cancel_does_not_publish_terminal_or_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / "native-request.json").write_text(json.dumps({"device":{"runtime":"cuda"},"expected_terminal_phase":"native_smoke_complete"}), encoding="utf-8")
            child = Mock(); child.poll.return_value = None; child.wait.return_value = 1
            child.stdout = io.BytesIO(); child.stderr = io.BytesIO()
            receipt = Mock(); receipt.claim.return_value = {"decision":"claimed"}
            client = Mock(); client.get_worker_run.return_value = {"run":{"status":"cancel_requested"},"cancel_requested":True}
            job = {"run_id":"nf-test123456", "worker_manifest":{"policy":{"timeout_seconds":60}}}
            config = SimpleNamespace(run_root=out)
            with patch.object(worker, "LocalRunReceiptStore", return_value=receipt), \
                 patch.object(worker, "receipt_manifest", return_value={}), \
                 patch.object(worker, "build_command", return_value=(["fixture"],out)), \
                 patch.object(worker, "manifest_local_request", return_value=({},Path(),Path(),{})), \
                 patch.object(worker.subprocess, "Popen", return_value=child), \
                 patch.object(worker, "publish_terminal_update") as terminal:
                with self.assertRaises(worker.TerminationUncertain):
                    worker.execute_job(job,config,client)
            terminal.assert_not_called(); receipt.complete.assert_not_called()
            self.assertEqual([c.kwargs.get("status") for c in client.update.call_args_list], ["running"])

    def test_postlaunch_poll_error_does_not_publish_terminal_or_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / "native-request.json").write_text(json.dumps({"device":{"runtime":"cuda"},"expected_terminal_phase":"native_smoke_complete"}), encoding="utf-8")
            child = Mock(); child.poll.side_effect = OSError("poll failed"); child.wait.return_value = 1
            child.stdout = io.BytesIO(); child.stderr = io.BytesIO()
            receipt = Mock(); receipt.claim.return_value = {"decision":"claimed"}
            client = Mock(); client.get_worker_run.return_value = {"run":{"status":"cancel_requested"},"cancel_requested":True}
            job = {"run_id":"nf-test123456", "worker_manifest":{"policy":{"timeout_seconds":60}}}
            config = SimpleNamespace(run_root=out)
            with patch.object(worker, "LocalRunReceiptStore", return_value=receipt), \
                 patch.object(worker, "receipt_manifest", return_value={}), \
                 patch.object(worker, "build_command", return_value=(["fixture"],out)), \
                 patch.object(worker, "manifest_local_request", return_value=({},Path(),Path(),{})), \
                 patch.object(worker.subprocess, "Popen", return_value=child), \
                 patch.object(worker, "publish_terminal_update") as terminal:
                with self.assertRaises(worker.TerminationUncertain):
                    worker.execute_job(job,config,client)
            terminal.assert_not_called(); receipt.complete.assert_not_called()
            self.assertEqual([c.kwargs.get("status") for c in client.update.call_args_list], ["running"])

    def test_late_controller_cancel_cannot_become_a_success_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / "native-request.json").write_text(json.dumps({"device":{"runtime":"cuda"},"expected_terminal_phase":"native_smoke_complete"}), encoding="utf-8")
            child = Mock(); child.poll.return_value = 0; child.wait.return_value = 0; child.returncode = 0
            child.stdout = io.BytesIO(b'{"type":"step"}\n'); child.stderr = io.BytesIO()
            receipt = Mock(); receipt.claim.return_value = {"decision":"claimed"}
            client = Mock(); client.update.side_effect = [None, worker.WorkerError("hub state changed")]
            client.get_worker_run.return_value = {"run":{"status":"cancel_requested"},"cancel_requested":True}
            job = {"run_id":"nf-test123456", "worker_manifest":{"policy":{"timeout_seconds":60}}}
            config = SimpleNamespace(run_root=out,max_output_bytes=100000,max_output_files=10)
            with patch.object(worker, "LocalRunReceiptStore", return_value=receipt), \
                 patch.object(worker, "receipt_manifest", return_value={}), \
                 patch.object(worker, "build_command", return_value=(["fixture"],out)), \
                 patch.object(worker, "manifest_local_request", return_value=({},Path(),Path(),{})), \
                 patch.object(worker.subprocess, "Popen", return_value=child), \
                 patch.object(worker, "output_usage", return_value=(0,0)), \
                 patch.object(worker, "completed_metrics") as metrics, \
                 patch.object(worker, "publish_terminal_update") as terminal:
                with self.assertRaises(worker.TerminationUncertain):
                    worker.execute_job(job,config,client)
            metrics.assert_not_called(); terminal.assert_not_called(); receipt.complete.assert_not_called()

    def test_unfinished_output_reader_never_publishes_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "native-request.json").write_text(json.dumps({
                "device": {"runtime": "cuda"},
                "expected_terminal_phase": "native_smoke_complete"}), encoding="utf-8")
            child = Mock()
            child.poll.return_value = 0
            child.wait.return_value = 0
            child.returncode = 0
            child.stdout = io.BytesIO()
            child.stderr = io.BytesIO()
            reader = Mock()
            reader.is_alive.return_value = True
            receipt = Mock()
            receipt.claim.return_value = {"decision": "claimed"}
            client = Mock()
            job = {"run_id": "nf-reader123456", "worker_manifest": {"policy": {"timeout_seconds": 60}}}
            config = SimpleNamespace(run_root=out, max_output_bytes=100000, max_output_files=10)
            with patch.object(worker, "LocalRunReceiptStore", return_value=receipt), \
                 patch.object(worker, "receipt_manifest", return_value={}), \
                 patch.object(worker, "build_command", return_value=(["fixture"], out)), \
                 patch.object(worker, "manifest_local_request", return_value=({}, Path(), Path(), {})), \
                 patch.object(worker.subprocess, "Popen", return_value=child), \
                 patch.object(worker.threading, "Thread", return_value=reader), \
                 patch.object(worker, "output_usage", return_value=(0, 0)), \
                 patch.object(worker, "publish_terminal_update") as terminal:
                with self.assertRaises(worker.TerminationUncertain):
                    worker.execute_job(job, config, client)
            terminal.assert_not_called()
            receipt.complete.assert_not_called()
            self.assertFalse(child.stdout.closed)
            self.assertFalse(child.stderr.closed)
            self.assertEqual([c.kwargs.get("status") for c in client.update.call_args_list], ["running"])

    def test_retry_of_unresolved_run_does_not_override_controller_cancel(self):
        receipt = Mock(); receipt.claim.return_value = {"decision":"busy", "status":"in_progress"}
        client = Mock(); client.get_worker_run.return_value = {"run":{"status":"cancel_requested"},"cancel_requested":True}
        with patch.object(worker, "LocalRunReceiptStore", return_value=receipt), \
             patch.object(worker, "receipt_manifest", return_value={}):
            with self.assertRaisesRegex(worker.WorkerError, "operator reconciliation"):
                worker.execute_job({"run_id":"nf-test123456"},SimpleNamespace(run_root=Path("unused")),client)
        client.update.assert_not_called(); receipt.complete.assert_not_called()

if __name__ == "__main__": unittest.main()
