import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from clientbench.cli import main
from clientbench.local import apply_local_options, select_device


class LocalDeviceTests(unittest.TestCase):
    def target(self):
        return {"name": "local/test", "backend": "llamacpp", "metadata": {},
                "installation": {"server": "llama-server.exe"}, "server_args": [],
                "context_size": 32768, "gpu_layers": 99}

    def probe(self, output):
        return subprocess.CompletedProcess([], 0, output, "")

    def test_selects_amd_when_nvidia_is_first_and_records_actual_id(self):
        output = "  Vulkan0: NVIDIA GeForce RTX\n  Vulkan1: AMD Radeon(TM) 890M Graphics (16000 MiB)\n"
        targets = [self.target(), self.target()]
        with patch("clientbench.local.subprocess.run", return_value=self.probe(output)) as probe:
            apply_local_options(targets, "amd", 16384, 99)
        probe.assert_called_once()
        for target in targets:
            self.assertEqual(target["metadata"]["selected_device"], "Vulkan1")
            self.assertEqual(target["context_size"], 16384)
            self.assertEqual(target["server_args"], ["--device", "Vulkan1", "--split-mode", "none", "--fit", "off"])

    def test_cuda_only_rejected_instead_of_selecting_nvidia(self):
        with patch("clientbench.local.subprocess.run", return_value=self.probe("  CUDA0: NVIDIA GeForce RTX\n")):
            with self.assertRaisesRegex(ValueError, "found 0"):
                select_device("server", "amd")

    def test_multiple_amd_devices_require_exact_selection(self):
        output = "Vulkan0: AMD Radeon 890M\nVulkan1: AMD Radeon RX 7900\n"
        with patch("clientbench.local.subprocess.run", return_value=self.probe(output)):
            with self.assertRaisesRegex(ValueError, "found 2"):
                select_device("server", "amd")
            self.assertEqual(select_device("server", "Vulkan1")["id"], "Vulkan1")

    def test_probe_timeout_and_failed_exit_are_errors(self):
        with patch("clientbench.local.subprocess.run", side_effect=subprocess.TimeoutExpired([], 30)):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                select_device("server", "amd")
        with patch("clientbench.local.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "failed")):
            with self.assertRaisesRegex(RuntimeError, "enumeration failed"):
                select_device("server", "amd")

    def test_conflicting_config_flags_rejected(self):
        for flags in (["--device=Vulkan1"], ["-sm", "layer"], ["--fit", "on"], ["-ngl", "0"]):
            target = self.target()
            target["server_args"] = flags
            with patch("clientbench.local.select_device", return_value={"id": "Vulkan0", "name": "AMD Radeon", "enumeration": "fixture"}):
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    apply_local_options([target], "amd")

    def test_mock_does_not_probe_and_external_server_cannot_be_reconfigured(self):
        target = self.target()
        del target["installation"]
        with patch("clientbench.local.select_device") as probe:
            apply_local_options([target], "amd", mock=True)
            probe.assert_not_called()
            with self.assertRaisesRegex(ValueError, "managed local server"):
                apply_local_options([target], "amd")

    def test_cloud_only_target_rejects_hardware_options(self):
        with self.assertRaisesRegex(ValueError, "require a llama.cpp target"):
            apply_local_options([{"backend": "responses"}], "amd")

    def test_cli_dry_run_exposes_hardware_configuration(self):
        output = io.StringIO()
        with patch("clientbench.local.select_device") as probe, redirect_stdout(output):
            main(["run", "--target", "local", "--suite", "llm", "--extended", "--mock", "--dry-run",
                  "--local-device", "amd", "--context-size", "16384", "--gpu-layers", "99"])
        probe.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["cases"], 15)
        self.assertEqual(plan["local_hardware"][0]["context_size"], 16384)


if __name__ == "__main__":
    unittest.main()
