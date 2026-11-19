"""Installer integrity, archive containment, and partial-failure regressions."""
import hashlib
import io
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_cuda as installer
import setup_deep


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.component = installer.Component("cuda_test", "1", 0, "unused")

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, entries):
        archive = self.root / "component.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name, content in entries:
                output.writestr(name, content)
        return archive

    def test_component_layout_and_licenses(self):
        prefix = self.component.archive_root + "/"
        archive = self.archive([(prefix + "bin/nvcc.exe", b"compiler"),
                                (prefix + "LICENSE", b"license")])
        stage = self.root / "stage"
        stage.mkdir()
        installer._extract(self.component, archive, stage)
        self.assertEqual((stage / "bin/nvcc.exe").read_bytes(), b"compiler")
        self.assertEqual((stage / "licenses/cuda_test/LICENSE").read_bytes(), b"license")

    def test_unsafe_members_rejected_before_any_extraction(self):
        prefix = self.component.archive_root + "/"
        for bad in (prefix + "../escape.txt", "/absolute.txt",
                    prefix + "C:/escape.txt", prefix + "..\\escape.txt", "other/file"):
            with self.subTest(path=bad):
                archive = self.archive([(prefix + "valid.txt", b"ok"), (bad, b"bad")])
                stage = self.root / "stage"
                stage.mkdir(exist_ok=True)
                with self.assertRaises(RuntimeError):
                    installer._extract(self.component, archive, stage)
                self.assertEqual(list(stage.iterdir()), [])
        link = zipfile.ZipInfo(prefix + "link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive = self.archive([(link, b"../../outside")])
        with self.assertRaises(RuntimeError):
            installer._extract(self.component, archive, stage)

    def test_conflicting_components_do_not_overwrite(self):
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "header.h").write_bytes(b"first")
        archive = self.archive([(self.component.archive_root + "/header.h", b"different")])
        with self.assertRaises(RuntimeError):
            installer._extract(self.component, archive, stage)
        self.assertEqual((stage / "header.h").read_bytes(), b"first")

    def test_tampered_download_not_published_or_left_partial(self):
        expected = b"known verified bytes"
        component = installer.Component("test", "1", len(expected), hashlib.sha256(expected).hexdigest())
        cache = self.root / "cache"
        cache.mkdir()
        target = cache / component.filename
        target.write_bytes(b"previous cached file")
        with patch.object(installer.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                installer._download(component, cache)
        self.assertEqual(target.read_bytes(), b"previous cached file")
        self.assertEqual(list(cache.iterdir()), [target])

    def test_incomplete_stage_does_not_become_installed_toolkit(self):
        archive = self.archive([(self.component.archive_root + "/present", b"ok")])
        destination = self.root / "cuda" / "v12.8"
        with patch.object(installer.sys, "platform", "win32"), \
                patch.object(installer, "COMPONENTS", (self.component,)), \
                patch.object(installer, "REQUIRED", ("missing",)), \
                patch.object(installer, "_download", return_value=archive):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                installer.ensure_cuda(destination, self.root / "cache")
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.iterdir()), [])

    def test_success_reuses_matching_install_without_network(self):
        archive = self.archive([(self.component.archive_root + "/present", b"ok")])
        destination = self.root / "cuda" / "v12.8"
        with patch.object(installer.sys, "platform", "win32"), \
                patch.object(installer, "COMPONENTS", (self.component,)), \
                patch.object(installer, "REQUIRED", ("present",)), \
                patch.object(installer, "_download", return_value=archive) as download:
            self.assertEqual(installer.ensure_cuda(destination, self.root / "cache"), destination)
            download.assert_called_once()
            download.reset_mock()
            self.assertEqual(installer.ensure_cuda(destination, self.root / "cache"), destination)
            download.assert_not_called()

    def test_failed_subprocess_aborts_deep_setup(self):
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            setup_deep.run([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(caught.exception.returncode, 7)


if __name__ == "__main__":
    unittest.main()
