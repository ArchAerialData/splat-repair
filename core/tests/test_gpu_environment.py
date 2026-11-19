"""Desktop GPU setup contracts, without loading CUDA, compiling or downloading."""
import builtins
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatcore import deep, gpu


class GpuEnvironmentTests(unittest.TestCase):
    def test_desktop_prefers_local_cuda_and_adds_venv_tools_once(self):
        prefix = os.path.join('test-runtime', 'venv')
        scripts = os.path.join(prefix, 'Scripts')
        local = os.path.join(prefix, 'cuda', 'v12.8')
        cuda_bin = os.path.join(local, 'bin')
        directories = {scripts, cuda_bin}
        with patch.dict(os.environ, {'PATH': 'existing', 'CUDA_PATH': 'other-cuda'}, clear=True), \
                patch.object(sys, 'prefix', prefix), patch.object(os, 'name', 'nt'), \
                patch('os.path.isdir', side_effect=lambda p: p in directories), \
                patch('os.path.isfile', side_effect=lambda p: p == os.path.join(cuda_bin, 'nvcc.exe')), \
                patch.object(gpu, '_add_msvc_to_path') as compiler:
            gpu._ensure_build_env()
            gpu._ensure_build_env()
            self.assertEqual(os.environ['CUDA_HOME'], local)
            entries = os.environ['PATH'].split(os.pathsep)
            self.assertEqual(entries.count(scripts), 1)
            self.assertEqual(entries.count(cuda_bin), 1)
            self.assertIn('existing', entries)
            self.assertEqual(compiler.call_count, 2)

    def test_explicit_cuda_home_is_preserved(self):
        with patch.dict(os.environ, {'PATH': '', 'CUDA_HOME': 'chosen-cuda'}, clear=True), \
                patch.object(os, 'name', 'nt'), patch('os.path.isdir', return_value=False), \
                patch('os.path.isfile', return_value=True), patch.object(gpu, '_add_msvc_to_path'):
            gpu._ensure_build_env()
            self.assertEqual(os.environ['CUDA_HOME'], 'chosen-cuda')

    def test_cuda_path_is_used_when_local_compiler_is_absent(self):
        with patch.dict(os.environ, {'PATH': '', 'CUDA_PATH': 'system-cuda'}, clear=True), \
                patch.object(os, 'name', 'nt'), patch('os.path.isdir', return_value=False), \
                patch('os.path.isfile', return_value=False), patch.object(gpu, '_add_msvc_to_path'):
            gpu._ensure_build_env()
            self.assertEqual(os.environ['CUDA_HOME'], 'system-cuda')

    def test_non_windows_keeps_cuda_discovery_to_torch(self):
        with patch.dict(os.environ, {'PATH': ''}, clear=True), \
                patch.object(os, 'name', 'posix'), patch('os.path.isdir', return_value=True), \
                patch.object(gpu, '_add_msvc_to_path') as compiler:
            gpu._ensure_build_env()
            self.assertNotIn('CUDA_HOME', os.environ)
            compiler.assert_not_called()

    def test_desktop_activates_compiler_headers_and_libraries(self):
        compiler_env = 'VSCMD_ARG_TGT_ARCH=x64\nINCLUDE=vc-headers\nLIB=vc-libraries\nPATH=vc-tools\n'
        replies = [SimpleNamespace(returncode=0, stdout='test-visual-studio\n'),
                   SimpleNamespace(returncode=0, stdout=compiler_env)]
        with patch.dict(os.environ, {'PATH': 'existing'}, clear=True), \
                patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True), \
                patch.object(subprocess, 'CREATE_NO_WINDOW', 0, create=True), \
                patch('subprocess.run', side_effect=replies) as run:
            gpu._add_msvc_to_path()
            query = run.call_args_list[0].args[0]
            self.assertIn('[16.0,18.0)', query, 'Exclude unsupported newer Visual Studio installations')
            self.assertIn('Microsoft.VisualStudio.Component.VC.Tools.x86.x64', query)
            self.assertEqual(os.environ['INCLUDE'], 'vc-headers')
            self.assertEqual(os.environ['LIB'], 'vc-libraries')
            self.assertEqual(os.environ['VSCMD_ARG_TGT_ARCH'], 'x64')
            self.assertEqual(os.environ['DISTUTILS_USE_SDK'], '1')

    def test_compiler_activation_failure_is_not_accepted(self):
        replies = [SimpleNamespace(returncode=0, stdout='test-visual-studio\n'),
                   SimpleNamespace(returncode=1, stdout='')]
        with patch.dict(os.environ, {}, clear=True), \
                patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True), \
                patch.object(subprocess, 'CREATE_NO_WINDOW', 0, create=True), \
                patch('subprocess.run', side_effect=replies), self.assertRaises(RuntimeError):
            gpu._add_msvc_to_path()

    def test_compatible_active_x64_environment_is_preserved(self):
        for version in ('14.29.30133', '14.44.35207'):
            with self.subTest(version=version), \
                    patch.dict(os.environ, {'VSCMD_ARG_TGT_ARCH': 'x64', 'VCToolsVersion': version}, clear=True), \
                    patch('subprocess.run') as run:
                gpu._add_msvc_to_path()
                run.assert_not_called()

    def test_incompatible_active_compiler_environment_fails_clearly(self):
        for arch, version in (('x86', '14.44.35207'), ('arm64', '14.44.35207'),
                              ('x64', '14.50.35717'), ('x64', '14.19.0'),
                              ('x64', ''), ('x64', 'unrecognized')):
            with self.subTest(arch=arch, version=version), \
                    patch.dict(os.environ, {'VSCMD_ARG_TGT_ARCH': arch, 'VCToolsVersion': version}, clear=True), \
                    patch('subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError, 'x64 Visual Studio'):
                    gpu._add_msvc_to_path()
                run.assert_not_called()

    def test_wrong_architecture_returned_by_vcvars_is_rejected(self):
        replies = [SimpleNamespace(returncode=0, stdout='test-visual-studio\n'),
                   SimpleNamespace(returncode=0, stdout='VSCMD_ARG_TGT_ARCH=x86\n')]
        with patch.dict(os.environ, {}, clear=True), \
                patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True), \
                patch.object(subprocess, 'CREATE_NO_WINDOW', 0, create=True), \
                patch('subprocess.run', side_effect=replies), \
                self.assertRaisesRegex(RuntimeError, 'x64 build environment'):
            gpu._add_msvc_to_path()

    def probe_with_mock_runtime(self, explicit_arch=None):
        events = []
        tensor = MagicMock()
        cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 1,
                               get_device_capability=MagicMock(return_value=(8, 9)),
                               get_device_name=lambda _: 'mock GPU')
        torch = SimpleNamespace(cuda=cuda, __version__='test', zeros=lambda *a, **k: tensor,
                                tensor=lambda *a, **k: tensor, full=lambda *a, **k: tensor,
                                ones=lambda *a, **k: tensor, eye=lambda *a, **k: tensor)
        gsplat = SimpleNamespace(rasterization=MagicMock())
        real_import = builtins.__import__

        def import_checked(name, *args, **kwargs):
            if name == 'gsplat':
                self.assertIn('environment', events)
                self.assertEqual(os.environ['CUDA_HOME'], 'prepared-cuda')
                self.assertEqual(os.environ['TORCH_CUDA_ARCH_LIST'], explicit_arch or '8.9')
                events.append('gsplat')
            return real_import(name, *args, **kwargs)

        def prepare():
            events.append('environment')
            os.environ['CUDA_HOME'] = 'prepared-cuda'

        environ = {} if explicit_arch is None else {'TORCH_CUDA_ARCH_LIST': explicit_arch}
        with patch.dict(os.environ, environ, clear=True), patch.dict(gpu._CACHE, {}, clear=True), \
                patch.dict(sys.modules, {'torch': torch, 'gsplat': gsplat}), \
                patch.object(gpu, '_ensure_build_env', side_effect=prepare), \
                patch('builtins.__import__', side_effect=import_checked):
            ok, why = gpu.available()
            self.assertTrue(ok, why)
            self.assertTrue(gpu.available()[0], 'cached status must remain usable')
            self.assertEqual(events.count('environment'), 1)
            cuda.get_device_capability.assert_called_once_with(1)
            gsplat.rasterization.assert_called_once()

    def test_build_environment_and_device_arch_precede_gsplat_import(self):
        self.probe_with_mock_runtime()

    def test_explicit_architecture_override_is_preserved(self):
        self.probe_with_mock_runtime('8.6;8.9+PTX')

    def test_environment_failure_is_reported_before_importing_gsplat(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(gpu._CACHE, {}, clear=True), patch.dict(sys.modules, {'torch': torch}), \
                patch.object(gpu, '_ensure_build_env', side_effect=RuntimeError('toolchain unavailable')):
            ok, why = gpu.available()
            self.assertFalse(ok)
            self.assertIn('toolchain unavailable', why)

    def check_model_cache(self, override=None):
        expected = override or str(Path(deep.__file__).resolve().parents[1] / 'models' / 'torch')
        calls = []

        def create_lama():
            calls.append(os.environ.get('TORCH_HOME'))
            return object()

        environ = {} if override is None else {'TORCH_HOME': override}
        with patch.dict(os.environ, environ, clear=True), patch.dict(deep._LAMA, {}, clear=True), \
                patch.dict(sys.modules, {'simple_lama_inpainting': SimpleNamespace(SimpleLama=create_lama)}):
            self.assertEqual(deep.load_inpainter('lama')[1], 'LaMa')
            self.assertEqual(deep.load_inpainter('lama')[1], 'LaMa')
            self.assertEqual(calls, [expected], 'cache configured before one model construction')

    def test_model_cache_defaults_to_engine_directory(self):
        self.check_model_cache()

    def test_explicit_model_cache_is_preserved(self):
        self.check_model_cache('custom-model-cache')


if __name__ == '__main__':
    unittest.main()
