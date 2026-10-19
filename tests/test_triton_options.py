"""Declarative compiler options apply to precompilation and execution alike."""

import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from hydroforge.kernels import KernelSpec, TritonKernel, TritonSequence
from hydroforge.kernels.toolchain import cuda_toolkit
from hydroforge.platform.backend import TRITON


class TritonCompilerOptionsTests(unittest.TestCase):
    def test_nested_compiler_data_is_detached_and_immutable(self):
        class Kernel:
            arg_names = ("n", "BLOCK_SIZE")

        libraries = {"libdevice": "/selected/libdevice.bc"}
        dimensions = [1, 1, 1]
        declaration = TritonKernel(
            Kernel(),
            compiler_options={"extern_libs": libraries, "cluster_dims": dimensions},
        )
        libraries["libdevice"] = "/different/libdevice.bc"
        dimensions[0] = 2
        self.assertEqual(
            declaration.compiler_options["extern_libs"]["libdevice"],
            "/selected/libdevice.bc",
        )
        self.assertEqual(declaration.compiler_options["cluster_dims"], (1, 1, 1))
        with self.assertRaises(TypeError):
            declaration.compiler_options["extern_libs"]["libdevice"] = "/different.bc"

    def test_precompilation_and_execution_share_immutable_options(self):
        calls = []

        class Kernel:
            arg_names = ("n", "BLOCK_SIZE")

            def warmup(self, **kwargs):
                calls.append(("warmup", kwargs))

            def __getitem__(self, grid):
                def run(**kwargs):
                    calls.append((grid, kwargs))

                return run

        spec = KernelSpec(name="probe", size="n", parameters={"n": "index"})
        configured = {"enable_fp_fusion": False, "enable_reflect_ftz": False}
        declarations = (
            TritonKernel(Kernel(), compiler_options=configured),
            TritonSequence(steps=((Kernel(), "n"),), compiler_options=configured),
        )
        configured["enable_fp_fusion"] = True
        for declaration in declarations:
            with self.subTest(declaration=type(declaration).__name__):
                calls.clear()
                with self.assertRaises(TypeError):
                    declaration.compiler_options["enable_fp_fusion"] = True
                call = declaration.build(spec, TRITON).call({"n": 1})
                with (
                    patch(
                        "hydroforge.kernels.triton.proven_triton_device",
                        return_value=SimpleNamespace(active=nullcontext),
                    ),
                    patch("hydroforge.kernels.triton.triton_call_device"),
                    patch(
                        "hydroforge.kernels.triton.launch_options",
                        return_value={
                            "enable_fp_fusion": True,
                            "enable_reflect_ftz": True,
                        },
                    ),
                ):
                    for request in call.requests():
                        request.build()
                    call.compile()()
                self.assertEqual(len(calls), 2)
                for _, options in calls:
                    self.assertIs(options["enable_fp_fusion"], False)
                    self.assertIs(options["enable_reflect_ftz"], False)
                    self.assertEqual(options["n"], 1)

    def test_options_cannot_replace_the_canonical_abi_or_launch_geometry(self):
        class Kernel:
            arg_names = ("n", "BLOCK_SIZE")

        spec = KernelSpec(name="probe", size="n", parameters={"n": "index"})
        for name in ("n", "BLOCK_SIZE", "grid", "warmup", "stream"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "cannot replace launch arguments"),
            ):
                TritonKernel(Kernel(), compiler_options={name: 2}).build(spec, TRITON)


class CudaToolkitTests(unittest.TestCase):
    def tearDown(self):
        cuda_toolkit.include_roots.cache_clear()
        cuda_toolkit.libdevice_path.cache_clear()

    def test_explicit_toolkit_is_respected_without_modifying_torch_or_environment(self):
        from torch.utils import cpp_extension

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "nvvm/libdevice/libdevice.10.bc"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"toolkit libdevice")
            with patch.dict(os.environ, {"CUDA_HOME": str(root)}):
                before = dict(os.environ)
                previous = cpp_extension.CUDA_HOME
                cuda_toolkit.include_roots.cache_clear()
                self.assertEqual(cuda_toolkit.include_roots(), (root / "include",))
                self.assertEqual(cuda_toolkit.libdevice_path(), str(library))
                self.assertEqual(dict(os.environ), before)
                self.assertIs(cpp_extension.CUDA_HOME, previous)

    def test_pip_toolkit_is_a_read_only_fallback(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "include").mkdir()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("torch.utils.cpp_extension.CUDA_HOME", None),
                patch("torch.version.cuda", "13.2"),
                patch(
                    "hydroforge.kernels.toolchain.cuda_toolkit.importlib.util.find_spec",
                    side_effect=lambda name: (
                        SimpleNamespace(submodule_search_locations=[str(root)])
                        if name == "nvidia.cu13"
                        else None
                    ),
                ),
            ):
                cuda_toolkit.include_roots.cache_clear()
                self.assertEqual(cuda_toolkit.include_roots(), (root / "include",))
                self.assertNotIn("CUDA_HOME", os.environ)
                from torch.utils import cpp_extension

                self.assertIsNone(cpp_extension.CUDA_HOME)

    def test_cuda12_wheels_do_not_use_an_installed_cuda13_toolkit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("cu13", "cuda_runtime", "cuda_cccl"):
                (root / name / "include").mkdir(parents=True)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("torch.utils.cpp_extension.CUDA_HOME", None),
                patch("torch.version.cuda", "12.8"),
                patch(
                    "hydroforge.kernels.toolchain.cuda_toolkit.importlib.util.find_spec",
                    side_effect=lambda name: SimpleNamespace(
                        submodule_search_locations=[str(root / name.split(".")[-1])]
                    ),
                ),
            ):
                cuda_toolkit.include_roots.cache_clear()
                self.assertEqual(
                    cuda_toolkit.include_roots(),
                    (root / "cuda_runtime/include", root / "cuda_cccl/include"),
                )


if __name__ == "__main__":
    unittest.main()
