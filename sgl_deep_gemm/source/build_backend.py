"""Build the existing SGLang wheel from a full DeepGEMM Git checkout.

Use ``#subdirectory=sgl_deep_gemm/source`` in a pip VCS requirement. The
wheel builder and its package metadata remain the source of truth; this
adapter only connects them to PEP 517 and gives the native wheel a platform tag.
"""

from __future__ import annotations

import email.parser
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

from wheel.wheelfile import WheelFile


class UnsupportedOperation(Exception):
    """The VCS adapter needs the complete repository and its submodules."""


def build_sdist(sdist_directory, config_settings=None):
    raise UnsupportedOperation(
        "Install from a full Git checkout (including submodules) or a binary "
        "wheel. This adapter does not produce standalone source archives."
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path(__file__).resolve().parents[2]
    output = Path(wheel_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # No prepare_metadata hook: pip obtains metadata from this wheel and reuses
    # it for installation, so _C.so is only built once.
    with tempfile.TemporaryDirectory(prefix="sgl-deep-gemm-") as tmp:
        staging = Path(tmp)
        env = dict(os.environ)
        env.update(
            PYTHON_EXE=sys.executable,
            DG_BUILD_DIR=str(staging / "build"),
            DG_DIST_DIR=str(staging / "dist"),
            DG_BUILD_FROM_PIP="1",
        )
        subprocess.run(
            ["bash", str(root / "build_sgl_deep_gemm.sh")],
            cwd=root,
            env=env,
            check=True,
        )
        wheels = list((staging / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one DeepGEMM wheel, found {wheels}")
        return _platform_wheel(wheels[0], output)


def _platform_wheel(wheel_path: Path, output: Path) -> str:
    # TVM FFI is independent of the CPython ABI, but _C.so is platform-specific.
    # The release workflow retags its wheels separately; VCS builds need the
    # correct native-platform tag before pip sees the artifact.
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    tag = f"py3-none-{platform}"
    name = f"{wheel_path.name.rsplit('-', 3)[0]}-{tag}.whl"
    with WheelFile(wheel_path) as src, WheelFile(output / name, "w") as dst:
        for item in src.infolist():
            if item.filename.endswith(".dist-info/RECORD"):
                continue  # WheelFile regenerates hashes for all entries.
            data = src.read(item.filename)
            if item.filename.endswith(".dist-info/WHEEL"):
                metadata = email.parser.Parser().parsestr(data.decode())
                metadata.replace_header("Root-Is-Purelib", "false")
                del metadata["Tag"]
                metadata["Tag"] = tag
                data = metadata.as_bytes()
            dst.writestr(item, data)
    return name
